import traceback

import torch.nn.functional as F
import re
from ultralytics.utils.nms import TorchNMS
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from PIL import Image
from peft import PeftModel
import yaml
from ultralytics.utils.plotting import Annotator
from inference.paligemma_coco_benchmark import PaliGemmaCocoPredictor
from pathlib import Path
from tqdm import tqdm
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import json
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.utils.ops import xyxy2ltwh, ltwh2xyxy
import numpy as np
import datetime
import os
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from peft import PeftModel


def get_synced_timestamp():
    """Generate timestamp on main process and broadcast to all ranks."""
    distributed_state = PartialState(timeout=datetime.timedelta(minutes=600))

    if distributed_state.is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    else:
        timestamp = None

    # Broadcast from rank 0 to all ranks
    timestamp_list = [timestamp]
    broadcast_object_list(timestamp_list, from_process=0)

    return timestamp_list[0]


if "RUN_TIMESTAMP" not in os.environ:
    os.environ["RUN_TIMESTAMP"] = get_synced_timestamp()
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"


class PositionBasedTokenFilter(LogitsProcessor):
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:

        current_position = input_ids.shape[-1] - self.prefix_len
        mask = torch.full_like(scores, float("-inf"))
        if current_position == 255:
            mask[:, 1] = 0.0
        elif (current_position + 1) % 5 != 0:
            mask[:, 256000:257024] = 0.0
        else:
            mask[:, 7:88] = 0.0

        scores = scores + mask

        return scores


class RestrictedPredictor:
    def __init__(self, model, processor, partial_state: PartialState | None = None):
        self.model = model
        self.processor = processor
        self.prompt = "<image>detect all classes\n"
        self.distributed_state = partial_state or PartialState()
        self.new_tokens = 256
        self.noise_token = 87

        with open(to_absolute_path("../assets/coco_id2class.yaml"), "r") as file:
            id2class = yaml.safe_load(file)
        self.coco_id2class = id2class["classes"]

    def run_generation_batch(self, images: list[Image]):

        if not images:
            return []

        model_inputs = (
            self.processor(
                text=[self.prompt] * len(images), images=images, return_tensors="pt"
            )
            .to(torch.bfloat16)
            .to(self.distributed_state.device)
        )

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=self.new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=LogitsProcessorList(
                    [PositionBasedTokenFilter(input_len)]
                ),
            )

        results = [[] for _ in range(len(images))]

        for img_id in range(len(images)):
            sequence = (
                generation.sequences[img_id, input_len:].detach().cpu().numpy().tolist()
            )

            scores = torch.stack(
                [
                    generation.scores[pos_id][img_id].detach().cpu()
                    for pos_id in range(self.new_tokens)
                ]
            )
            probabilities = F.softmax(scores, dim=-1)

            assert len(sequence) == len(probabilities) == self.new_tokens

            for token, probas in zip(sequence, probabilities):
                p_top, tokens_top = torch.topk(probas, 2)

                assert torch.isclose(probas[token], probas[tokens_top[0]])

                if self.noise_token == tokens_top[0]:
                    results[img_id].append(
                        (
                            self.processor.tokenizer.decode(tokens_top[1].item()),
                            p_top[1].item(),
                        )
                    )
                else:
                    results[img_id].append(
                        (
                            self.processor.tokenizer.decode(tokens_top[0].item()),
                            p_top[0].item(),
                        )
                    )

        return RestrictedPredictor._parse_predictions(
            results, [img.size for img in images]
        )

    @staticmethod
    def _extract_validate_bbox(
        y1: int, x1: int, y2: int, x2: int, original_image_shapes: tuple
    ):
        width, height = original_image_shapes
        y1, x1, y2, x2 = map(lambda x: x / 1024.0, [y1, x1, y2, x2])
        y1, y2 = y1 * height, y2 * height
        x1, x2 = x1 * width, x2 * width
        is_valid_bbox = True

        if (y1 >= y2) or (x1 >= x2):
            is_valid_bbox = False

        return (x1, y1, x2, y2), is_valid_bbox

    @staticmethod
    def _parse_predictions(pred_seqs: list[list[tuple]], input_img_shapes: list[tuple]):
        results = []
        for seq, img_sizes in zip(pred_seqs, input_img_shapes):
            result = []
            block = []
            for token, score in seq:
                if token == "<eos>":
                    continue

                block.append(int(re.findall(r"\d+", token)[0]))

                if "loc" in token:
                    continue

                y1_d, x1_d, y2_d, x2_d = block[:4]
                class_ = block[4]

                (x1, y1, x2, y2), is_valid_bbox = (
                    RestrictedPredictor._extract_validate_bbox(
                        y1_d, x1_d, y2_d, x2_d, img_sizes
                    )
                )

                if is_valid_bbox:
                    result.append((x1, y1, x2, y2, score, class_))

                block.clear()

            results.append(result)

        return results

    @staticmethod
    def apply_nms_to_predictions(
        detections, iou_threshold=0.7, conf_threshold=0.25, class_agnostic=False
    ):

        detections = [det for det in detections if det[4] > conf_threshold]
        if not detections:
            return []

        categories = set(det[-1] for det in detections)
        category2id = {cat: i for i, cat in enumerate(categories)}

        boxes = []
        scores = []
        idxs = []

        for det in detections:
            boxes.append([*det[:4]])
            scores.append(det[4])

            if class_agnostic:
                idxs.append(0)
            else:
                idxs.append(category2id[det[5]])

        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        idxs = torch.tensor(idxs)

        result_idx = (
            TorchNMS.batched_nms(boxes, scores, idxs, iou_threshold=iou_threshold)
            .numpy()
            .tolist()
        )

        return [detections[i] for i in result_idx]

    @staticmethod
    def annotate_bbox(item: dict):
        item["annotator"].box_label(
            (item["x1"], item["y1"], item["x2"], item["y2"]),
            item["label"] + f"_{item['score']:.3f}",
            color=(255, 0, 0),
            txt_color=(0, 0, 0),
        )

    def detect_and_show_demo(
        self,
        image: Image,
        line_width: int = 1,
        font_size: int = 11,
    ):
        r = self.run_generation_batch([image])
        annotator = Annotator(
            image,
            pil=True,
            line_width=line_width,
            font_size=font_size,
        )

        for x1, y1, x2, y2, score, cls in RestrictedPredictor.apply_nms_to_predictions(
            r[0]
        ):
            annotator.box_label(
                (x1, y1, x2, y2),
                f"{self.coco_id2class [cls]}_{score:.3f}",
                color=(255, 0, 0),
                txt_color=(0, 0, 0),
            )

        return image

    def _extract_detection_results_in_benchmark_format(
        self, predictions: list, valid_image_ids: list[int]
    ):

        if not predictions:
            return []

        results = []
        for image_predictions, image_id in zip(predictions, valid_image_ids):
            cat_ids = [coco80_to_coco91_class()[x[5]] for x in image_predictions]
            boxes = xyxy2ltwh(
                np.array([[x[0], x[1], x[2], x[3]] for x in image_predictions])
            )
            scores = [x[4] for x in image_predictions]

            for bbox, cat, score in zip(boxes, cat_ids, scores):
                results.append(
                    {
                        "image_id": image_id,
                        "file_name": f"{image_id:012d}.jpg",
                        "category_id": cat,
                        "bbox": bbox.tolist(),
                        "score": score,
                    }
                )

        return results

    def _save_local_predictions_rank(
        self, my_predictions: list, predictions_folder: Path
    ):
        os.makedirs(predictions_folder, exist_ok=True)
        rank_file_path = (
            predictions_folder
            / f"predictions_rank_{self.distributed_state.process_index}.json"
        )
        with open(rank_file_path, "w") as f:
            json.dump(my_predictions, f)

    def _collect_all_json_and_save(
        self, predictions_folder: Path, predictions_filename: Path | str
    ):
        all_predictions = []
        for i in range(self.distributed_state.num_processes):
            with open(predictions_folder / f"predictions_rank_{i}.json", "r") as f:
                all_predictions.extend(json.load(f))

        with open(predictions_folder / predictions_filename, "w") as f:
            json.dump(all_predictions, f)

    def process_directory(
        self,
        path_to_images: Path | str,
        json_annotations: Path | str,
        predictions_folder=Path("predictions"),
        predictions_filename="all_predictions.json",
        batch_size=5,
    ):
        """
        Spits images between ranks and runs predictions.
        Images are expected to be in the RGB mode.
        """

        my_predictions = []
        failed_images = []
        images_ids_list = PaliGemmaCocoPredictor._parse_valid_image_ids(
            json_annotations
        )

        with self.distributed_state.split_between_processes(
            images_ids_list
        ) as images_ids_rank:
            # cnt = 0
            print(
                f"Rank {self.distributed_state.process_index} processes {len(images_ids_rank)} images: {images_ids_rank[:10]}..."
            )
            for i in tqdm(
                range(0, len(images_ids_rank), batch_size),
                "Processing batches...",
                disable=not self.distributed_state.is_main_process,
            ):
                # cnt += 1
                # if cnt >= 5:
                #     break
                images_ids = images_ids_rank[i : i + batch_size]
                images_batch = []
                valid_image_ids = []
                for image_id in images_ids:
                    try:
                        path_to_image = path_to_images / f"{image_id:012d}.jpg"

                        image = Image.open(path_to_image)
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        images_batch.append(image)
                        valid_image_ids.append(image_id)
                    except Exception as e:
                        print(f"Error opening image id: {image_id}: {e}")
                        failed_images.append(image_id)

                if not images_batch:
                    print(
                        f"Empty batch detected at rank {self.distributed_state.process_index}"
                    )
                    continue

                try:
                    results = self.run_generation_batch(images_batch)
                    my_predictions.extend(
                        self._extract_detection_results_in_benchmark_format(
                            results, valid_image_ids
                        )
                    )
                except Exception as exc:
                    tb_str = traceback.format_exc()
                    print(
                        f"Error processing batch: {image_id} at rank"
                        f" {self.distributed_state.process_index}: {exc}\n\n{tb_str}"
                    )

                finally:
                    for img in images_batch:
                        if img:
                            img.close()

        self._save_local_predictions_rank(my_predictions, predictions_folder)

        if failed_images:
            print(
                f"Rank {self.distributed_state.local_process_index} "
                f"failed to process {len(failed_images)} image(s): {failed_images}",
            )

        self.distributed_state.wait_for_everyone()

        if self.distributed_state.is_main_process:
            self._collect_all_json_and_save(predictions_folder, predictions_filename)

            PaliGemmaCocoPredictor.calculate_coco_metrics(
                json_annotations, predictions_folder / predictions_filename
            )

        self.distributed_state.wait_for_everyone()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="paligemma_structured_coco_benchmark",
)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    distributed_state = PartialState(timeout=datetime.timedelta(minutes=600))

    model_id = cfg.model.base
    adapter_dir = cfg.model.adapter
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16
    )
    processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
    model = (
        PeftModel.from_pretrained(base_model, adapter_dir)
        .to(distributed_state.device)
        .eval()
    )

    try:
        pg_predictor = RestrictedPredictor(
            model,
            processor,
            distributed_state,
        )

        path_to_images = Path(cfg.coco.path_to_images)
        json_annotations = Path(cfg.coco.json_annotations)
        pg_predictor.process_directory(
            path_to_images, json_annotations, batch_size=cfg.processing.batch_size
        )

    finally:
        distributed_state.wait_for_everyone()
        distributed_state.destroy_process_group()


if __name__ == "__main__":
    main()
