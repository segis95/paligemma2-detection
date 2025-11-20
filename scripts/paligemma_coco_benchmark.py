import json
import re
from pathlib import Path
import traceback
import os
import datetime
import warnings
from functools import partial
from typing import Callable
import tempfile
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.utils.ops import xyxy2ltwh, ltwh2xyxy
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.nms import TorchNMS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = get_logger(__name__)


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


class PaliGemmaCocoPredictor:
    def __init__(
        self,
        model_path: Path,
        distributed_state: PartialState,
        max_new_tokens=256,
        do_sample=False,
        classes_per_call=5,
        batch_size=80,
        debug_verbose=False,
        score_aggregation_mode="product_norm",
        max_input_length=None,
        use_flash_attention=False,
    ):
        self.distributed_state = distributed_state
        self.model, self.processor = self._load_model(model_path, use_flash_attention)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.classes_per_call = classes_per_call
        self.batch_size = batch_size
        self.verbose = debug_verbose
        self.score_aggregation_mode = score_aggregation_mode
        self.pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>([^;<]+)"
        self.loc_tokens_expected_number = 4

        with open(to_absolute_path("coco_classes2id.yaml"), "r") as file:
            data = yaml.safe_load(file)

        self.coco_class2id = data["classes"]
        self.coco_classes = list(self.coco_class2id.keys())
        self.max_input_length = max_input_length

    def _load_model(self, model_path: Path | str, use_flash_attention=False):
        logger.info(f"Loading model from {model_path}.", main_process_only=True)
        attn_implementation_ = "flash_attention_2" if use_flash_attention else "sdpa"
        logger.info(
            f"Using {attn_implementation_} as attention implementation.",
            main_process_only=True,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn_implementation_
        ).to(self.distributed_state.device)
        logger.info("Compiling model with torch.compile.", main_process_only=True)
        model = torch.compile(model, dynamic=True)
        model.eval()
        processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)

        return model, processor

    @staticmethod
    def _extract_validate_bbox(
        y1: str, x1: str, y2: str, x2: str, original_image_shapes: tuple
    ):
        width, height = original_image_shapes
        y1, x1, y2, x2 = map(lambda x: int(x) / 1024.0, [y1, x1, y2, x2])
        y1, y2 = y1 * height, y2 * height
        x1, x2 = x1 * width, x2 * width
        is_valid_bbox = True

        if (y1 >= y2) or (x1 >= x2):
            is_valid_bbox = False

        return (y1, x1, y2, x2), is_valid_bbox

    @staticmethod
    def _parse_label(label: str):
        label_clean = label.strip().lower()
        label_clean = re.sub(r"[^a-z]", " ", label_clean)
        label_clean = list(
            filter(lambda x: len(x) > 0, label_clean.split(" "))
        )  # additional spaces are tolerated
        label_clean = " ".join(label_clean)
        return label_clean

    def _decode_string_extract_scores(
        self, new_tokens_generated: np.array, probabilities: np.array
    ):
        decoded_strings = []
        decoded_scores = []
        string_position = 0
        kept_token_id = 0
        string_pos2kept_token_id = {}

        if self.verbose and not all(
            np.allclose(np.max(probas), probas[token])
            for token, probas in zip(new_tokens_generated, probabilities)
        ):
            logger.debug(
                "Non argmax generation detected\n"
                f"Generated tokens: {new_tokens_generated}\n"
                f"Argmax values: {np.argmax(probabilities, axis=-1)}"
            )

        for token, probas in zip(new_tokens_generated, probabilities):

            s = self.processor.decode([token], skip_special_tokens=True)
            if not s:
                continue

            decoded_strings.append(s)
            decoded_scores.append(probas[token].item())

            for sp in range(string_position, string_position + len(s)):
                string_pos2kept_token_id[sp] = kept_token_id

            kept_token_id += 1
            string_position += len(s)

        decoded_string = "".join(decoded_strings)

        return decoded_string, decoded_scores, string_pos2kept_token_id

    @staticmethod
    def annotate_bbox(item: dict):
        item["annotator"].box_label(
            (item["x1"], item["y1"], item["x2"], item["y2"]),
            item["label"] + f"_{item['score']:.3f}",
            color=(255, 0, 0),
            txt_color=(0, 0, 0),
        )

    def _aggregate_scores(self, scores: np.array):
        match self.score_aggregation_mode:
            case "first":
                return scores[0]
            case "average":
                return np.mean(scores).item()
            case "product":
                return np.prod(scores).item()
            case "product_norm":
                return np.pow(np.prod(scores), 1.0 / len(scores)).item()
            case "max":
                return np.max(scores).item()
            case "min":
                return np.min(scores).item()
            case _:
                raise NotImplementedError(
                    f"Unknown aggregation mode: {self.score_aggregation_mode}"
                )

    def _parse_predictions(
        self,
        new_tokens_generated: np.ndarray,
        prob_distribution_generated: np.ndarray,
        original_image_shapes: tuple,
    ):
        """
        Parses bboxes, labels and scores from the model.generate() output.

        Args:
            new_tokens_generated (np.ndarray): Generated tokens excluding context.
            logits_generated (np.ndarray): Generated logits for each token.
            original_image_shapes (tuple): Original image shapes.

        Returns list of parsed bboxes, labels and scores.
        """

        assert len(new_tokens_generated) == len(
            prob_distribution_generated
        ), "Numbers of generated tokens and logit vectors must be the same."

        if not self.processor.decode(new_tokens_generated, skip_special_tokens=True):
            return []

        decoded_string, decoded_scores, string_pos2kept_token_id = (
            self._decode_string_extract_scores(
                new_tokens_generated, prob_distribution_generated
            )
        )
        pattern = re.compile(self.pattern)

        results = []
        for match in pattern.finditer(decoded_string):
            fr, to = match.span()

            token_from = string_pos2kept_token_id[fr]

            if self.verbose:
                tt = string_pos2kept_token_id[to] if to < len(decoded_string) else ""
                logger.debug(
                    f"fr: {fr}, to: {to}, len: {len(decoded_string)} {decoded_string} "
                    f"token_from: {token_from} "
                    f"token_to: {tt}; max_token: {max(string_pos2kept_token_id.values())}; "
                    f"{match.groups()}"
                )
            if to < len(decoded_string):
                label_probabilities = decoded_scores[
                    token_from
                    + self.loc_tokens_expected_number : string_pos2kept_token_id[to]
                ]
            else:
                label_probabilities = decoded_scores[
                    token_from + self.loc_tokens_expected_number :
                ]

            if not label_probabilities:
                continue

            y1, x1, y2, x2, label = match.groups()
            (y1, x1, y2, x2), is_valid_bbox = (
                PaliGemmaCocoPredictor._extract_validate_bbox(
                    y1, x1, y2, x2, original_image_shapes
                )
            )

            if not is_valid_bbox:
                continue

            label_clean = PaliGemmaCocoPredictor._parse_label(label)

            label_score = self._aggregate_scores(label_probabilities)

            results.append([y1, x1, y2, x2, label_score, label_clean])

        return results

    def _execute_batch(
        self,
        batch_images: list,
        batch_prompts: list,
        original_image_shapes: tuple,
    ):
        """
        Runs generation on batch, parses and returns bboxes, labels and scores.
        """

        model_inputs = (
            self.processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                truncation=False,
                padding="longest" if not self.max_input_length else "max_length",
                max_length=self.max_input_length,
            )
            .to(torch.bfloat16)
            .to(self.distributed_state.device)
        )

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                return_dict_in_generate=True,
                output_scores=True,
            )
        results = []
        for pos_id in range(len(batch_images)):
            new_tokens_generated = (
                generation.sequences[pos_id][input_len:].detach().cpu().numpy().tolist()
            )
            logits_generated = torch.stack([x[pos_id] for x in generation.scores])
            prob_distribution_generated = (
                F.softmax(logits_generated, dim=-1).detach().cpu().numpy()
            )

            r = self._parse_predictions(
                new_tokens_generated,
                prob_distribution_generated,
                original_image_shapes,
            )
            if r:
                results.extend(r)

        return results

    def detect(
        self,
        original_image: Image.Image,
        custom_classes: list[str] | None = None,
        filter_coco_classes=True,
    ):
        """
        original_image: RGB image

        Forms all prompts for a given image w.r.t. self.classes_per_call or
        custom_classes (if nonempty) and processes them in batches.
        Extracts predictions for all COCO or custom_classes for the image .
        """

        if custom_classes and filter_coco_classes:
            warnings.warn(
                "filter_coco_classes=True by default. "
                "Make sure it's expected behavior when using custom classes "
                "or set filter_coco_classes=False.",
                UserWarning,
            )
        classes_to_detect = custom_classes if custom_classes else self.coco_classes

        splits = [
            classes_to_detect[i : i + self.classes_per_call]
            for i in range(0, len(classes_to_detect), self.classes_per_call)
        ]

        results = []

        batch_images = []
        batch_prompts = []

        for classes in splits:
            batch_images.append(original_image)

            joined_classes = " ; ".join(classes)
            prompt = "<image>detect " + joined_classes + "\n"
            batch_prompts.append(prompt)

            if len(batch_images) == self.batch_size:
                r = self._execute_batch(
                    batch_images,
                    batch_prompts,
                    original_image.size,
                )
                results.extend(r)
                batch_images.clear()
                batch_prompts.clear()

        if batch_images:
            r = self._execute_batch(batch_images, batch_prompts, original_image.size)
            results.extend(r)

        if not filter_coco_classes:
            return results

        return [r for r in results if r[-1] in self.coco_class2id]

    def _extract_detection_results_in_benchmark_format(
        self, predictions: list, image_id: int
    ):
        """
        Prepares output to the benchmark format: xyxy instead of yxyx.
        Also maps classes to native coco class ids (0-90).
        """

        if not predictions:
            return []

        results = []
        cat_ids = [
            coco80_to_coco91_class()[self.coco_class2id[x[5]]] for x in predictions
        ]
        boxes = xyxy2ltwh(np.array([[x[1], x[0], x[3], x[2]] for x in predictions]))
        scores = [x[4] for x in predictions]

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

    @staticmethod
    def _parse_valid_image_ids(json_annotations: Path | str):
        """
        Not all coco val2017 are annotated but 4952.
        The rest are skipped internally by the benchmark.
        Extracts the annotated image_ids.
        """

        with open(json_annotations, "r") as f:
            data = json.load(f)

        i2gt: dict[int, list] = {}
        for item in data["annotations"]:
            if item["image_id"] not in i2gt:
                i2gt[item["image_id"]] = []

            item = {x: item[x] for x in ["image_id", "bbox", "category_id"]}
            i2gt[item["image_id"]].append(item)

        return sorted(i2gt.keys())

    def process_directory(
        self,
        path_to_images: Path | str,
        json_annotations: Path | str,
        predictions_folder=Path("predictions"),
        predictions_filename="all_predictions.json",
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
            for image_id in tqdm(
                images_ids_rank,
                "Processing images...",
                disable=not self.distributed_state.is_main_process,
            ):

                try:
                    path_to_image = path_to_images / f"{image_id:012d}.jpg"
                    original_image = Image.open(path_to_image)
                    results = self.detect(original_image)
                    my_predictions.extend(
                        self._extract_detection_results_in_benchmark_format(
                            results, image_id
                        )
                    )
                except Exception:
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"Error processing image id: {image_id}\n{tb_str}",
                        main_process_only=False,
                    )
                    failed_images.append(image_id)
                finally:
                    original_image.close()

        self._save_local_predictions_rank(my_predictions, predictions_folder)

        if failed_images:
            logger.warning(
                f"Rank {self.distributed_state.local_process_index} "
                f"failed to process {len(failed_images)} image(s): {failed_images}",
                main_process_only=False,
            )

        self.distributed_state.wait_for_everyone()

        if self.distributed_state.is_main_process:
            self._collect_all_json_and_save(predictions_folder, predictions_filename)

            PaliGemmaCocoPredictor.calculate_coco_metrics(
                json_annotations, predictions_folder / predictions_filename
            )

        self.distributed_state.wait_for_everyone()

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

    @staticmethod
    def _appply_nms_benchmark_format(
        json_predictions: Path | str, nms_callable: Callable
    ):
        with open(json_predictions, "r") as f:
            data = json.load(f)

        i2gt: dict[int, list] = {}
        for item in data:
            if item["image_id"] not in i2gt:
                i2gt[item["image_id"]] = []
            i2gt[item["image_id"]].append(item)

        results = []
        for image_id, group in i2gt.items():
            bboxes_scores_labels = []
            for item in group:
                box_array_ltwh = np.array([*item["bbox"]])
                box_list_xyxy = ltwh2xyxy(box_array_ltwh).tolist()
                box_list = [*box_list_xyxy, item["score"], item["category_id"]]
                bboxes_scores_labels.append(box_list)

            nms_result = nms_callable(bboxes_scores_labels)

            for x1, y1, x2, y2, score, category_id in nms_result:

                results.append(
                    {
                        "image_id": image_id,
                        "file_name": f"{image_id:012d}.jpg",
                        "category_id": category_id,
                        "bbox": xyxy2ltwh(np.array([x1, y1, x2, y2])).tolist(),
                        "score": score,
                    }
                )

        return results

    @staticmethod
    def apply_nms_to_predictions(
        detections, iou_threshold=0.7, conf_threshold=0.001, class_agnostic=False
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
    def calculate_coco_metrics(
        json_annotations: Path | str,
        json_predictions: Path | str,
        nms_callable: Callable | None = partial(
            apply_nms_to_predictions,
            conf_threshold=0.001,
            iou_threshold=0.7,
            class_agnostic=False,
        ),
    ):
        logger.info(
            f"Calculating COCO metrics. NMS: {nms_callable.keywords if nms_callable else ''}"
        )

        coco_gt = COCO(str(json_annotations))

        if nms_callable is not None:
            with tempfile.NamedTemporaryFile(mode="w+") as temp:

                filtered_data = PaliGemmaCocoPredictor._appply_nms_benchmark_format(
                    json_predictions, nms_callable
                )

                with open(temp.name, "w") as f:
                    json.dump(filtered_data, f)

                coco_pred = coco_gt.loadRes(temp.name)
        else:
            coco_pred = coco_gt.loadRes(str(json_predictions))

        coco_eval = COCOeval(coco_gt, coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats

        # fmt: off
        metrics = [
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[0]),
            ('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]', stats[1]),
            ('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]', stats[2]),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[3]),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[4]),
            ('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[5]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]', stats[6]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]', stats[7]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]', stats[8]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]', stats[9]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]', stats[10]),
            ('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]', stats[11]),
        ]
        # fmt: on

        result = []
        for metric_name, value in metrics:
            result.append(f" {metric_name} = {value:.3f}")

        logger.info("\n".join(result))


def run_detection(
    paligemma_predictor: PaliGemmaCocoPredictor,
    path_to_image: Path,
    classes: list[str] | None = None,
    save_path: Path | str | None = None,
    line_width: int = 1,
    font_size: int = 11,
    nms_callable: Callable | None = partial(
        PaliGemmaCocoPredictor.apply_nms_to_predictions,
        conf_threshold=0.25,
        iou_threshold=0.7,
        class_agnostic=False,
    ),
):
    """
    Runs detection with paligemma_predictor on image at path_to_image.
    Detects classes from classes list if nonempty, otherwise detects coco classes (80).
    Optionally saves the image to save_path.
    Returns detections in xyxy format.
    """
    original_image = Image.open(path_to_image)
    if original_image.mode != "RGB":
        tmp_image = original_image.convert("RGB")
        original_image.close()
        original_image = tmp_image

    annotator = Annotator(
        original_image, line_width=line_width, font_size=font_size, pil=True
    )
    results = paligemma_predictor.detect(
        original_image,
        custom_classes=classes,
        filter_coco_classes=not bool(classes),
    )

    if nms_callable is not None:
        results = nms_callable(results)

    for x1, y1, x2, y2, score, label in results:
        PaliGemmaCocoPredictor.annotate_bbox(
            {
                "annotator": annotator,
                "x1": y1,
                "y1": x1,
                "x2": y2,
                "y2": x2,
                "label": label,
                "score": score,
            }
        )

    annotated_img = annotator.result()

    if save_path:
        plt.imsave(save_path, annotated_img)

    plt.close()
    original_image.close()

    return results, annotated_img


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="paligemma_coco_benchmark",
)
def main(cfg: DictConfig):
    distributed_state = PartialState()

    logger.info(OmegaConf.to_yaml(cfg))

    try:
        pg_predictor = PaliGemmaCocoPredictor(
            Path(cfg.model.path),
            distributed_state,
            use_flash_attention=cfg.model.use_flash_attention,
            max_new_tokens=cfg.generation.max_new_tokens,
            do_sample=cfg.generation.do_sample,
            classes_per_call=cfg.coco.classes_per_call,
            batch_size=cfg.generation.batch_size,
            score_aggregation_mode=cfg.coco.score_aggregation_mode,
            max_input_length=cfg.generation.max_input_length,
        )

        path_to_images = Path(cfg.coco.path_to_images)
        json_annotations = Path(cfg.coco.json_annotations)
        pg_predictor.process_directory(path_to_images, json_annotations)

        # coco_img_example = "000000398742.jpg"
        # results, _ = run_detection(
        #     pg_predictor,
        #     Path(cfg.coco.path_to_images) / coco_img_example,
        #     save_path="test_predictions.jpg"
        # )
        # print(results)
    finally:
        distributed_state.wait_for_everyone()
        distributed_state.destroy_process_group()


if __name__ == "__main__":
    main()
