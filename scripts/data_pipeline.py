import ray
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import yaml
from functools import partial
import albumentations as A
import torch
import traceback

from typing import Optional
from ultralytics.data.converter import coco91_to_coco80_class
from ultralytics.utils.ops import ltwh2xyxy
from ultralytics.utils.plotting import Annotator


def select_by_rank(batch, world_size, rank):
    mask = (batch["image_id"] % world_size) == rank
    batch["bbox"] = [bbox.tolist() for bbox in batch["bbox"]]
    return {key: batch[key][mask] for key in batch}


def process_group(batch):
    image_id = batch["image_id"][0]
    cat_ids = batch["category_id"].tolist()
    boxes = batch["bbox"].tolist()
    return {"image_id": [image_id], "category_id": [cat_ids], "boxes": [boxes]}


def load_images_safe(batch, images_dir, is_train_mode):
    images = []
    valid_indices = []

    for idx, img_id in enumerate(batch["image_id"]):
        filename = f"{int(img_id):012d}.jpg"

        if is_train_mode:
            filepath = images_dir / f"{img_id % 1000}" / filename
        else:
            filepath = images_dir / filename

        try:
            with Image.open(filepath) as img:
                images.append(np.array(img.convert("RGB")))

            valid_indices.append(idx)
        except FileNotFoundError:
            print(f"Warning: Image {filepath} not found, skipping")
            continue

    if len(valid_indices) < len(batch["image_id"]):
        valid_indices = np.array(valid_indices)
        batch = {key: [batch[key][i] for i in valid_indices] for key in batch}

    batch["image"] = images
    return batch


class ImageAugmentationActor:
    def __init__(self, is_train_mode):

        self.transform_train = A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.7,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0, p=0.5
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.2,
                ),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_id"],
                min_area=1.0,
                min_visibility=0.3,
            ),
        )

        self.transform_val = A.Compose(
            [A.NoOp(p=1.0)],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_id"],
            ),
        )

        self.is_train_mode = is_train_mode

    @staticmethod
    def calculate_iou_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Vectorized IoU calculation between two sets of bounding boxes.

        Args:
            boxes1: Array of shape (N, 4) with boxes in xyxy format
            boxes2: Array of shape (M, 4) with boxes in xyxy format

        Returns:
            Array of shape (N, M) with pairwise IoU values
        """
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, None, :]  # (N, 1, 4)
        boxes2 = boxes2[None, :, :]  # (1, M, 4)

        # Calculate intersection coordinates
        x1_inter = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1_inter = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2_inter = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2_inter = np.minimum(boxes1[..., 3], boxes2[..., 3])

        # Calculate intersection area
        inter_width = np.maximum(0, x2_inter - x1_inter)
        inter_height = np.maximum(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        # Calculate areas of boxes
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1]
        )
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (
            boxes2[..., 3] - boxes2[..., 1]
        )

        # Calculate union area
        union_area = boxes1_area + boxes2_area - inter_area

        # Calculate IoU (avoid division by zero)
        iou = np.where(union_area > 0, inter_area / union_area, 0)

        return iou.squeeze()

    @staticmethod
    def generate_boxes_with_iou_constraint(
        existing_boxes: np.ndarray,
        image_width: int,
        image_height: int,
        num_boxes: int,
        iou_threshold: float = 0.15,
        min_box_size: int = 16,
        max_box_size: Optional[int] = None,
        max_attempts: int = 2000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate bounding boxes with at most IoU threshold with any existing box.

        Args:
            existing_boxes: Array of shape (N, 4) with existing boxes in xyxy format
            num_boxes: Number of new boxes to generate
            iou_threshold: Maximum allowed IoU with any existing box (0 to 1)
            image_width: Width of the image/canvas
            image_height: Height of the image/canvas
            min_box_size: Minimum width/height of generated boxes
            max_box_size: Maximum width/height (None = half canvas size)
            max_attempts: Maximum attempts to generate a valid box
            seed: Random seed for reproducibility

        Returns:
            Array of shape (M, 4) with generated boxes (M ≤ num_boxes)
        """

        if seed is not None:
            np.random.seed(seed)

        if max_box_size is None:
            max_box_size = min(image_width, image_height) // 2

        generated_boxes = []
        all_boxes = (
            existing_boxes.copy() if len(existing_boxes) > 0 else np.empty((0, 4))
        )

        for _ in range(num_boxes):
            attempts = 0
            valid_box_found = False

            while attempts < max_attempts and not valid_box_found:
                # Generate random box size
                width = np.random.randint(min_box_size, min(max_box_size, image_width))
                height = np.random.randint(
                    min_box_size, min(max_box_size, image_height)
                )

                # Generate random position
                x1 = np.random.randint(0, image_width - width)
                y1 = np.random.randint(0, image_height - height)
                x2 = x1 + width
                y2 = y1 + height

                new_box = np.array([[x1, y1, x2, y2]])

                # Check IoU with all existing boxes using vectorized calculation
                if len(all_boxes) > 0:
                    iou_values = ImageAugmentationActor.calculate_iou_vectorized(
                        new_box, all_boxes
                    )
                    if np.all(iou_values <= iou_threshold):
                        valid_box_found = True
                else:
                    valid_box_found = True

                if valid_box_found:
                    generated_boxes.append(new_box[0])
                    all_boxes = np.vstack([all_boxes, new_box])

                attempts += 1
                if attempts >= max_attempts // 2:
                    iou_threshold += 0.2
                    attempts = 0

            # if not valid_box_found:
            #     # print(f"Warning: Could not generate box {len(generated_boxes) + 1} after {max_attempts} attempts")
            #     break

        return np.array(generated_boxes) if generated_boxes else np.empty((0, 4))

    def _build_sequence(self, boxes, classes, width, height, gen_boxes=None):
        tokens = []
        mask = []
        if self.is_train_mode:
            assert gen_boxes is not None, "in train mode gen_boxes must be passed"

        classes_ext = classes + [None] * (len(gen_boxes) if gen_boxes else 0)
        boxes_ext = boxes + (gen_boxes if gen_boxes else [])
        for (x1, y1, x2, y2), cls in zip(boxes_ext, classes_ext):
            x1_loc = min(round(x1 * 1024.0 / width), 1023)
            x2_loc = min(round(x2 * 1024.0 / width), 1023)
            y1_loc = min(round(y1 * 1024.0 / height), 1023)
            y2_loc = min(round(y2 * 1024.0 / height), 1023)

            tokens.append(f"<loc{y1_loc:04d}>")
            tokens.append(f"<loc{x1_loc:04d}>")
            tokens.append(f"<loc{y2_loc:04d}>")
            tokens.append(f"<loc{x2_loc:04d}>")

            if cls is None:
                tokens.append("<unused80>")
                mask.extend([0, 0, 0, 0, 1])
            else:
                tokens.append(f"<unused{cls}>")
                mask.extend([1, 1, 1, 1, 1])

        return tokens, mask

    def __call__(self, batch, total_bboxes_aug=51):

        results = []

        for row in batch.itertuples(index=False):

            try:
                transform = (
                    self.transform_train if self.is_train_mode else self.transform_val
                )

                transformed = transform(
                    image=row.image,
                    bboxes=row.boxes.tolist(),
                    category_id=row.category_id.tolist(),
                )

                boxes = []
                for box in transformed["bboxes"]:
                    boxes.append(ltwh2xyxy(np.array(box)).tolist())

                # filtering out items with no bboxes after augmentations
                if not boxes:
                    continue

                categories = list(map(int, transformed["category_id"]))
                categories = [
                    coco91_to_coco80_class()[int(cat) - 1] for cat in categories
                ]

                if self.is_train_mode:
                    boxes_order = list(range(len(boxes)))
                    np.random.shuffle(boxes_order)
                    boxes = [boxes[i] for i in boxes_order]
                    categories = [categories[i] for i in boxes_order]

                boxes = boxes[:total_bboxes_aug]
                categories = categories[:total_bboxes_aug]

                r = {
                    "image_id": [row.image_id],
                    "category_id": categories,
                    "boxes": boxes,
                    "image": transformed["image"],
                }

                height = len(row.image)
                width = len(row.image[0])
                if self.is_train_mode:
                    new_boxes = (
                        ImageAugmentationActor.generate_boxes_with_iou_constraint(
                            existing_boxes=np.stack(boxes),
                            image_width=width,
                            image_height=height,
                            num_boxes=max(0, total_bboxes_aug - len(boxes)),
                        ).tolist()
                    )

                    r["gen_boxes"] = new_boxes

                tokens, mask = self._build_sequence(
                    boxes=boxes,
                    classes=categories,
                    width=width,
                    height=height,
                    gen_boxes=r["gen_boxes"] if "gen_boxes" in r else None,
                )

                if self.is_train_mode:
                    assert len(tokens) == 255, f"{len(tokens)}"
                    assert len(mask) == 255, f"{len(mask)}"

                r["tokens"] = tokens
                r["mask"] = mask

                results.append(r)
            except Exception:
                print(f"A problem occurred while augmenting image_id {row.image_id}.")
                traceback.print_exc()

            if not results:
                return {}

        return {key: [x[key] for x in results] for key in results[0]}

        # return self._preprocess_batch(results)


class PreprocessActor:
    def __init__(self, preprocessor, is_train_mode):
        self.preprocessor = preprocessor
        self.is_train_mode = is_train_mode

    def __call__(self, batch):
        prompt_instruct = "<image>detect all classes\n"

        images = []
        suffixes = []
        masks = []

        for row in batch.itertuples(index=False):
            images.append(row.image)
            suffixes.append("".join(row.tokens))
            masks.append(row.mask)

        prompts = [prompt_instruct for _ in range(len(images))]

        result_batch = self.preprocessor(
            text=prompts,
            images=images,
            suffix=suffixes,
            padding="max_length",
            max_length=1285,
            return_tensors="pt",
        )

        if self.is_train_mode:
            for idx in range(len(result_batch["labels"])):
                result_batch["labels"][idx][-256:-1][
                    ~torch.tensor(masks[idx], dtype=torch.bool)
                ] = -100

        return result_batch


def get_train_ds(json_annotations_path, images_dir, preprocessor, limit=None):

    with open(json_annotations_path, "r") as f:
        coco2017 = json.load(f)

    annotations = ray.data.from_items(
        [
            {
                "category_id": x["category_id"],
                "bbox": x["bbox"],
                "image_id": x["image_id"],
            }
            for x in coco2017["annotations"]
        ]
    )

    annotations = annotations.groupby("image_id").map_groups(
        process_group, batch_format="pandas", concurrency=1, num_cpus=1
    )

    if limit is not None:
        annotations = annotations.limit(limit)

    annotations = annotations.materialize()

    annotations = annotations.map_batches(
        load_images_safe,
        fn_kwargs={"images_dir": images_dir, "is_train_mode": True},
        batch_format="pandas",
        batch_size=64,
        concurrency=1,
        num_cpus=1,
    )

    annotations = annotations.map_batches(
        ImageAugmentationActor,
        fn_constructor_kwargs={"is_train_mode": True},
        batch_format="pandas",
        batch_size=32,
        concurrency=1,
        num_cpus=1,
    )

    annotations = annotations.map_batches(
        PreprocessActor,
        fn_constructor_kwargs={"is_train_mode": True, "preprocessor": preprocessor},
        batch_format="pandas",
        batch_size=8,
        concurrency=1,
        num_cpus=1,
    )

    return annotations


def get_val_ds(json_annotations_path, images_dir, preprocessor, limit=1000):

    with open(json_annotations_path, "r") as f:
        coco2017 = json.load(f)

    annotations = ray.data.from_items(
        [
            {
                "category_id": x["category_id"],
                "bbox": x["bbox"],
                "image_id": x["image_id"],
            }
            for x in coco2017["annotations"]
        ]
    )

    annotations = annotations.groupby("image_id").map_groups(
        process_group, batch_format="pandas", concurrency=1, num_cpus=1
    )

    if limit is not None:
        annotations = annotations.limit(limit)

    annotations = annotations.map_batches(
        load_images_safe,
        fn_kwargs={"images_dir": images_dir, "is_train_mode": False},
        batch_size=32,
        batch_format="pandas",
        concurrency=1,
        num_cpus=1,
    )

    annotations = annotations.map_batches(
        ImageAugmentationActor,
        fn_constructor_kwargs={"is_train_mode": False},
        batch_format="pandas",
        batch_size=32,
        concurrency=1,
        num_cpus=1,
    )

    annotations = annotations.map_batches(
        PreprocessActor,
        fn_constructor_kwargs={"is_train_mode": False, "preprocessor": preprocessor},
        batch_format="pandas",
        batch_size=32,
        concurrency=1,
        num_cpus=1,
    )

    return annotations


def draw_gt(item, cocoid2class=None, show_generated=False):
    annotator = Annotator(
        Image.fromarray(item["image"]), line_width=2, font_size=11, pil=True
    )

    bboxes = item["boxes"] if "boxes" in item else item["bboxes"]
    for box, category in zip(bboxes, item["category_id"]):
        x1, y1, x2, y2 = np.array(box)
        annotator.box_label(
            (x1, y1, x2, y2),
            f"{cocoid2class[category] if cocoid2class else category}",
            color=(255, 0, 0),
            txt_color=(0, 0, 0),
        )

    if show_generated and "gen_boxes" in item:
        for box in item["gen_boxes"]:
            x1, y1, x2, y2 = box
            annotator.box_label(
                (x1, y1, x2, y2),
                "",
                color=(0, 0, 255),
                txt_color=(0, 0, 0),
            )

    return annotator.result()
