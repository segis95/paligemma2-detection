import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Callable, Optional
from PIL import Image
import tritonclient.http as httpclient
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression


class TritonYOLOInference:
    """
    Unified class for YOLO inference via Triton Inference Server.
    Supports image paths and PIL Image objects with preprocessing.
    """

    def __init__(
            self,
            url: str = "localhost:8000",
            model_name: str = "yolo",
            preprocess_fn: Optional[Callable] = None,
            conf_threshold: float = 0.001,
            iou_threshold: float = 0.7,
            max_det: int = 300,
    ):
        """
        Initialize Triton inference client.

        Args:
            url: Triton server URL
            model_name: Name of the model in Triton
            preprocess_fn: Callable for preprocessing images (replaces model.predictor.preprocess)
            conf_threshold: Confidence threshold for NMS
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections
        """
        self.url = url
        self.model_name = model_name
        self.preprocess_fn = preprocess_fn
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # Create Triton client once during initialization
        self.triton_client = httpclient.InferenceServerClient(url=self.url)

    def _load_image(self, image_source: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Load and convert image to RGB format.

        Args:
            image_source: Path to image, PIL Image, or numpy array

        Returns:
            Image as numpy array in RGB format
        """
        if isinstance(image_source, (str, Path)):
            # Load from file path
            img = cv2.imread(str(image_source))
            if img is None:
                raise ValueError(f"Failed to load image from {image_source}")

        elif isinstance(image_source, Image.Image):
            # Convert PIL Image to RGB if not already
            if image_source.mode != 'RGB':
                image_source = image_source.convert('RGB')
            img = np.array(image_source)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            # Assume already in BGR format from OpenCV
            img = image_source
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")

        return img

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image using provided callable or default method.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image ready for inference
        """
        if self.preprocess_fn is not None:
            return self.preprocess_fn([image])
        else:
            raise ValueError("Preprocessing function must be provided in constructor")

    def _postprocess(
            self,
            raw_output: np.ndarray,
            input_shape: tuple,
            original_shape: tuple
    ) -> torch.Tensor:
        """
        Apply NMS and scale boxes to original image size.

        Args:
            raw_output: Raw model output from Triton
            input_shape: Shape of preprocessed input
            original_shape: Shape of original image

        Returns:
            Filtered predictions with scaled boxes
        """
        # Apply Non-Maximum Suppression
        pred = non_max_suppression(
            torch.from_numpy(raw_output),
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            max_det=self.max_det,
        )[0]

        # Scale boxes to original image size
        pred[:, :4] = scale_boxes(input_shape[2:], pred[:, :4], original_shape)

        return pred

    def predict(
            self,
            image_source: Union[str, Path, Image.Image, np.ndarray],
            conf_threshold: Optional[float] = None,
            iou_threshold: Optional[float] = None,
            max_det: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run inference on image via Triton server.

        Args:
            image_source: Image path, PIL Image, or numpy array
            conf_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            max_det: Override default max detections

        Returns:
            Tensor with predictions [x1, y1, x2, y2, conf, class]
        """
        # Override thresholds if provided
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if max_det is not None:
            self.max_det = max_det

        # Load and convert image to RGB
        original_img = self._load_image(image_source)
        original_shape = original_img.shape

        # Preprocess image
        input_data = self._preprocess(original_img)

        # Convert to numpy if tensor
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()

        # Prepare Triton inputs
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # Define outputs
        outputs = [httpclient.InferRequestedOutput("output0")]

        # Run inference via Triton
        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        raw_output = response.as_numpy("output0")

        # Postprocess and return predictions
        predictions = self._postprocess(raw_output, input_data.shape, original_shape)

        return predictions

    def close(self):
        """Close Triton client connection."""
        self.triton_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Usage example
if __name__ == "__main__":
    from ultralytics.data.converter import coco80_to_coco91_class
    from ultralytics.utils.ops import xyxy2ltwh
    from tqdm import tqdm
    from ultralytics import YOLO

    with open('../../datasets/coco/annotations/instances_val2017.json', 'r') as f:
        data = json.load(f)

    i2gt = {}
    for item in data['annotations']:
        if item['image_id'] not in i2gt:
            i2gt[item['image_id']] = []
        item = {x: item[x] for x in {'image_id', 'bbox', 'category_id'}}
        i2gt[item['image_id']].append(item)

    model_ = YOLO("yolo11x.pt")
    model_.predict(Image.fromarray(np.zeros((224, 224))))

    # Initialize inference client once
    with TritonYOLOInference(
            url="localhost:8000",
            model_name="yolo",
            preprocess_fn=model_.predictor.preprocess,
            conf_threshold=0.001,
            iou_threshold=0.6,
            max_det=300
    ) as inference_client:

        my_predictions = []

        for image_id in tqdm(i2gt):
            # Inference from image path
            predictions = inference_client.predict(
                f'../../datasets/coco/images/val2017/{image_id:012d}.jpg'
            )

            # Alternative: Inference from PIL Image
            # pil_img = Image.open(f'../datasets/coco/images/val2017/{image_id:012d}.jpg')
            # predictions = inference_client.predict(pil_img)

            predictions = predictions.cpu().numpy()
            boxes = xyxy2ltwh(predictions[:, :4])
            cat_ids = [coco80_to_coco91_class()[int(x)] for x in predictions[:, -1]]
            scores = predictions[:, -2]

            for box, cat, score in zip(boxes, cat_ids, scores):
                my_predictions.append({
                    'image_id': image_id,
                    'file_name': f'{image_id:012d}.jpg',
                    'category_id': cat,
                    'bbox': box.tolist(),
                    'score': score.item()
                })
