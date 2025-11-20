import gradio as gr
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image
from yolo_triton_client import TritonYOLOInference


class YOLOGradioDemo:
    """Gradio demo for object detection with Triton Inference Server"""

    def __init__(self, triton_client, class_names=None):
        """
        Demo initialization

        Args:
            triton_client: instance of TritonYOLOInference
            class_names: Dict or list of class names (optional)
        """
        self.triton_client = triton_client

        if class_names is None:
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        else:
            self.class_names = class_names

    def detect_and_draw(self, image: np.ndarray) -> np.ndarray:
        """
        Performing detection and drawing on an image

        Args:
            image: Input Image (numpy array RGB from Gradio)

        Returns:
            Image with drawn detections (numpy array RGB)
        """
        if image is None:
            return None

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # predictions: tensor [x1, y1, x2, y2, conf, class]
        predictions = self.triton_client.predict(image_bgr)

        predictions = predictions.cpu().numpy()

        annotator = Annotator(
            image_bgr,
            line_width=2,
            font_size=12,
            pil=False
        )

        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred

            class_id = int(cls)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            label = f"{class_name} {conf:.2f}"

            annotator.box_label(
                [x1, y1, x2, y2],
                label,
                color=colors(class_id, bgr=True)
            )

        annotated_image = annotator.result()

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image_rgb


def create_gradio_app(triton_client):
    """
    Creates Gradio interface

    Args:
        triton_client: Instance TritonYOLOInference

    Returns:
        Gradio Blocks app
    """
    demo = YOLOGradioDemo(triton_client)

    with gr.Blocks(title="YOLO Object Detection") as app:
        gr.HTML(
            """
            <h1 style='text-align: center'>
            🚀 YOLO Object Detection with Triton Inference Server
            </h1>
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload image",
                    type="numpy"
                )
                detect_btn = gr.Button("🔍 Run detection", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(
                    label="Detection result",
                    type="numpy"
                )

        detect_btn.click(
            fn=demo.detect_and_draw,
            inputs=input_image,
            outputs=output_image
        )

    return app


if __name__ == "__main__":
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image

    model = YOLO("yolo11x.pt")
    model.predict(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))

    triton_client = TritonYOLOInference(
        url="localhost:8000",
        model_name="yolo",
        preprocess_fn=model.predictor.preprocess,
        conf_threshold=0.25,
        iou_threshold=0.6,
        max_det=300
    )

    app = create_gradio_app(triton_client)
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )