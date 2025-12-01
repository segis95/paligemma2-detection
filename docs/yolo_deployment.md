# Building Triton Inference Server for YOLOv11x and running COCO2017 benchmark

## Export YOLOv11x to ONNX

```python
from ultralytics import YOLO
model = YOLO("yolo11x.pt")
onnx_file = model.export(format="onnx", dynamic=True)
```

Also see ``deploy/convert_yolo_to_onnx.ipynb``

## Convert ONNX to TensorRT

### Download official NVIDIA tritonserver image

It's important that the TensorRT plan is obtained and used within the same system and TensorRT version.

```shell
mv yolo11x.onnx /path/to/paligemma2-detection/deploy/models
docker pull nvcr.io/nvidia/tritonserver:25.10-py3
docker run --gpus=1 -it -v /path/to/paligemma2-detection/deploy/models:/models nvcr.io/nvidia/tritonserver:25.10-py3 /bin/bash
```

### Launch the conversion in the downloaded container

This may take up to half an hour with an NVIDIA RTX3090.

```shell
export PATH=/usr/src/tensorrt/bin:$PATH
cd /models
trtexec \
  --onnx=yolo11x.onnx \
  --saveEngine=model.plan \
  --minShapes=images:1x3x128x128 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:16x3x1280x1280 \
  --memPoolSize=workspace:8192 \
  --fp16
exit
```

## Launch Triton Server

```shell
cp deploy/models/model.plan deploy/triton/yolo/1
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/paligemma2-detection/deploy/triton:/models \
   nvcr.io/nvidia/tritonserver:25.10-py3 tritonserver --model-repository=/models
```

## Running benchmarks

Primary goals:
- Confirm the YOLOv11x value from the official [page](https://docs.ultralytics.com/fr/models/yolo11/#performance-metrics) and [paper](https://arxiv.org/abs/2410.17725)
- Learn how to create a prediction manually to be able to benchmark other detection models
- Compare quality of a TensorRT+Triton deploy versus original model

COCO2017 data (train and val) is loaded automatically when calling ``model.val(data="coco.yaml", augment=False)``.
See the notebook at ``deploy/coco_benchmark_yolo.ipynb`` for details.

### Results

mAP@[IoU=0.50:0.95] is **0.546** (original weights) vs **0.536** (TensorRT+Triton).

The official result for YOLOv11x is confirmed. The quality loss is 1 percentage point. 
It is now possible to benchmark Paligemma-2 models in the same way.


