# Paligemma-2 Object Detection

This repository contains implementations for research work into PaliGemma-2 object detection capabilities.

## Main results

- Deployed a Triton Inference Server with TensorRT backend for YOLOv11x model and reproduced the official COCO2017 object detection benchmark results
- Implemented a framework for distributed evaluation of Paligemma-2 models on COCO2017 object detection validation dataset in both "open class" and "closed set" modes
- Implemented a pipeline for distributed fine-tuning of Paligemma-2 models using Ray Data and Ray Train
- Implemented Gradio demos for YOLOv11x and Paligemma-2 models

| Model                                                                                                               | COCO2017_val mAP@[IoU=0.50:0.95] |
|---------------------------------------------------------------------------------------------------------------------|----------------------------------|
| YOLOv11x                                                                                                            | 0.546                            |
| Paligemma-2, open class mode (best)                                                                                 | 0.239                            |
| Paligemma-2, closed set ([fine-tuned](https://huggingface.co/segis95/paligemma2-10b-pt-448-adapter_detection_coco)) | 0.296                            |

## Install Dependencies

- ``uv sync``
- ``source .venv/bin/activate``

## Model Evaluation

Configure Accelerate (GPUs, bf16): ``accelerate config``

### Open class mode

1. Update paths in ``configs/paligemma_coco_benchmark.yaml``
2. ``uv run --module accelerate.commands.launch inference/paligemma_coco_benchmark.py``

### Closed set mode

1. Update paths in ``configs/paligemma_structured_coco_benchmark.yaml``
2. ``uv run --module accelerate.commands.launch inference/paligemma_structured_coco_benchmark.py``

## Training

1. Set up Ray: ``ray start --head --dashboard-host=0.0.0.0 --disable-usage-stats [--object-store-memory=20GiB --temp-dir /path/to/ray_tmp --system-config='{"object_spilling_config":"{"type":"filesystem","params":{"directory_path":"/path/to/spill"}}"}']``
2. Configure ``configs/train/default.yaml``
3. ``uv run training/train.py``
4. (Optional) ``tensorboard --logdir tb_dir_from_train_config``


## Demos

### YOLOv11x demo

1. Follow [yolo deployment instructions](docs/yolo_deployment.md)
2. ``uv run deploy/demo_yolo.py [--url yolo_endpoint_url]``

### Paligemma-2 demo
1. Download model from Huggingface (e.g. ``google/paligemma2-10b-pt-448`` or ``google/paligemma2-3b-pt-448``) and adapter (``https://huggingface.co/segis95/paligemma2-10b-pt-448-adapter_detection_coco``)
2. ``uv run deploy/demo_paligemma.py --model_path /path/to/paligemma2-10b-pt-448 [--adapter_path /path/to/adapter]``

## Predictions and results

Prediction files for some models (COCO2017_val benchmark) are located at `assets/predictions/`. 
To get these files run `git lfs pull`.
The corresponding benchmark results are showcased in `assets/benchmark_results.ipynb`.