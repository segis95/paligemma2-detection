# Key Findings: Object Detection Performance Analysis on COCO val2017

## <a href="coco_metrics.pdf" target="_blank">COCO Benchmark Results</a>

## Overall Performance Comparison
YOLO v11x significantly outperforms all PaliGemma-2 variants on COCO val2017 (36,781 objects, 80 classes), achieving `mAP@0.5:0.95` of 0.536 compared to the best PaliGemma variant (10B-finetuned) at 0.296.
This represents an 81% performance gap, demonstrating that specialized object detectors remain substantially superior to vision-language models (VLMs) for this task.

## Critical Weaknesses of PaliGemma-2

### Small Object Detection

PaliGemma-2 exhibits severe limitations on small objects:
- YOLO `AP_small`: 0.370 
- Best PaliGemma `AP_small`: 0.097 (10B-finetuned)

### Recall Limitations

PaliGemma-2 demonstrates architectural constraints in object discovery:

- YOLO `mAR@100`: 0.685 (finds 68.5% of all objects)
- Best PaliGemma `mAR@100`: 0.399 (finds 39.9% of all objects)

Notably, PaliGemma `mAR@10` (0.396) and `mAR@100` (0.399) show minimal improvement (+0.3pp), indicating the model cannot generate more valid detections even when allowed higher detection limits.
This contrasts sharply with YOLO's +4.4pp improvement, demonstrating fundamental architectural limitations rather than calibration issues.

## Evaluation Methodology

All models were evaluated using identical low-threshold settings (`conf_threshold=0.001`, `iou_threshold=0.7`) to measure theoretical capability rather than production performance. 

This approach reveals:
- Maximum recall potential: how many objects each model can theoretically detect 
- Precision at minimal filtering: whether models spam false positives or maintain reasonable precision 
- Architectural vs. calibration issues: low recall at minimal thresholds indicates fundamental limitations

### AP vs. Precision Interpretation
The apparent paradox - YOLO having lower per-class precision (0.03-0.65) but higher AP (0.536) - is explained by metric definitions:

- Precision (in table): Single-point metric at `conf>=0.001`, measuring `TP/(TP+FP)` at one threshold
- AP: Area under precision-recall curve across all confidence thresholds

YOLO's aggressive detection strategy (`recall=0.93`) generates many predictions, lowering precision at minimal thresholds but providing excellent precision-recall trade-off across the full confidence range.
PaliGemma's conservative strategy (`recall=0.54`) shows higher precision at low thresholds but cannot find enough objects to compete in AP.

## Hyperparameter Sensitivity

### Resolution Impact

Higher resolution does not improve performance beyond 448px, suggesting computational or attention mechanism constraints.

### Classes-per-Call Batching

Increasing `classes_per_call` beyond 10 typically degrades performance, indicating the model struggles with multi-class simultaneous detection.

### Catastrophic 28B Model Failure

PaliGemma-2 28B with 4 classes-per-call achieved only 0.009 mAP, the worst result despite being the largest model tested.
This catastrophic underperformance may be related to an observation from the [original paper](https://arxiv.org/pdf/2412.03555),
which notes that "a possible factor related to the relatively worse transferability of PaliGemma 2 28B is that the underlying Gemma 2 27B model is trained from scratch, as opposed to the 2B and 9B models, which are distilled."

This training methodology difference could explain why the 28B variant fails to leverage its larger capacity effectively for object detection fine-tuning,
demonstrating that distilled models may exhibit superior transfer learning properties compared to models trained from scratch.

## Per-Class Performance Patterns

### PaliGemma-2 Strengths (higher precision)
Models perform well on:

- Large, distinctive objects: fire hydrant, clock, scissors
- Animals with clear silhouettes: cat, dog, giraffe, elephant
- High-contrast objects: stop sign, airplane

### PaliGemma-2 Weaknesses (lower recall)
Critical failures on:

- Small objects: apple, knife, remote
- Multiple similar objects: book, chair
- Thin/occluded objects: skis, tie

These patterns indicate PaliGemma-2 relies on distinctive visual features and struggles with scale, multiplicity, and occlusion.

## Conclusions

VLMs are not production-ready for object detection: The 2-3x performance gap between YOLO and PaliGemma-2 on COCO demonstrates specialized detectors remain essential for serious detection tasks.

Architectural constraints dominate: Low recall even at minimal confidence thresholds (`conf=0.001`) indicates fundamental model limitations, not calibration issues. Small object detection remains a critical weakness.

Optimal configuration for PaliGemma-2 inference: 10B parameter model, 448px resolution, 1 class per call, fine-tuned for closed-set setup.

Research directions: small object detection, multiple object handling, and scaling efficiency (28B underperforms 10B) require architectural innovations rather than parameter scaling.

This analysis demonstrates that while vision-language models show promise for multimodal tasks, they currently lack the architectural specialization required for competitive object detection performance on standard benchmarks like COCO.