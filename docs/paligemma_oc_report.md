# Paligemma-2 Open Class Object Detection Evaluation

As a naturally open class detection models, Paligemma-2 models need a special evaluation protocol for a closed-set detection benchmark like COCO2017.


## Prediction Building Protocol

- classes_per_call is set to an integer between 1 and 80
- 80 COCO classes are divided (evenly when possible) groups, each containing classes_per_call classes
- For each group with the set of class names c_1, c_2, ... , c_{classes_per_call}, a prompt is constructed as follows:
  - detect c1 ; c2 ; … ; c_{classes_per_call}\n
- The resulting prompt is sent to the model along with the input image. The model outputs a set of bounding boxes in the format:
  `<locXXXX><locXXXX><locXXXX><locXXXX> class_string ;`
  - Typically (as expected), `class_string` matches one of the classes c1,...,c_{classes_per_call}
  - \<locXXXX\> is a special token type used by Paligemma-2 models family to encode bounding boxes in the “XYXY” format 
  - XXXX takes values from 0 to 1023, inclusive, and represents coordinates relative to the input image sizes 
  - `class_string` may consist of multiple tokens
- The prediction score is calculated as geometric mean of the probabilities of tokens within the `class_string`
- Predictions from all groups and all model calls are combined, and non-maximum suppression (NMS) is applied across all classes.

### Example

**classes_per_call: 5**

The total of 80 COCO2017 classes are divided into 16 groups by 5. 

- **CALL 1 PROMPT**:
`detect person ; bicycle ; car ; motorcycle ; airplane\n`
- **CALL 2 PROMPT**:
`detect bus ; train ; truck ; boat ; traffic light\n`
- **CALL 3 PROMPT**:
`detect fire hydrant ; stop sign ; parking meter ; bench ; bird\n`
- ... 
- **CALL 16 PROMPT**:
`detect vase ; scissors ; teddy bear ; hair drier ; toothbrush\n`

All predictions (e.g. `<loc0001><loc0123><loc0530><loc1012> stop sign`) are then extracted 
from all the 16 model outputs (which can naturally be imperfect and of arbitrary length).
Class scores (e.g. for class `stop sign`) are calculated as geometric mean of generation probabilities
of tokens in `stop sign` (e.g. a token for `stop` and a token for ` sign`).

In practice scores parsing implementation is a bit more complex because on the one hand it has to account for generation imperfections
(e.g. `<loc0001><loc0123><loc0530><loc1012> truck ; <loc0031><loc0900><loc0127><loc1000> ;` - with no class predicted for the second bounding box), structure breaks etc. 
On the other hand the tokenizer itself may have its own rules set affecting the way how `class_string` 
will be split into tokens and how their scores must be extracted.

## Results

There is a number of interesting questions related to how Paligemma-2 models perform on object detection task.
In the present work we'll try to find answers to them.

- How does the `classes_per_call` value affect the quality of detection?
- How is the `mix` series of Paligemma-2 models compares to the `pt` series?
- How is detection quality affected by the model size?
- How is detection quality affected by the model input resolution?

In all cases the `COCO2017 validation mAP@[IoU=0.50:0.95]` is used as object detection quality assessment metric.

### Mix vs Pt

According to [official blog](https://github.com/huggingface/blog/blob/main/paligemma2mix.md),
"pt checkpoints are meant to be fine-tuned on a downstream task and were released for that purpose",
whereas "the mix models give a quick idea of the performance one would get when fine-tuning the pre-trained checkpoints on a downstream task".

![10b-448 models](../assets/images/mix_vs_pt_10b.png)

![3b-448 models](../assets/images/mix_vs_pt_3b.png)

![10b-448 models verbosity](../assets/images/mix_vs_pt_verbosity.png)

<div style="display: flex;">
  <img src="../assets/images/n_objects_dist_mix.png" alt="Number of objects distribution, 10b-mix-448" width="500">
  <img src="../assets/images/n_objects_dist_pt.png" alt="Number of objects distribution, 10b-pt-448" width="500">
</div>

- `mix` models appear to be more verbose than `pt`
- interestingly the detection quality of `pt` models peaks at `classes_per_call=1` whereas for `mix` models it peaks at `classes_per_call=10`

### Classes_per_call

- As the value of `classes_per_call` increases, there is a tendency for model verbosity to grow
  - Longer text input is naturally correlated with longer output, so more detections are generated for more input classes
- Every single detection call with a large `classes_per_call` becomes more difficult for model to process, which leads to an increased risk of false positives and undetected objects
- With a large `classes_per_call` a lot of predictions are duplicated (the spread before/after NMS explodes)

### Scores distribution

Model is usually pretty sure about the predicted classes as the 10th quantile is at 0.82 and the 20th is at 0.96.

### Input Image Resolution

![input resolution table](../assets/images/input_resolution_table.png)

<div style="display: flex;">
  <img src="../assets/images/resolution_sizes_distr_1.png" alt="classes_per_call=1" width="500">
  <img src="../assets/images/resolution_sizes_distr_4.png" alt="classes_per_call=4" width="500">
</div>

- As `classes_per_call` to 8, the `10b-pt-224` produces too many false positives (especially smaller objects), which degrades the metric
  - this could be because low resolution makes it difficult for model to distinguish between small objects
- `10b-pt-896` detects uniformly more smaller objects than `10b-pt-448` without significant performance degradation

### Model Size

![model size table](../assets/images/model_size_table.png)

- `28B-pt-448` model performs poorly on object detection task
  - According to the [Paligemma-2 paper](https://arxiv.org/pdf/2412.03555), "a possible factor related to the relatively worse transferability of PaliGemma 2 28B is that the underlying Gemma 2 27B model is trained from scratch, as opposed to the 2B and 9B models, which are distilled"
- In terms of open class object detection capabilities the `10b-448` model performs slightly better than the `3b-448` model