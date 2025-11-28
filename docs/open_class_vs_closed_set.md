
## Open Class vs Closed-Set Object Detection

Open class detection models can identify objects from both known and novel/unseen classes, adapting to new object categories dynamically, while closed-set detection models are limited to a fixed set of predefined classes they were trained on and cannot recognize objects outside this set.

## YOLOv11 models

Released in 2024 by Ultralytics, YOLOv11 is a relatively recent update to the famous YOLO object detection architecture. 
These models are trained and evaluated on a widely-used [COCO2017](https://docs.ultralytics.com/datasets/detect/coco/) ('Common Objects in Context') dataset which contains annotations for training object detection.
Train2017 contains 118K images, Val2017 contains 5K images which are used for evaluation. COCO2017 comprises 80 object categories in total.

COCO dataset and YOLO series traditionally represent the closed-set object detection setup when the class set (80 object classes in total) is defined before any detection model is created.
Any such model is then trained to detect objects of the pre-defined class set and is unable to output predictions for any class outside the set without a special adjustment.


## Paligemma-2 models

Paligemma-2 is a family of vision-language models released by Google in 2024.
These models can be evaluated on more than 30 tasks, including image captioning and visual question answering (VQA).
Object detection is also among the suite of tasks that Paligemma-2 is designed to perform.

Paligemma-2 flexibility stems from the fact that a vision-language model may be flexibly prompted by means of natural language. 
Some of Paligemma-2 family models were trained specifically to solve the open class detection task, allowing user-driven detection via natural language prompts.

This work focuses on:

- implementation of an evaluation framework allowing to benchmark Paligemma-2 models on COCO val2017
- fine-tuning of Paligemma-2 models on COCO train2017 as a closed-set object detection models


### Paligemma-2 object detection mode input format

A prompt that enables the object detection mode must be of a specific form:
- “detect {object description 1} ; {object description 2} ; {object description 3}...\n”
  - “detect white car ; red car ; car wheel ; house window ; plant\n”

### Paligemma-2 object detection mode output format

Model outputs a set of bounding boxes in the format:
- \<locXXXX\>\<locXXXX\>\<locXXXX\>\<locXXXX\> class_string1 ; \<locXXXX\>\<locXXXX\>\<locXXXX\>\<locXXXX\> class_string2 ; ... ; \<locXXXX\>\<locXXXX\>\<locXXXX\>\<locXXXX\> class_stringN
  - \<locXXXX\> is a special token type used by the Paligemma-2 models family to encode bounding boxes in the “XYXY” format
  - XXXX takes values from 0 to 1023, inclusive, and represents coordinates relative to the input image sizes
  - class_string may consist of multiple tokens


## Roadmap

We first need to reproduce the COCO2017 benchmark results for the well known YOLOv11 family. 
Once a reliable and reproducible benchmark result is achieved, it can be used to evaluate any other model including one from Paligemma-2 family. 