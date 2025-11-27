# Paligemma-2 Closed-Set Object Detection

As discussed earlier, Paligemma-2 models can be naturally used as open class detectors. 
In this part we discuss an approach that allows to fine-tune Paligemma-2 models in the closed-set detection style.

## Making a closed-set detector out of Paligemma-2

Natural language typically comprises an extremely large and complex set of dependencies. In order to adequately embrace this complexity
language models typically consist of a huge amount of parameters. 

Imposing a certain structure on the model's input and output often allows to efficiently leverage its pattern recognition capabilities
in a partially known domain. Our approach is inspired by transfer learning examples from 
the [Paligemma-2 paper](https://arxiv.org/pdf/2412.03555) and by [Pix2seq framework](https://arxiv.org/pdf/2109.10852). 

It consists of two major steps:
- encode 80 COCO classes with a set of special tokens, which simplifies the structure of the model's output
- use restricted decoding to guarantee the imposed output structure at inference time

### \<unusedXX> tokens

Adding new tokens to a pre-trained language model is not the most exciting thing to do. 
It is extremely likely that such a surgery would break its inherent structures, dependencies to say nothing about performance degradation.

The authors of the PaliGemma-2 model family included a set of "unused tokens" within the model’s architecture, which are reserved and integrated into the model structure,
even though they have not yet been assigned any specific semantic meaning.

These tokens are [numbered](https://huggingface.co/google/paligemma2-10b-pt-448/blob/main/tokenizer_config.json) from 7 to 105 inclusive and take the form `<unused0>`, `<unused1>`, `<unused2>`...`<unused98>`.

In this work, we use 81 of them: tokens numbered 7 through 87 (corresponding to `<unused0>` through `<unused80>`). The first 80 are used to represent the 80 COCO classes, while the last one (`<unused80>`) represents the so-called "noise token".

## Model Input and Output Structure

The structure of the model's input and output is as follows:

- The input text prompt is fixed: `detect all classes\n`
- The model generates 256 new tokens 
- The last token of the 256 is `<eos>`
- The first 255 tokens are divided into 51 groups of 5 tokens each 
- Each group has the structure `<locAAAA><locBBBB><locCCCC><locDDDD><unusedXX>`, where the first 4 tokens encode the bounding box as before, 
and the last token matches one of the 81 selected <unused> tokens above

This structure uses a limited number of tokens and eliminates the diversity of natural language, simplifies dependencies, and allows fine-tuning the model for detecting a fixed set of classes

### During Training

- The training dataset is built based on COCO2017 train annotations 
- The vast majority of images contain fewer than 51 objects 
- For each object in the annotation set, a corresponding combination of the form `<locAAAA><locBBBB><locCCCC><locDDDD><unusedXX>` is constructed 
- All such combinations for a given image are shuffled and fill N of the required 51 target combinations 
- The remaining max(0, 51-N) combinations are padded with strings of the form `<locAAAA><locBBBB><locCCCC><locDDDD><unused80>`,
where AAAA, BBBB, CCCC, and DDDD are chosen, when possible, so that their IoU with any of the max(0, 51-N) bounding boxes does not exceed a threshold of 0.15 
- The 255 target tokens are appended with a final `<eos>`

**Input**: `<image>detect all classes\n`

**(Tokenized) Output**: `<locAAAA_1><locBBBB_1><locCCCC_1><locDDDD_1><unusedXX_1><locAAAA_2><locBBBB_2><locCCCC_2><locDDDD_2><unusedXX_2>...<locAAAA_51><locBBBB_51><locCCCC_51><locDDDD_51><unusedXX_51><eos>`

### During Inference

- Restricted decoding is implemented 
- The same 51 groups of the form `<locAAAA><locBBBB><locCCCC><locDDDD><unusedXX>` + `<eos>` are generated 
- The difference during inference is that the token `<unused80>` participates in computing the probability distribution for every fifth token, but cannot actually be generated 
- Instead of `<unused80>`, even when it is the most probable among all 81 `<unusedXX>` tokens, the next most probable `<unusedXX>` token is generated 
- This ensures exactly 51 groups are generated without the risk of premature termination 
- The probabilities of the generated `<unusedXX>` tokens are used as scores for the corresponding bounding boxes

## Training

The model `paligemma2-10b-pt-448` is fine-tuned with a LoRA adapter and an effective batch size of 32. 
Training employs the AdamW optimizer along with a cosine learning rate scheduler, with the learning rate ranging from 1e-6 to 1e-5.

The data pipeline is implemented with Ray Data, while distributed training uses Ray Train, which internally leverages the Accelerate library.

![learning curve](../assets/images/learning_curve.png)

![validation](../assets/images/validation.png)

## Results

### COCO benchmark

The fine-tuned `paligemma2-10b-pt-448` model achieves a `COCO2017_val mAP@[IoU=0.50:0.95]` of **0.296**,
which is an improvement of more than 7 percentage points over the open-set configuration value.

### Score distribution

![Score distribution](../assets/images/score_cs_distr.png)

- Deduplicated bounding boxes exhibit a score distribution peak near zero, which may result from noise token replacement during inference

### Detections number distribution

![number of detections distribution](../assets/images/cs_objnum_distr.png)

- The model tends to produce redundant predictions compared to the ground truth which most likely contribute to false positives

### Detection size distribution

![Detection size distribution](../assets/images/cs_sizes_distr.png)

- A huge peak and gap relative to the ground truth near zero indicate that the model predicts a large number of unnecessary small bounding boxes, most likely contributing to false positives
  - simply filtering out all detections smaller than 15 pixels helps boost the metric value by 0.6 percentage point
- smaller peaks towards the right repeat the same behavior observed in the open class mode and correspond to cases where
the predicted bounding box covers the entire image 




