from pathlib import Path
import tempfile
from typing import Callable
from functools import partial

from PIL import Image
import torch
import torch.nn.functional as F
from accelerate import PartialState
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import gradio as gr
from ultralytics.utils.plotting import Annotator
from peft import PeftModel

from inference.paligemma_coco_benchmark import PaliGemmaCocoPredictor, run_detection as run_detection_open_class
from inference.paligemma_structured_coco_benchmark import RestrictedPredictor


class PGCocoPredictorDemo(PaliGemmaCocoPredictor):
    def _load_model(self, model_path, use_flash_attention=False):
        return None, None

    def set_model_processor(self, model, processor):
        self.model = model
        self.processor = processor

class PaligemmaGradioDemo:
    def __init__(self, model_path:str|Path, adapter_path: str|Path|None=None):

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_path,
                                                                       dtype=torch.bfloat16,
                                                                       device_map="auto")
        self.adapter_path = adapter_path

        if adapter_path:
            self.model.load_adapter(adapter_path, adapter_name="default")
            self.model.set_adapter("default")
            # self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

        self.processor = PaliGemmaProcessor.from_pretrained(model_path, use_fast=True)

        self.oc_predictor = PGCocoPredictorDemo(Path(''), PartialState(), batch_size=1, classes_per_call=10)
        self.oc_predictor.set_model_processor(self.model, self.processor)

        self.cs_predictor = RestrictedPredictor(self.model, self.processor)

        self.prompt_input = gr.Textbox(label="Prompt input",
                                       placeholder="e.g. `describe en` or `detect cat ; dog; person`")
        self.image_input = gr.Image(label="Image input", type="pil")
        self.run_btn = gr.Button("Run model")
        self.output_text = gr.Textbox(label="Output", interactive=False, max_lines=10)
        self.detections_output = gr.Image(label="Detections", interactive=False)
        self.nms_checkbox = gr.Checkbox(label="NMS", value=False)
        self.apply_adapter = gr.Checkbox(label="Adapter",
                                         value=self.adapter_path is not None,
                                         interactive=self.adapter_path is not None)

        self.detect_coco_btn = gr.Button("Detect COCO")
        self.coco_mode_radio = gr.Radio(label="COCO detection mode",
                                choices=["Open class", "Closed set"] if self.adapter_path else ["Open class"],
                                value="Open class")

        self.slider_classes = gr.Slider(
                label="Classes per call",
                minimum=1,
                maximum=80,
                step=1,
                value=10,
                interactive=True
            )

    def run_coco_inference(self, input_image, mode):
        if mode == "Open class":
            with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
                temp_path = tmp.name
                input_image.save(temp_path)
                _, annotated_img = run_detection_open_class(self.oc_predictor, Path(temp_path))
                return annotated_img
        elif mode == "Closed set":
            return self.cs_predictor.detect_and_show_demo(input_image.copy())
        else:
            raise ValueError(mode)

    def run_model_inference(self,
                            prompt,
                            input_image: Image.Image,
                            apply_nms,
                            max_new_tokens=256,
                            nms_callable: Callable | None = partial(
                                PaliGemmaCocoPredictor.apply_nms_to_predictions,
                                conf_threshold=0.25,
                                iou_threshold=0.7,
                                class_agnostic=False,
                            ),
                            ):
        prompt = "<image>" + prompt
        model_inputs = self.processor(text=prompt,
                                      images=input_image,
                                      return_tensors="pt").to(torch.bfloat16).to(self.model.device)

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs,
                                             max_new_tokens=max_new_tokens,
                                             do_sample=False,
                                             return_dict_in_generate=True,
                                             output_scores=True,
                                             )
            new_tokens_generated = (
                generation.sequences[0][input_len:].detach().cpu().numpy().tolist()
            )

            logits_generated = torch.stack([x[0] for x in generation.scores])

            prob_distribution_generated = (
                F.softmax(logits_generated, dim=-1).detach().cpu().numpy()
            )

            r = self.oc_predictor._parse_predictions(
                new_tokens_generated,
                prob_distribution_generated,
                input_image.size,
            )

            if apply_nms:
                r = nms_callable(r)

            annotated_img = None

            if r:
                annotator = Annotator(
                    input_image.copy(), line_width=1, font_size=11, pil=True
                )


                for x1, y1, x2, y2, score, label in r:
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

            decoded = self.processor.decode(new_tokens_generated, skip_special_tokens=True)

        return decoded, annotated_img

    def set_classes_per_call(self, value):
        self.oc_predictor.classes_per_call = value

    def set_adapter(self, value):
        if self.adapter_path:
            if value:
                self.model.enable_adapters()
                self.model.set_adapter("default")
            else:
                self.model.disable_adapters()

            self.model.eval()

    def build_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Demo paligemma-2")
            self.apply_adapter.render()
            self.prompt_input.render()

            with gr.Row():
                self.image_input.render()
                self.detections_output.render()

            with gr.Row():
                self.run_btn.render()
                self.nms_checkbox.render()

            self.output_text.render()

            self.coco_mode_radio.render()
            self.slider_classes.render()

            self.detect_coco_btn.render()

            self.run_btn.click(
                fn=self.run_model_inference,
                inputs=[self.prompt_input, self.image_input, self.nms_checkbox],
                outputs=[self.output_text, self.detections_output],
            )

            self.detect_coco_btn.click(
                fn=self.run_coco_inference,
                inputs=[self.image_input, self.coco_mode_radio],
                outputs=self.detections_output,
            )

            self.coco_mode_radio.change(
                fn=lambda option: gr.update(interactive=(option == "Open class")),
                inputs=self.coco_mode_radio,
                outputs=self.slider_classes
            )

            self.apply_adapter.change(
                fn=lambda checked: gr.update(
                    choices=["Open class", "Closed set"] if checked else ["Open class"],
                    value="Open class",
                    interactive=True
                ),
                inputs=self.apply_adapter,
                outputs=self.coco_mode_radio
            )

            self.apply_adapter.change(
                fn=self.set_adapter,
                inputs=self.apply_adapter,
            )

            self.slider_classes.change(
                fn=self.set_classes_per_call,
                inputs=self.slider_classes,
            )


        return demo


def main():
    model_path = "/mnt/d/Sergey/ML/models/paligemma_models/paligemma2-10b-pt-448"
    adapter_path = '/mnt/d/Sergey/ML/models/paligemma_models/adapter_last'
    # adapter_path = None
    demo_instance = PaligemmaGradioDemo(model_path, adapter_path)
    interface = demo_instance.build_interface()
    interface.launch()

if __name__ == "__main__":
    main()












