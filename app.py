import os
import re
import traceback
from datetime import datetime
from typing import Any, Literal

import gradio as gr
import numpy as np
import requests
import spaces
import torch
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from transformers import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForImageTextToText
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# --- Configuration ---
MODEL_ID = "prithivMLmods/Herculis-CUA-GUI-Actioner-4B"

# --- Model and Processor Loading (Load once) ---
print(f"Loading model and processor for {MODEL_ID}...")
model = None
processor = None
model_loaded = False
load_error_message = ""


try:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_loaded = True
    print("Model and processor loaded successfully.")
except Exception as e:
    load_error_message = (
        f"Error loading model/processor: {e}\n"
        "This might be due to network issues, an incorrect model ID, or missing dependencies (like flash_attention_2 if enabled by default in some config).\n"
        "Ensure you have a stable internet connection and the necessary libraries installed."
    )
    print(load_error_message)

def array_to_image_path(image_array):
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    # Convert numpy array to PIL Image
    img = Image.fromarray(np.uint8(image_array))

    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"

    # Save the image
    img.save(filename)

    # Get the full path of the saved image
    full_path = os.path.abspath(filename)

    return full_path


LOCALIZATION_PROMPT: str = """Localize an element on the GUI image according to the provided target and output a click position.
          * Only output the click position, do not output any other text.
          * The click position should be in the format 'Click(x, y)' with x: num pixels from the left edge and y: num pixels from the top edge
          Your target is:"""


class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""

    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")


def get_localization_prompt(component, image, step=1):
    """
    Get the prompt for the localization task.
    - component: The component to localize
    - image: The current screenshot of the web page
    - step: The current step of the task
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": LOCALIZATION_PROMPT + "\n" + component},
            ],
        },
    ]


def array_to_image(image_array: np.ndarray) -> Image.Image:
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    # Convert numpy array to PIL Image
    img = Image.fromarray(np.uint8(image_array))
    return img


@spaces.GPU
def run_inference_localization(
    messages_for_template: list[dict[str, Any]], pil_image_for_processing: Image.Image
) -> str:
    model.to("cuda")
    torch.cuda.set_device(0)
    """
    Runs inference using the Holo1 model.
    - messages_for_template: The prompt structure, potentially including the PIL image object
                             (which apply_chat_template converts to an image tag).
    - pil_image_for_processing: The actual PIL image to be processed into tensors.
    """
    # 1. Apply chat template to messages. This will create the text part of the prompt,
    #    including image tags if the image was part of `messages_for_template`.
    text_prompt = processor.apply_chat_template(messages_for_template, tokenize=False, add_generation_prompt=True)

    # 2. Process text and image together to get model inputs
    inputs = processor(
        text=[text_prompt],
        images=[pil_image_for_processing],  # Provide the actual image data here
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 3. Generate response
    # Using do_sample=False for more deterministic output, as in the model card's structured output example
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    # 4. Trim input_ids from generated_ids to get only the generated part
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    # 5. Decode the generated tokens
    decoded_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return decoded_output[0] if decoded_output else ""


# --- Gradio processing function ---
def localize(input_numpy_image: np.ndarray, task: str) -> str:
    # if not model_loaded or not processor or not model:
    #     return f"Model not loaded. Error: {load_error_message}", None
    # if not input_pil_image:
    #     return "No image provided. Please upload an image.", None
    # if not task or task.strip() == "":
    #     return "No task provided. Please type an task.", input_pil_image.copy().convert("RGB")

    # 1. Prepare image: Resize according to model's image processor's expected properties
    #    This ensures predicted coordinates match the (resized) image dimensions.
    input_pil_image = array_to_image(input_numpy_image)
    assert isinstance(input_pil_image, Image.Image)
    image_proc_config = processor.image_processor
    try:
        resized_height, resized_width = smart_resize(
            input_pil_image.height,
            input_pil_image.width,
            factor=image_proc_config.patch_size * image_proc_config.merge_size,
            min_pixels=image_proc_config.min_pixels,
            max_pixels=image_proc_config.max_pixels,
        )
        # Using LANCZOS for resampling as it's generally good for downscaling.
        # The model card used `resample=None`, which might imply nearest or default.
        # For visual quality in the demo, LANCZOS is reasonable.
        resized_image = input_pil_image.resize(
            size=(resized_width, resized_height),
            resample=Image.Resampling.LANCZOS,  # type: ignore
        )
    except Exception as e:
        print(f"Error resizing image: {e}")
        return f"Error resizing image: {e}", input_pil_image.copy().convert("RGB")

    # 2. Create the prompt using the resized image (for correct image tagging context) and task
    prompt = get_localization_prompt(task, resized_image, step=1)

    print("Prompt:")
    print(prompt)

    # 3. Run inference
    #    Pass `messages` (which includes the image object for template processing)
    #    and `resized_image` (for actual tensor conversion).
    try:
        localization = run_inference_localization(prompt, resized_image)
    except Exception as e:
        print(f"Error during model inference: {e}")
        return f"Error during model inference: {e}", resized_image.copy().convert("RGB")

    # 4) Parse coordinates and draw marker
    output_image_with_click = resized_image.copy().convert("RGB")
    match = re.search(r"Click\((\d+),\s*(\d+)\)", localization)
    if match:
        try:
            x = int(match.group(1))
            y = int(match.group(2))
            draw = ImageDraw.Draw(output_image_with_click)
            radius = max(5, min(resized_width // 100, resized_height // 100, 15))
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, outline="red", width=max(2, radius // 4))
            print(f"Predicted and drawn click at: ({x}, {y}) on resized image ({resized_width}x{resized_height})")
        except Exception as e:
            print(f"Error drawing on image: {e}")
            traceback.print_exc()
    else:
        print(f"Could not parse 'Click(x, y)' from model output: {localization}")

    return localization, output_image_with_click


url = "https://huggingface.co/prithivMLmods/Herculis-CUA-GUI-Actioner-4B/resolve/main/example/example-image.png?download=true"
example_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

with gr.Blocks() as demo:
    gr.Markdown("## **Herculis-CUA-GUI-Actioner💻**")

    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(label="Input UI Image", value=example_image, height=450)
            task_component = gr.Textbox(
                label="Prompt CUA",
                value="Locate the `microsoft/Fara-7B` model.",
                placeholder="Locate the `microsoft/Fara-7B` model.",
                info="Explain which UI component needs to be found.",
            )
            submit_button = gr.Button("Localize", variant="primary")

        with gr.Column():
            output_coords_component = gr.Textbox(label="Localization Step")

            output_image_component = gr.Image(
                type="pil", label="Image with coordinates of the component", height=480, interactive=False
            )

    submit_button.click(
        localize, [input_image_component, task_component], [output_coords_component, output_image_component]
    )

demo.launch(share=True, debug=True)
