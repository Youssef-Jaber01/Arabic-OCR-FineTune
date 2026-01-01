# import torch
# from transformers import AutoModel, AutoProcessor, AutoModelForVision2Seq
# from peft import PeftModel
# from PIL import Image

# # ================= CONFIG =================
# BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# LORA_ADAPTER_PATH = r"C:\Users\yousof\Desktop\LLM From Scratch\Arabic-OCR\LoRA adapter"
# IMAGE_PATH = r"C:\Users\yousof\Desktop\LLM From Scratch\Arabic-OCR\Images\Clear.jpg"

# PROMPT = "Extract the whole text in the image without any additions"
# # =========================================

# def load_model():
#     model = AutoModel.from_pretrained(
#         BASE_MODEL_ID,
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )

#     model = PeftModel.from_pretrained(
#         model,
#         LORA_ADAPTER_PATH,
#         is_trainable=False
#     )

#     model.eval()
#     return model

# def load_processor():
#     return AutoProcessor.from_pretrained(BASE_MODEL_ID)

# def run_inference(model, processor, image_path, prompt):
#     image = Image.open(image_path).convert("RGB")

#     messages = [
#         {"role": "user", "content": [
#             {"type": "image"},
#             {"type": "text", "text": prompt}
#         ]}
#     ]

#     chat_prompt = processor.tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = processor(
#         images=image,
#         text=chat_prompt,
#         return_tensors="pt"
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=2000
#         )

#     prediction = processor.batch_decode(
#         outputs,
#         skip_special_tokens=True
#     )[0]

#     return prediction

# if __name__ == "__main__":
#     model = load_model()
#     processor = load_processor()
#     result = run_inference(model, processor, IMAGE_PATH, PROMPT)
#     print("\n===== OCR OUTPUT =====\n")
#     print(result)

##################################################################################################
##################################################################################################

import os
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_ADAPTER_PATH = r"C:\Users\yousof\Desktop\LLM From Scratch\Arabic-OCR\LoRA adapter"
IMAGE_PATH = r"C:\Users\yousof\Desktop\LLM From Scratch\Arabic-OCR\Images\Clear.jpg"
PROMPT = "Extract the whole text in the image without any additions"
MAX_NEW_TOKENS = 2000


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_processor():
    device = pick_device()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()
    model.to(device)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    return model, processor, device


def run_inference(model, processor, image_path: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    out = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return out


if __name__ == "__main__":
    model, processor, device = load_model_and_processor()
    print(f"Using device: {device}")

    result = run_inference(model, processor, IMAGE_PATH, PROMPT)
    print("\n===== OCR OUTPUT =====\n")
    print(result)
