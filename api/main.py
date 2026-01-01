from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import io
from pathlib import Path
import traceback
from qwen_vl_utils import process_vision_info
import logging
import time
import uuid

app = FastAPI(title="Arabic OCR API", debug=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("arabic-ocr")

@app.get("/")
def root():
    return {"message": "OCR API is running"}

DEVICE = "cpu"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

API_DIR = Path(__file__).resolve().parent
LORA_PATH = (API_DIR.parent / "LoRA adapter").resolve()

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map=None,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.load_adapter(str(LORA_PATH))
model.to(DEVICE)
model.eval()

print("Model loaded successfully")

logger.info("Model and LoRA adapter loaded successfully")

def extract_ocr_text(model_output: str) -> str:
    """
    Extract only the assistant's OCR text from Qwen output.
    """
    if "assistant" in model_output:
        return model_output.split("assistant")[-1].strip()
    return model_output.strip()


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        logger.info(f"[{request_id}] OCR request received")

        if not file.content_type.startswith("image/"):
            logger.warning(f"[{request_id}] Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            inputs = processor(
                images=image,
                text="Extract all Arabic text from the image accurately.",
                return_tensors="pt"
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512
                )

            text = processor.batch_decode(output, skip_special_tokens=True)[0]

            duration = time.perf_counter() - start_time
            logger.info(f"[{request_id}] OCR completed in {duration:.2f} seconds")

            return {
                "request_id": request_id,
                "text": text,
                "inference_time_seconds": round(duration, 2)
            }

        except Exception as e:
            logger.exception(f"[{request_id}] OCR failed")
            raise HTTPException(status_code=500, detail="OCR failed")

