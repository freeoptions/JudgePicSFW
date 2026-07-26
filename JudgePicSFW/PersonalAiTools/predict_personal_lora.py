import json
import math
import os
import sys
import time

import torch
from PIL import Image, ImageFile, ImageOps
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, logging as transformers_logging


LABEL_TO_ID = {"sfw": 0, "nsfw": 1}
ID_TO_LABEL = {0: "SFW", 1: "NSFW"}
DEFAULT_BASE_MODEL_ID = "google/vit-base-patch16-224"
LEGACY_BASE_MODEL_IDS = {"google/vit-base-patch16-224-in21k"}
BROKEN_DEFAULT_ENDPOINTS = {"https://hf-mirror.com"}
MAX_PREPROCESS_IMAGE_SIDE = 1024
MAX_PREPROCESS_IMAGE_PIXELS = 4_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_label_token(value, fallback):
    text = str(value or fallback).strip().lower().replace("-", "_")
    return text or fallback


def configure_label_mapping(request):
    global LABEL_TO_ID, ID_TO_LABEL

    primary_label = normalize_label_token(request.get("primaryLabel"), "sfw")
    secondary_label = normalize_label_token(request.get("secondaryLabel"), "nsfw")
    if primary_label == secondary_label:
        raise RuntimeError("primaryLabel and secondaryLabel must be different")

    LABEL_TO_ID = {primary_label: 0, secondary_label: 1}
    ID_TO_LABEL = {0: primary_label.upper(), 1: secondary_label.upper()}


def write_event(name, payload):
    print(f"{name} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", flush=True)


def normalize_huggingface_endpoint():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    if endpoint in BROKEN_DEFAULT_ENDPOINTS:
        os.environ.pop("HF_ENDPOINT", None)
        print("HF_ENDPOINT=hf-mirror.com was ignored; using huggingface.co directly.", flush=True)


def normalize_base_model_id(base_model_id):
    model_id = (base_model_id or DEFAULT_BASE_MODEL_ID).strip()
    if model_id in LEGACY_BASE_MODEL_IDS:
        print(f"Base model {model_id} was replaced with {DEFAULT_BASE_MODEL_ID}.", flush=True)
        return DEFAULT_BASE_MODEL_ID
    return model_id


def load_image_processor(model_or_path):
    try:
        return AutoImageProcessor.from_pretrained(model_or_path)
    except OSError:
        return ViTImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )


def prepare_image_for_processor(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image size is invalid")

    pixel_count = width * height
    max_side = MAX_PREPROCESS_IMAGE_SIDE
    if pixel_count > MAX_PREPROCESS_IMAGE_PIXELS:
        pixel_scale = math.sqrt(MAX_PREPROCESS_IMAGE_PIXELS / pixel_count)
        max_side = min(max_side, max(224, int(max(width, height) * pixel_scale)))

    if max(width, height) > max_side:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_side, max_side), resampling)

    return image


def describe_device(device):
    if str(device).startswith("cuda"):
        try:
            device_name = torch.cuda.get_device_name(0)
        except (AttributeError, RuntimeError):
            device_name = ""
        return "cuda", f"CUDA ({device_name})" if device_name else "CUDA"
    return "cpu", "CPU"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: predict_personal_lora.py <request.json>")

    request = read_json(sys.argv[1])
    normalize_huggingface_endpoint()
    configure_label_mapping(request)
    base_model_id = normalize_base_model_id(request["baseModelId"])
    active_model_path = request["activeModelPath"]
    output_path = request["outputPath"]
    images = request.get("images") or []
    batch_size = max(1, min(64, int(request.get("batchSize") or 8)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.backends.cudnn.benchmark = True
    device_name, device_label = describe_device(device)
    total_images = len(images)
    started_at = time.time()
    processed_count = 0
    skipped_count = 0
    last_report_count = 0
    last_report_at = 0.0
    progress_interval = max(batch_size, 16)

    def report_progress(event_name, current_item="personal-lora", force=False):
        nonlocal last_report_count, last_report_at
        now = time.time()
        should_report = (
            force
            or processed_count >= total_images
            or processed_count - last_report_count >= progress_interval
            or now - last_report_at >= 5
        )
        if not should_report:
            return
        write_event(event_name, {
            "current": processed_count,
            "total": total_images,
            "skipped": skipped_count,
            "currentItem": os.path.basename(current_item) if current_item else "personal-lora",
            "device": device_name,
            "deviceDisplay": device_label,
            "elapsedSeconds": now - started_at,
        })
        last_report_count = processed_count
        last_report_at = now

    report_progress("PREDICT_START", "loading-model", force=True)
    processor = load_image_processor(active_model_path)
    base_model = AutoModelForImageClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base_model, active_model_path)
    model.to(device)
    model.eval()
    report_progress("PREDICT_PROGRESS", "model-loaded", force=True)

    predictions = []

    def is_cuda_out_of_memory(exc):
        return is_cuda and isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def clear_cuda_cache():
        if not is_cuda:
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def finish_batch_items(batch_items, current_item, skipped_increment=0):
        nonlocal processed_count, skipped_count
        skipped_count += skipped_increment
        for entry in batch_items:
            try:
                entry["image"].close()
            except Exception:
                pass
        processed_count += len(batch_items)
        report_progress("PREDICT_PROGRESS", current_item, force=False)

    def append_batch_predictions(batch_items):
        if not batch_items:
            return

        batch_images = [entry["image"] for entry in batch_items]
        encoded = None
        pixel_values = None
        logits = None
        probabilities = None
        try:
            encoded = processor(images=batch_images, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_cuda):
                logits = model(pixel_values=pixel_values).logits
            probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            for entry, item_probabilities in zip(batch_items, probabilities):
                nsfw_probability = float(item_probabilities[1])
                confidence = abs(nsfw_probability - 0.5) * 2.0
                predictions.append({
                    "isAvailable": True,
                    "contentId": entry["contentId"],
                    "filePath": entry["filePath"],
                    "nsfwProbability": nsfw_probability,
                    "confidence": confidence,
                    "message": "ok",
                })
        except Exception as exc:
            if is_cuda_out_of_memory(exc) and len(batch_items) > 1:
                encoded = None
                pixel_values = None
                logits = None
                probabilities = None
                clear_cuda_cache()
                midpoint = max(1, len(batch_items) // 2)
                append_batch_predictions(batch_items[:midpoint])
                append_batch_predictions(batch_items[midpoint:])
                return

            for entry in batch_items:
                predictions.append({
                    "isAvailable": False,
                    "contentId": entry["contentId"],
                    "filePath": entry["filePath"],
                    "nsfwProbability": 0.0,
                    "confidence": 0.0,
                    "message": str(exc),
                })
            finish_batch_items(
                batch_items,
                batch_items[-1]["filePath"] if batch_items else "",
                skipped_increment=len(batch_items))
            clear_cuda_cache()
            return
        finally:
            del encoded, pixel_values, logits, probabilities

        finish_batch_items(batch_items, batch_items[-1]["filePath"] if batch_items else "")

    pending_batch = []
    with torch.inference_mode():
        for item in images:
            file_path = item.get("filePath") or ""
            content_id = item.get("contentId") or ""
            if not os.path.exists(file_path):
                processed_count += 1
                skipped_count += 1
                predictions.append({
                    "isAvailable": False,
                    "contentId": content_id,
                    "filePath": file_path,
                    "nsfwProbability": 0.0,
                    "confidence": 0.0,
                    "message": "file missing",
                })
                report_progress("PREDICT_PROGRESS", file_path, force=False)
                continue

            try:
                with Image.open(file_path) as image:
                    image = prepare_image_for_processor(image)
                pending_batch.append({
                    "contentId": content_id,
                    "filePath": file_path,
                    "image": image,
                })
                if len(pending_batch) >= batch_size:
                    append_batch_predictions(pending_batch)
                    pending_batch = []
            except Exception as exc:
                predictions.append({
                    "isAvailable": False,
                    "contentId": content_id,
                    "filePath": file_path,
                    "nsfwProbability": 0.0,
                    "confidence": 0.0,
                    "message": str(exc),
                })
                processed_count += 1
                skipped_count += 1
                report_progress("PREDICT_PROGRESS", file_path, force=False)

        append_batch_predictions(pending_batch)

    write_json(output_path, {"predictions": predictions})
    report_progress("PREDICT_DONE", "personal-lora", force=True)


if __name__ == "__main__":
    main()
