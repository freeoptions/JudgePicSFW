import json
import math
import os
import random
import hashlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, logging as transformers_logging


LABEL_TO_ID = {"sfw": 0, "nsfw": 1}
ID_TO_LABEL = {0: "SFW", 1: "NSFW"}
INTEGER_LABEL_TO_NAME = {1: "sfw", 2: "nsfw"}
DEFAULT_BASE_MODEL_ID = "google/vit-base-patch16-224"
LEGACY_BASE_MODEL_IDS = {"google/vit-base-patch16-224-in21k"}
BROKEN_DEFAULT_ENDPOINTS = {"https://hf-mirror.com"}
MAX_PREPROCESS_IMAGE_SIDE = 1024
MAX_PREPROCESS_IMAGE_PIXELS = 4_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


@dataclass
class TrainingSample:
    file_path: str
    label: int
    weight: float
    source: str = ""
    created_at_utc: str = ""
    content_id: str = ""


class PersonalImageDataset(Dataset):
    def __init__(self, samples, processor, tensor_cache_root=None, tensor_cache_max_bytes=0):
        self.samples = samples
        self.processor = processor
        self.tensor_cache_root = Path(tensor_cache_root) if tensor_cache_root else None
        self.tensor_cache_max_bytes = int(tensor_cache_max_bytes or 0)
        self.skipped_paths = set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            cache_path = None
            if self.tensor_cache_root is not None:
                self.tensor_cache_root.mkdir(parents=True, exist_ok=True)
                cache_path = build_tensor_cache_path(self.tensor_cache_root, sample.file_path, build_sample_cache_fingerprint(sample))
                if cache_path.exists():
                    try:
                        cached_payload = torch.load(cache_path, map_location="cpu")
                        pixel_values = cached_payload["pixel_values"]
                        touch_cache_file(cache_path)
                    except (OSError, RuntimeError, KeyError, ValueError):
                        pixel_values = preprocess_image_tensor(sample.file_path, self.processor)
                        save_tensor_cache(cache_path, pixel_values, self.tensor_cache_root, self.tensor_cache_max_bytes)
                else:
                    pixel_values = preprocess_image_tensor(sample.file_path, self.processor)
                    save_tensor_cache(cache_path, pixel_values, self.tensor_cache_root, self.tensor_cache_max_bytes)
            else:
                pixel_values = preprocess_image_tensor(sample.file_path, self.processor)

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(sample.label, dtype=torch.long),
                "weights": torch.tensor(sample.weight, dtype=torch.float32),
            }
        except (OSError, UnidentifiedImageError, ValueError, MemoryError) as exc:
            if sample.file_path not in self.skipped_paths:
                self.skipped_paths.add(sample.file_path)
                write_event("TRAIN_SKIP", {"filePath": sample.file_path, "message": str(exc)})
            return None


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
    global LABEL_TO_ID, ID_TO_LABEL, INTEGER_LABEL_TO_NAME

    primary_label = normalize_label_token(request.get("primaryLabel"), "sfw")
    secondary_label = normalize_label_token(request.get("secondaryLabel"), "nsfw")
    if primary_label == secondary_label:
        raise RuntimeError("primaryLabel and secondaryLabel must be different")

    primary_value = int(request.get("primaryLabelValue") or 1)
    secondary_value = int(request.get("secondaryLabelValue") or 2)
    LABEL_TO_ID = {primary_label: 0, secondary_label: 1}
    ID_TO_LABEL = {0: primary_label.upper(), 1: secondary_label.upper()}
    INTEGER_LABEL_TO_NAME = {primary_value: primary_label, secondary_value: secondary_label}


def build_checkpoint_manifest(
    checkpoint_dir,
    base_model_id,
    epoch,
    batch,
    total_epochs,
    total_batches,
    train_samples,
    validation_samples,
    device,
    cached_sample_count,
    is_complete=False,
    dataset_fingerprint="",
    continue_from_model_path="",
    new_sample_count=0,
    replay_sample_count=0,
):
    return {
        "checkpointDir": str(checkpoint_dir),
        "baseModelId": base_model_id,
        "epoch": epoch,
        "batch": batch,
        "totalEpochs": total_epochs,
        "totalBatches": total_batches,
        "trainSamples": train_samples,
        "validationSamples": validation_samples,
        "device": device,
        "cachedSampleCount": cached_sample_count,
        "isComplete": is_complete,
        "datasetFingerprint": dataset_fingerprint,
        "continueFromModelPath": continue_from_model_path,
        "newSampleCount": new_sample_count,
        "replaySampleCount": replay_sample_count,
    }


def preprocess_image_tensor(source_path, processor):
    with Image.open(source_path) as image:
        image = prepare_image_for_processor(image)
        encoded = processor(images=image, return_tensors="pt")
    return encoded["pixel_values"].squeeze(0)


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


def build_source_fingerprint(source_path):
    stats = os.stat(source_path)
    return f"{os.path.abspath(source_path)}|{stats.st_size}|{stats.st_mtime_ns}"


def build_sample_cache_fingerprint(sample):
    if sample.content_id:
        return f"content:{sample.content_id}"
    return build_source_fingerprint(sample.file_path)


def build_tensor_cache_path(cache_root, source_path, fingerprint=None):
    cache_root = Path(cache_root)
    fingerprint = fingerprint or os.path.abspath(source_path)
    cache_key = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return cache_root / f"{cache_key}.pt"


def touch_cache_file(cache_path):
    try:
        os.utime(cache_path, None)
    except OSError:
        pass


def save_tensor_cache(cache_path, pixel_values, cache_root=None, max_bytes=0):
    cache_path = Path(cache_path)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    torch.save({"pixel_values": pixel_values}, temp_path)
    os.replace(temp_path, cache_path)
    if cache_root is not None and max_bytes and max_bytes > 0:
        prune_tensor_cache(cache_root, max_bytes, protected_paths={cache_path})


def prune_tensor_cache(cache_root, max_bytes, protected_paths=None):
    if not cache_root or not max_bytes or max_bytes <= 0:
        return 0

    cache_root = Path(cache_root)
    if not cache_root.exists():
        return 0

    protected = {Path(path).resolve() for path in (protected_paths or set())}
    entries = []
    total_bytes = 0
    for path in cache_root.glob("*.pt"):
        try:
            stats = path.stat()
        except OSError:
            continue
        total_bytes += stats.st_size
        entries.append((path, stats.st_size, stats.st_atime))

    if total_bytes <= max_bytes:
        return 0

    deleted_count = 0
    for path, size, _ in sorted(entries, key=lambda item: item[2]):
        try:
            resolved_path = path.resolve()
        except OSError:
            resolved_path = path
        if resolved_path in protected:
            continue

        try:
            path.unlink()
        except OSError:
            continue

        total_bytes -= size
        deleted_count += 1
        if total_bytes <= max_bytes:
            break

    return deleted_count


def resolve_tensor_cache_max_bytes(value):
    try:
        max_bytes = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, max_bytes)


def count_cached_samples(samples, cache_root):
    if cache_root is None:
        return 0

    cache_root = Path(cache_root)
    if not cache_root.exists():
        return 0

    cached_count = 0
    for sample in samples:
        cache_path = build_tensor_cache_path(cache_root, sample.file_path, build_sample_cache_fingerprint(sample))
        if cache_path.exists():
            cached_count += 1
    return cached_count


def resolve_checkpoint_root(root_folder):
    return Path(root_folder) / "checkpoints"


def resolve_checkpoint_state_path(checkpoint_dir):
    return Path(checkpoint_dir) / "checkpoint-state.json"


def resolve_checkpoint_payload_path(checkpoint_dir):
    return Path(checkpoint_dir) / "training-state.pt"


def choose_resume_checkpoint(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists():
        return None

    candidates = []
    for checkpoint_dir in checkpoint_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue

        state_path = resolve_checkpoint_state_path(checkpoint_dir)
        if not state_path.exists():
            continue

        try:
            state = read_json(state_path)
        except (OSError, json.JSONDecodeError):
            continue

        if bool(state.get("isComplete")):
            continue

        epoch = int(state.get("epoch") or 0)
        batch = int(state.get("batch") or 0)
        candidates.append((epoch, batch, checkpoint_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[-1][2]


def save_training_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    base_model_id,
    epoch,
    batch,
    total_epochs,
    total_batches,
    train_samples,
    validation_samples,
    device,
    cached_sample_count,
    is_complete=False,
    dataset_fingerprint="",
    continue_from_model_path="",
    new_sample_count=0,
    replay_sample_count=0,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelStateDict": model.state_dict(),
            "optimizerStateDict": optimizer.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "baseModelId": base_model_id,
        },
        resolve_checkpoint_payload_path(checkpoint_dir),
    )
    manifest = build_checkpoint_manifest(
        checkpoint_dir=checkpoint_dir,
        base_model_id=base_model_id,
        epoch=epoch,
        batch=batch,
        total_epochs=total_epochs,
        total_batches=total_batches,
        train_samples=train_samples,
        validation_samples=validation_samples,
        device=str(device),
        cached_sample_count=cached_sample_count,
        is_complete=is_complete,
        dataset_fingerprint=dataset_fingerprint,
        continue_from_model_path=continue_from_model_path,
        new_sample_count=new_sample_count,
        replay_sample_count=replay_sample_count,
    )
    write_json(resolve_checkpoint_state_path(checkpoint_dir), manifest)


def load_resume_state(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    payload_path = resolve_checkpoint_payload_path(checkpoint_dir)
    state_path = resolve_checkpoint_state_path(checkpoint_dir)
    if not payload_path.exists() or not state_path.exists():
        return None

    try:
        payload = torch.load(payload_path, map_location="cpu")
        manifest = read_json(state_path)
        return {
            "checkpointDir": checkpoint_dir,
            "payload": payload,
            "manifest": manifest,
        }
    except (OSError, json.JSONDecodeError, RuntimeError, KeyError, ValueError):
        return None


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


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


def load_samples(dataset_path):
    latest_by_path = {}
    with open(dataset_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            path = item.get("filePath") or item.get("FilePath") or ""
            content_id = item.get("contentId") or item.get("ContentId") or ""
            label = item.get("label") or item.get("Label")
            if isinstance(label, int):
                label_name = INTEGER_LABEL_TO_NAME.get(label, "")
            else:
                label_name = normalize_label_token(label, "")
            if label_name not in LABEL_TO_ID or not path or not os.path.exists(path):
                continue
            source = str(item.get("source") or item.get("Source") or "").lower()
            if "mistake" in source:
                weight = 4.0
            elif "manual" in source:
                weight = 3.0
            elif "confirmed" in source:
                weight = 2.0
            else:
                weight = 1.0
            sample_key = f"content:{content_id}" if content_id else f"path:{os.path.abspath(path)}"
            latest_by_path[sample_key] = TrainingSample(
                os.path.abspath(path),
                LABEL_TO_ID[label_name],
                weight,
                source=source,
                created_at_utc=str(item.get("createdAtUtc") or item.get("CreatedAtUtc") or ""),
                content_id=str(content_id),
            )
    return list(latest_by_path.values())


def parse_created_at_utc(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def get_source_priority(source):
    source_text = (source or "").lower()
    if "mistake" in source_text:
        return 5
    if "manual" in source_text:
        return 4
    if "confirmed" in source_text:
        return 3
    if "sample" in source_text:
        return 2
    return 1


def select_replay_samples(samples, replay_count):
    if replay_count <= 0:
        return []

    sfw_pool = sorted(
        [sample for sample in samples if sample.label == 0],
        key=lambda sample: (get_source_priority(sample.source), parse_created_at_utc(sample.created_at_utc) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    nsfw_pool = sorted(
        [sample for sample in samples if sample.label == 1],
        key=lambda sample: (get_source_priority(sample.source), parse_created_at_utc(sample.created_at_utc) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )

    target_per_label = replay_count // 2
    selected = sfw_pool[:target_per_label] + nsfw_pool[:target_per_label]
    remaining_slots = replay_count - len(selected)
    if remaining_slots > 0:
        remainder = sorted(
            sfw_pool[target_per_label:] + nsfw_pool[target_per_label:],
            key=lambda sample: (get_source_priority(sample.source), parse_created_at_utc(sample.created_at_utc) or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        selected.extend(remainder[:remaining_slots])

    return selected


def is_fixed_replay_sample(sample, fixed_replay_markers):
    if not fixed_replay_markers:
        return False

    normalized_path = os.path.normcase(os.path.abspath(sample.file_path))
    return any(str(marker) and str(marker).lower() in normalized_path.lower() for marker in fixed_replay_markers)


def build_incremental_training_subset(samples, previous_model_created_at_utc, replay_count, fixed_replay_markers=None):
    cutoff = parse_created_at_utc(previous_model_created_at_utc)
    if cutoff is None:
        return list(samples), {
            "newSamples": list(samples),
            "replaySamples": [],
            "fixedReplaySamples": [],
            "isIncremental": False,
        }

    fixed_replay_pool = [
        sample for sample in samples
        if is_fixed_replay_sample(sample, fixed_replay_markers or [])
    ]
    ordinary_samples = [
        sample for sample in samples
        if not is_fixed_replay_sample(sample, fixed_replay_markers or [])
    ]
    new_samples = [
        sample for sample in ordinary_samples
        if (parse_created_at_utc(sample.created_at_utc) or datetime.min.replace(tzinfo=timezone.utc)) > cutoff
    ]
    new_paths = {sample.file_path for sample in new_samples}
    replay_pool = fixed_replay_pool if fixed_replay_pool else [sample for sample in ordinary_samples if sample.file_path not in new_paths]
    replay_samples = select_replay_samples(replay_pool, replay_count)

    subset = []
    seen_paths = set()
    for sample in new_samples + replay_samples:
        if sample.file_path in seen_paths:
            continue
        subset.append(sample)
        seen_paths.add(sample.file_path)

    return subset, {
        "newSamples": new_samples,
        "replaySamples": replay_samples,
        "fixedReplaySamples": replay_samples if fixed_replay_pool else [],
        "isIncremental": True,
    }


def build_training_subset_fingerprint(samples, continue_from_model_path):
    digest = hashlib.sha1()
    digest.update((continue_from_model_path or "").encode("utf-8"))
    for sample in sorted(samples, key=lambda item: item.file_path):
        digest.update(sample.file_path.encode("utf-8"))
        digest.update(str(sample.label).encode("utf-8"))
        digest.update((sample.source or "").encode("utf-8"))
        digest.update((sample.created_at_utc or "").encode("utf-8"))
    return digest.hexdigest()


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


def split_samples(samples):
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    for values in by_label.values():
        random.shuffle(values)

    train = []
    validation = []
    for label_samples in by_label.values():
        if len(label_samples) < 4:
            train.extend(label_samples)
            continue
        validation_count = max(1, math.floor(len(label_samples) * 0.12))
        validation.extend(label_samples[:validation_count])
        train.extend(label_samples[validation_count:])
    random.shuffle(train)
    random.shuffle(validation)
    return train, validation


def collate_batch(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in valid_batch]),
        "labels": torch.stack([item["labels"] for item in valid_batch]),
        "weights": torch.stack([item["weights"] for item in valid_batch]),
    }


def evaluate(model, data_loader, device):
    if len(data_loader.dataset) == 0:
        return None
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(dim=-1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
    model.train()
    return correct / max(1, total)


def resolve_lora_target_modules(model):
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    if {"q_proj", "v_proj"}.issubset(module_names):
        return ["q_proj", "v_proj"]
    if {"query", "value"}.issubset(module_names):
        return ["query", "value"]
    raise RuntimeError("LoRA target modules were not found in the base model")


def describe_device():
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except (AttributeError, RuntimeError):
            device_name = ""
        label = "CUDA"
        if device_name:
            label = f"CUDA ({device_name})"
        return "cuda", label
    return "cpu", "CPU"


def resolve_loader_worker_counts(cpu_count):
    if os.name == "nt":
        return 1, 0

    train_workers = min(2, max(0, (cpu_count or 1) - 1))
    validation_workers = 0
    return train_workers, validation_workers


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: train_personal_lora.py <request.json>")

    request = read_json(sys.argv[1])
    normalize_huggingface_endpoint()
    configure_label_mapping(request)
    random.seed(42)
    dataset_path = request["datasetPath"]
    root_folder = Path(request["rootFolder"])
    base_model_id = normalize_base_model_id(request.get("baseModelId"))
    epochs = int(request.get("epochs") or 3)
    batch_size = int(request.get("batchSize") or 8)
    learning_rate = float(request.get("learningRate") or 0.0002)
    continue_from_model_path = str(request.get("continueFromModelPath") or "").strip()
    previous_model_created_at_utc = str(request.get("previousModelCreatedAtUtc") or "").strip()
    replay_sample_count = int(request.get("replaySampleCount") or 0)
    enable_tensor_cache = bool(request.get("enableTensorCache") or False)
    tensor_cache_max_bytes = resolve_tensor_cache_max_bytes(request.get("tensorCacheMaxBytes"))
    fixed_replay_markers = request.get("fixedReplayPathMarkers") or []
    version_prefix = normalize_label_token(request.get("versionPrefix"), "personal-lora")
    checkpoint_root = resolve_checkpoint_root(root_folder)
    checkpoint_dir = checkpoint_root / "checkpoint-latest"
    tensor_cache_root = root_folder / "tensor-cache" if enable_tensor_cache else None
    if tensor_cache_root is not None and tensor_cache_max_bytes > 0:
        prune_tensor_cache(tensor_cache_root, tensor_cache_max_bytes)

    samples = load_samples(dataset_path)
    if len(samples) < 8:
        raise RuntimeError("training samples are not enough")
    if len({sample.label for sample in samples}) < 2:
        raise RuntimeError("both label classes are required")

    selected_samples, selection_summary = build_incremental_training_subset(
        samples,
        previous_model_created_at_utc,
        replay_sample_count,
        fixed_replay_markers=fixed_replay_markers)
    if continue_from_model_path:
        if len(selection_summary["newSamples"]) == 0:
            raise RuntimeError("No new corrected samples were added since the last model, so incremental LoRA training is not needed.")
    else:
        selected_samples = samples
        selection_summary = {
            "newSamples": list(samples),
            "replaySamples": [],
            "fixedReplaySamples": [],
            "isIncremental": False,
        }

    training_subset_fingerprint = build_training_subset_fingerprint(selected_samples, continue_from_model_path)
    train_samples, validation_samples = split_samples(selected_samples)
    device_name, device_label = describe_device()
    device = torch.device(device_name)
    is_cuda = str(device).startswith("cuda")
    worker_count, validation_worker_count = resolve_loader_worker_counts(os.cpu_count())
    cached_sample_count = count_cached_samples(train_samples, tensor_cache_root) + count_cached_samples(validation_samples, tensor_cache_root)
    resume_checkpoint = choose_resume_checkpoint(checkpoint_root)
    resume_state = load_resume_state(resume_checkpoint) if resume_checkpoint else None

    processor = load_image_processor(base_model_id)
    model = AutoModelForImageClassification.from_pretrained(
        base_model_id,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    if continue_from_model_path and os.path.isdir(continue_from_model_path):
        model = PeftModel.from_pretrained(model, continue_from_model_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=resolve_lora_target_modules(model),
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, lora_config)
    train_dataset = PersonalImageDataset(
        train_samples,
        processor,
        tensor_cache_root=tensor_cache_root,
        tensor_cache_max_bytes=tensor_cache_max_bytes)
    validation_dataset = PersonalImageDataset(
        validation_samples,
        processor,
        tensor_cache_root=tensor_cache_root,
        tensor_cache_max_bytes=tensor_cache_max_bytes)
    train_loader_options = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": is_cuda,
        "persistent_workers": False,
        "collate_fn": collate_batch,
    }
    if worker_count > 0:
        train_loader_options["prefetch_factor"] = 1

    train_loader = DataLoader(
        train_dataset,
        **train_loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=validation_worker_count,
        pin_memory=is_cuda,
        persistent_workers=False,
        collate_fn=collate_batch,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, foreach=False)
    resume_epoch = 0
    resume_batch = 0
    resumed_from_checkpoint = False
    if resume_state is not None:
        payload = resume_state["payload"]
        manifest = resume_state["manifest"]
        resume_fingerprint = str(manifest.get("datasetFingerprint") or "")
        resume_continue_path = str(manifest.get("continueFromModelPath") or "")
        if resume_fingerprint != training_subset_fingerprint or resume_continue_path != continue_from_model_path:
            resume_state = None
        else:
            model_state = payload.get("modelStateDict")
            optimizer_state = payload.get("optimizerStateDict")
            if model_state:
                model.load_state_dict(model_state, strict=False)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                move_optimizer_state_to_device(optimizer, device)
            resume_epoch = int(payload.get("epoch") or 0)
            resume_batch = int(payload.get("batch") or 0)
            resumed_from_checkpoint = resume_epoch > 0 or resume_batch > 0
    model.train()
    total_batches = max(1, len(train_loader))
    total_steps = max(1, epochs * total_batches)
    progress_interval = max(1, total_batches // 80)
    training_started_at = time.time()
    last_progress_at = 0.0
    start_epoch = resume_epoch
    start_batch = resume_batch
    if start_epoch > 0 and start_batch >= total_batches:
        start_epoch += 1
        start_batch = 0
    write_event("TRAIN_START", {
        "epochs": epochs,
        "totalBatches": total_batches,
        "totalSteps": total_steps,
        "trainSamples": len(train_samples),
        "validationSamples": len(validation_samples),
        "device": str(device),
        "deviceDisplay": device_label,
        "resumedFromCheckpoint": resumed_from_checkpoint,
        "resumeEpoch": resume_epoch,
        "resumeBatch": resume_batch,
        "workerCount": worker_count,
        "cachedSampleCount": cached_sample_count,
        "newSampleCount": len(selection_summary["newSamples"]),
        "replaySampleCount": len(selection_summary["replaySamples"]),
        "fixedReplaySampleCount": len(selection_summary["fixedReplaySamples"]),
        "isIncremental": selection_summary["isIncremental"],
    })

    for epoch in range(epochs):
        current_epoch = epoch + 1
        if current_epoch < start_epoch:
            continue

        total_loss = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            if current_epoch == start_epoch and batch_index <= start_batch:
                continue
            if batch is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            outputs = model(pixel_values=pixel_values)
            losses = F.cross_entropy(outputs.logits, labels, reduction="none")
            loss = (losses * weights).sum() / torch.clamp(weights.sum(), min=1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                base_model_id=base_model_id,
                epoch=current_epoch,
                batch=batch_index,
                total_epochs=epochs,
                total_batches=total_batches,
                train_samples=len(train_samples),
                validation_samples=len(validation_samples),
                device=device,
                cached_sample_count=cached_sample_count,
                dataset_fingerprint=training_subset_fingerprint,
                continue_from_model_path=continue_from_model_path,
                new_sample_count=len(selection_summary["newSamples"]),
                replay_sample_count=len(selection_summary["replaySamples"]),
            )
            now = time.time()
            is_progress_batch = batch_index == 1 or batch_index == total_batches or batch_index % progress_interval == 0
            if is_progress_batch or now - last_progress_at >= 10:
                completed_steps = ((current_epoch - 1) * total_batches) + batch_index
                write_event("TRAIN_PROGRESS", {
                    "epoch": current_epoch,
                    "epochs": epochs,
                    "batch": batch_index,
                    "totalBatches": total_batches,
                    "completedSteps": min(completed_steps, total_steps),
                    "totalSteps": total_steps,
                    "loss": total_loss / max(1, batch_count),
                    "skippedImages": len(train_dataset.skipped_paths) + len(validation_dataset.skipped_paths),
                    "elapsedSeconds": now - training_started_at,
                    "device": str(device),
                    "deviceDisplay": device_label,
                    "resumedFromCheckpoint": resumed_from_checkpoint,
                    "workerCount": worker_count,
                    "cachedSampleCount": cached_sample_count,
                })
                last_progress_at = now

        if batch_count == 0:
            continue

        save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            base_model_id=base_model_id,
            epoch=current_epoch,
            batch=total_batches,
            total_epochs=epochs,
            total_batches=total_batches,
            train_samples=len(train_samples),
            validation_samples=len(validation_samples),
            device=device,
            cached_sample_count=cached_sample_count,
            dataset_fingerprint=training_subset_fingerprint,
            continue_from_model_path=continue_from_model_path,
            new_sample_count=len(selection_summary["newSamples"]),
            replay_sample_count=len(selection_summary["replaySamples"]),
        )
        write_event("TRAIN_EPOCH", {
            "epoch": current_epoch,
            "epochs": epochs,
            "loss": total_loss / max(1, batch_count),
            "skippedImages": len(train_dataset.skipped_paths) + len(validation_dataset.skipped_paths),
            "elapsedSeconds": time.time() - training_started_at,
            "device": str(device),
            "deviceDisplay": device_label,
            "resumedFromCheckpoint": resumed_from_checkpoint,
            "cachedSampleCount": cached_sample_count,
        })

    validation_accuracy = evaluate(model, validation_loader, device)
    version = version_prefix + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    models_folder = root_folder / "models"
    model_folder = models_folder / version
    model_folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_folder)
    processor.save_pretrained(model_folder)

    manifest = {
        "version": version,
        "modelPath": str(model_folder),
        "baseModelId": base_model_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "trainingSamples": len(samples),
        "selectedTrainingSamples": len(selected_samples),
        "newTrainingSamples": len(selection_summary["newSamples"]),
        "replayTrainingSamples": len(selection_summary["replaySamples"]),
        "fixedReplayTrainingSamples": len(selection_summary["fixedReplaySamples"]),
        "sfwSamples": sum(1 for sample in samples if sample.label == 0),
        "nsfwSamples": sum(1 for sample in samples if sample.label == 1),
        "primaryLabel": ID_TO_LABEL[0],
        "secondaryLabel": ID_TO_LABEL[1],
        "validationAccuracy": validation_accuracy,
        "device": str(device),
        "deviceDisplay": device_label,
        "continueFromModelPath": continue_from_model_path,
        "previousModelCreatedAtUtc": previous_model_created_at_utc,
        "datasetFingerprint": training_subset_fingerprint,
    }
    write_json(model_folder / "model-manifest.json", manifest)
    write_json(root_folder / "candidate-model.json", manifest)
    save_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        base_model_id=base_model_id,
        epoch=epochs,
        batch=total_batches,
        total_epochs=epochs,
        total_batches=total_batches,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        device=device,
        cached_sample_count=cached_sample_count,
        is_complete=True,
        dataset_fingerprint=training_subset_fingerprint,
        continue_from_model_path=continue_from_model_path,
        new_sample_count=len(selection_summary["newSamples"]),
        replay_sample_count=len(selection_summary["replaySamples"]),
    )


if __name__ == "__main__":
    main()
