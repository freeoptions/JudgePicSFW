from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_training_import_stubs():
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_module.device = lambda value: value
    torch_module.optim = types.SimpleNamespace(AdamW=object)
    sys.modules.setdefault("torch", torch_module)

    torch_nn = types.ModuleType("torch.nn")
    torch_nn_functional = types.ModuleType("torch.nn.functional")
    torch_nn.functional = torch_nn_functional
    sys.modules.setdefault("torch.nn", torch_nn)
    sys.modules.setdefault("torch.nn.functional", torch_nn_functional)

    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = object
    torch_utils_data.Dataset = object
    torch_utils.data = torch_utils_data
    sys.modules.setdefault("torch.utils", torch_utils)
    sys.modules.setdefault("torch.utils.data", torch_utils_data)

    pil_module = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image_file = types.ModuleType("PIL.ImageFile")
    pil_image_file.LOAD_TRUNCATED_IMAGES = False
    pil_module.Image = pil_image
    pil_module.ImageFile = pil_image_file
    pil_module.UnidentifiedImageError = RuntimeError
    sys.modules.setdefault("PIL", pil_module)
    sys.modules.setdefault("PIL.Image", pil_image)
    sys.modules.setdefault("PIL.ImageFile", pil_image_file)

    peft_module = types.ModuleType("peft")
    peft_module.LoraConfig = object
    peft_module.PeftModel = types.SimpleNamespace(from_pretrained=lambda model, path, is_trainable=False: model)
    peft_module.get_peft_model = lambda model, config: model
    sys.modules.setdefault("peft", peft_module)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoImageProcessor = object
    transformers_module.AutoModelForImageClassification = object
    transformers_module.ViTImageProcessor = object
    transformers_module.logging = types.SimpleNamespace(
        set_verbosity_error=lambda: None,
        disable_progress_bar=lambda: None,
    )
    sys.modules.setdefault("transformers", transformers_module)


_install_training_import_stubs()

from train_personal_lora import build_checkpoint_manifest


def test_build_checkpoint_manifest_marks_latest_batch_position(tmp_path: Path):
    manifest = build_checkpoint_manifest(
        checkpoint_dir=tmp_path / "checkpoint-0003-0012",
        base_model_id="google/vit-base-patch16-224",
        epoch=3,
        batch=12,
        total_epochs=5,
        total_batches=40,
        train_samples=156,
        validation_samples=18,
        device="cuda",
        cached_sample_count=120,
    )

    assert manifest["epoch"] == 3
    assert manifest["batch"] == 12
    assert manifest["device"] == "cuda"
    assert manifest["cachedSampleCount"] == 120


def test_build_checkpoint_manifest_preserves_completion_flag(tmp_path: Path):
    manifest = build_checkpoint_manifest(
        checkpoint_dir=tmp_path / "checkpoint-latest",
        base_model_id="google/vit-base-patch16-224",
        epoch=5,
        batch=40,
        total_epochs=5,
        total_batches=40,
        train_samples=156,
        validation_samples=18,
        device="cpu",
        cached_sample_count=156,
        is_complete=True,
    )

    assert manifest["isComplete"] is True


if __name__ == "__main__":
    temp_root = Path.cwd() / ".tmp-checkpoint-metadata"
    temp_root.mkdir(parents=True, exist_ok=True)
    test_build_checkpoint_manifest_marks_latest_batch_position(temp_root)
    test_build_checkpoint_manifest_preserves_completion_flag(temp_root)
