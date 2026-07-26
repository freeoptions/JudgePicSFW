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

from train_personal_lora import build_source_fingerprint, build_tensor_cache_path, describe_device, resolve_loader_worker_counts


def test_build_tensor_cache_path_is_stable_for_same_source(tmp_path: Path):
    source = tmp_path / "a.jpg"
    source.write_bytes(b"fake")

    first = build_tensor_cache_path(tmp_path, str(source))
    second = build_tensor_cache_path(tmp_path, str(source))

    assert first == second
    assert first.suffix == ".pt"


def test_build_tensor_cache_path_changes_with_updated_source_fingerprint(tmp_path: Path):
    source = tmp_path / "a.jpg"
    source.write_bytes(b"first")

    first = build_tensor_cache_path(tmp_path, str(source), build_source_fingerprint(str(source)))
    source.write_bytes(b"second-version")
    second = build_tensor_cache_path(tmp_path, str(source), build_source_fingerprint(str(source)))

    assert first != second


def test_describe_device_returns_cpu_label():
    device, label = describe_device()

    assert device == "cpu"
    assert label == "CPU"


def test_resolve_loader_worker_counts_keeps_validation_single_process():
    train_workers, validation_workers = resolve_loader_worker_counts(8)

    assert train_workers == 4
    assert validation_workers == 0


if __name__ == "__main__":
    temp_root = Path.cwd() / ".tmp-tensor-cache"
    temp_root.mkdir(parents=True, exist_ok=True)
    test_build_tensor_cache_path_is_stable_for_same_source(temp_root)
    test_build_tensor_cache_path_changes_with_updated_source_fingerprint(temp_root)
    test_describe_device_returns_cpu_label()
    test_resolve_loader_worker_counts_keeps_validation_single_process()
