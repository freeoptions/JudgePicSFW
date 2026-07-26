from pathlib import Path
import json
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

from train_personal_lora import choose_resume_checkpoint, build_incremental_training_subset, TrainingSample


def test_choose_resume_checkpoint_prefers_latest_batch(tmp_path: Path):
    first = tmp_path / "checkpoint-0001-0004"
    second = tmp_path / "checkpoint-0002-0003"
    first.mkdir()
    second.mkdir()
    (first / "checkpoint-state.json").write_text(json.dumps({"epoch": 1, "batch": 4}), encoding="utf-8")
    (second / "checkpoint-state.json").write_text(json.dumps({"epoch": 2, "batch": 3}), encoding="utf-8")

    chosen = choose_resume_checkpoint(tmp_path)

    assert chosen == second


def test_choose_resume_checkpoint_ignores_completed_state(tmp_path: Path):
    finished = tmp_path / "checkpoint-finished"
    resumable = tmp_path / "checkpoint-resume"
    finished.mkdir()
    resumable.mkdir()
    (finished / "checkpoint-state.json").write_text(json.dumps({"epoch": 5, "batch": 40, "isComplete": True}), encoding="utf-8")
    (resumable / "checkpoint-state.json").write_text(json.dumps({"epoch": 4, "batch": 12, "isComplete": False}), encoding="utf-8")

    chosen = choose_resume_checkpoint(tmp_path)

    assert chosen == resumable


def test_build_incremental_training_subset_uses_new_samples_plus_fixed_replay():
    samples = [
        TrainingSample(file_path=f"old-sfw-{index}.jpg", label=0, weight=1.0, source="confirmed_move", created_at_utc=f"2026-07-10T10:{index:02d}:00+00:00")
        for index in range(120)
    ] + [
        TrainingSample(file_path=f"old-nsfw-{index}.jpg", label=1, weight=1.0, source="manual_correction", created_at_utc=f"2026-07-10T11:{index:02d}:00+00:00")
        for index in range(120)
    ] + [
        TrainingSample(file_path="new-sfw.jpg", label=0, weight=1.0, source="manual_correction", created_at_utc="2026-07-11T12:00:00+00:00"),
        TrainingSample(file_path="new-nsfw.jpg", label=1, weight=1.0, source="manual_correction", created_at_utc="2026-07-11T12:05:00+00:00"),
    ]

    subset, summary = build_incremental_training_subset(samples, "2026-07-11T00:00:00+00:00", 200)

    assert len(summary["newSamples"]) == 2
    assert len(summary["replaySamples"]) == 200
    assert len(subset) == 202
    assert any(sample.file_path == "new-sfw.jpg" for sample in subset)
    assert any(sample.file_path == "new-nsfw.jpg" for sample in subset)


if __name__ == "__main__":
    temp_root = Path.cwd() / ".tmp-resume-state"
    temp_root.mkdir(parents=True, exist_ok=True)
    for child in temp_root.iterdir():
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()
        else:
            child.unlink()
    test_choose_resume_checkpoint_prefers_latest_batch(temp_root)
    for child in temp_root.iterdir():
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()
        else:
            child.unlink()
    test_choose_resume_checkpoint_ignores_completed_state(temp_root)
    test_build_incremental_training_subset_uses_new_samples_plus_fixed_replay()
