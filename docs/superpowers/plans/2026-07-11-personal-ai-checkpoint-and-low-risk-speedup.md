# Personal AI Checkpoint And Low-Risk Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable personal LoRA training with automatic checkpointing, improve training throughput in low-risk ways, and make the UI show accurate device/progress information.

**Architecture:** Extend the Python training script to save and resume structured checkpoints, then surface that state through `PersonalAiTrainingService` into the WPF view model. Keep the current training entrypoints and file layout, but add small companion metadata files under the existing personal AI root so the app can resume safely without changing the user workflow.

**Tech Stack:** WPF/.NET 8, PowerShell install script, Python, PyTorch, transformers, PEFT, local JSON manifests

---

### Task 1: Add a minimal regression harness for resumable training metadata

**Files:**
- Create: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_checkpoint_metadata.py`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\train_personal_lora.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import json

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_checkpoint_metadata.py`

Expected: fail because `build_checkpoint_manifest` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a pure helper in `train_personal_lora.py` that returns a dictionary with:
- `checkpointDir`
- `baseModelId`
- `epoch`
- `batch`
- `totalEpochs`
- `totalBatches`
- `trainSamples`
- `validationSamples`
- `device`
- `cachedSampleCount`

- [ ] **Step 4: Run test to verify it passes**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_checkpoint_metadata.py`

Expected: PASS with no traceback.

### Task 2: Add batch/epoch checkpoint persistence and resume support in Python training

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\train_personal_lora.py`
- Create: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_resume_state.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import json

from train_personal_lora import choose_resume_checkpoint


def test_choose_resume_checkpoint_prefers_latest_batch(tmp_path: Path):
    first = tmp_path / "checkpoint-0001-0004"
    second = tmp_path / "checkpoint-0002-0003"
    first.mkdir()
    second.mkdir()
    (first / "checkpoint-state.json").write_text(json.dumps({"epoch": 1, "batch": 4}), encoding="utf-8")
    (second / "checkpoint-state.json").write_text(json.dumps({"epoch": 2, "batch": 3}), encoding="utf-8")

    chosen = choose_resume_checkpoint(tmp_path)

    assert chosen == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_resume_state.py`

Expected: fail because `choose_resume_checkpoint` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement in `train_personal_lora.py`:
- a stable checkpoints folder under the personal AI root
- checkpoint save at batch interval and epoch end
- `checkpoint-state.json` with epoch, batch, device, dataset stats, cache stats
- optimizer/model/processor save in checkpoint folder
- resume detection that loads the newest valid checkpoint and continues from the next batch

- [ ] **Step 4: Run test to verify it passes**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_resume_state.py`

Expected: PASS.

### Task 3: Add low-risk data pipeline speedups in Python

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\train_personal_lora.py`
- Create: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_tensor_cache.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from train_personal_lora import build_tensor_cache_path


def test_build_tensor_cache_path_is_stable_for_same_source(tmp_path: Path):
    source = tmp_path / "a.jpg"
    source.write_bytes(b"fake")

    first = build_tensor_cache_path(tmp_path, str(source))
    second = build_tensor_cache_path(tmp_path, str(source))

    assert first == second
    assert first.suffix == ".pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_tensor_cache.py`

Expected: fail because `build_tensor_cache_path` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- tensor cache folder under personal AI root
- per-image processed tensor cache keyed by source path + file metadata
- `DataLoader` `num_workers` based on CPU count
- `pin_memory=True` only when device is CUDA
- progress payload adds cached/uncached counts

- [ ] **Step 4: Run test to verify it passes**

Run: `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_tensor_cache.py`

Expected: PASS.

### Task 4: Prefer GPU torch install with CPU fallback

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\install_personal_ai_env.ps1`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\PersonalAiTrainingService.cs`

- [ ] **Step 1: Write the failing test**

Create a tiny PowerShell smoke script expectation in the plan only:

```powershell
powershell -ExecutionPolicy Bypass -File D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\install_personal_ai_env.ps1 -RootFolder D:\temp\JudgePicSFW-PersonalAi-Test
```

Expected now: no explicit GPU-first branch exists.

- [ ] **Step 2: Write minimal implementation**

Update the installer to:
- keep the domestic mirror for generic Python packages
- try installing GPU PyTorch from the official CUDA wheel index first
- if that fails, log the fallback and install CPU wheels
- refresh the ready marker only after a successful install path

Update C# status parsing to surface whether GPU-capable torch was detected.

- [ ] **Step 3: Run the smoke verification**

Run the installer command above only if explicitly asked later, because the current user has not requested dependency reinstallation right now.

Expected: deferred manual verification note recorded.

### Task 5: Surface accurate resume/device/progress state in WPF

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Models\PersonalAiTrainingStatus.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\PersonalAiTrainingService.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\ViewModels\MainViewModel.cs`

- [ ] **Step 1: Write the failing test**

Use a one-off C# smoke assertion later in this session if needed for helper methods; no dedicated test project exists yet.

- [ ] **Step 2: Write minimal implementation**

Add status fields for:
- training device
- whether a resumable checkpoint exists
- latest checkpoint epoch/batch
- cached tensor count

Update UI-facing progress text so it reads as:
- device (`CPU` / `CUDA`)
- epoch and batch progress
- processed batch count
- elapsed time
- whether this run resumed from a checkpoint

- [ ] **Step 3: Run smoke verification**

Run `dotnet build D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\JudgePicSFW.csproj -c Debug`

Expected: build succeeds after code changes.

### Task 6: Make app exit immediate while relying on frequent checkpointing

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\AppBootstrapper.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\ViewModels\MainViewModel.cs`

- [ ] **Step 1: Write the failing test**

Manual behavior target:
- while training is active, choose exit
- app should close immediately
- next launch should show resumable state and continue from the latest checkpoint instead of epoch 1 batch 1

- [ ] **Step 2: Write minimal implementation**

Keep shutdown behavior immediate; do not block for graceful stop.
Only ensure disposable cleanup does not try to wait for training completion and that frequent checkpointing is the safety net.

- [ ] **Step 3: Run smoke verification**

Run `dotnet build D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\JudgePicSFW.csproj -c Debug`

Expected: build succeeds.

### Task 7: Final verification

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\train_personal_lora.py`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\PersonalAiTrainingService.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\ViewModels\MainViewModel.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\install_personal_ai_env.ps1`

- [ ] **Step 1: Run Python regression scripts**

Run:
- `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_checkpoint_metadata.py`
- `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_resume_state.py`
- `python D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests\test_tensor_cache.py`

Expected: PASS for all three.

- [ ] **Step 2: Run .NET build**

Run: `dotnet build D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\JudgePicSFW.csproj -c Debug`

Expected: build succeeds.

- [ ] **Step 3: Manual behavior check**

Verify:
- training status shows `CPU` or `CUDA`
- progress text mentions epoch/batch instead of pretending processed image count
- force-close during training leaves a fresh checkpoint
- next training run resumes from the latest checkpoint

