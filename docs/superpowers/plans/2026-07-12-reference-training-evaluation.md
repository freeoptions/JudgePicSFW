# Reference Training And Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use long-lived desktop wallpaper folders as reference-only training data, exclude temporary phone wallpapers from training, add a fixed evaluation set, and allow the personal model to become primary only after measured performance is good enough.

**Architecture:** Keep images in their existing folders and store only metadata, paths, labels, and hashes. Add training-root settings and evaluation metadata to the existing app state, update sample import/training selection to filter to durable training roots, and add scoring helpers that can decide whether a personal model is eligible to lead classification.

**Tech Stack:** WPF/.NET 8, JSON app state, existing console test project, Python LoRA scripts

---

### Task 1: Add settings and metadata models

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Models\PersonalAiModelSettings.cs`
- Create: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Models\PersonalAiEvaluationModels.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Models\PersistedAppState.cs`

- [ ] Add durable SFW/NSFW training root settings.
- [ ] Add evaluation sample/result records stored by metadata only.
- [ ] Keep existing data compatible by using empty defaults.

### Task 2: Add test-first helper behavior

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW.Tests\Program.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\PersonalAiTrainingService.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\WorkspaceService.cs`

- [ ] Write failing tests for durable-root filtering, no-copy training records, evaluation eligibility, and generated evaluation sample limits.
- [ ] Run the test project and verify the new tests fail.
- [ ] Implement the minimal public/internal helpers.
- [ ] Run the test project and verify it passes.

### Task 3: Wire training roots into import and training

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\WorkspaceService.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\PersonalAiTrainingService.cs`

- [ ] When recording samples, exclude paths outside the configured durable SFW/NSFW roots if roots are configured.
- [ ] Keep sample-library imports valid because they represent the desktop wallpaper training source.
- [ ] Do not copy image files.

### Task 4: Add fixed evaluation set and report data

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\WorkspaceService.cs`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Models\PersonalAiEvaluationModels.cs`

- [ ] Generate a deterministic balanced evaluation set from durable training roots.
- [ ] Store only content id, path, label, source, and timestamps.
- [ ] Add evaluation summary fields for accuracy, SFW false positives, NSFW false negatives, uncertain rate, sample counts, and eligibility.

### Task 5: Let strong personal models become primary

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\Services\WorkspaceService.cs`

- [ ] Add conservative eligibility thresholds based on evaluation summary.
- [ ] If eligible, allow high-confidence personal LoRA results to lead.
- [ ] Keep uncertain routing for conflicts and low confidence.

### Task 6: Verification without building exe

**Files:**
- Test: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW.Tests\Program.cs`
- Test: Python helper tests under `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\PersonalAiTools\tests`

- [ ] Run the console test project only; do not publish/build exe.
- [ ] Run Python helper tests that do not download models.
- [ ] Report any build verification deferred because the user asked not to build until `11`.
