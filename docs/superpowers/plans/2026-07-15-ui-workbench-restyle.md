# UI Workbench Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the JudgePicSFW WPF interface to closely match the approved polished three-pane preview while preserving existing behavior.

**Architecture:** Keep the existing WPF MVVM structure intact. Update shared visual tokens and control templates in `App.xaml`, then adjust layout density, list rows, progress/log cards, preview surfaces, and scrollbar visibility in `MainWindow.xaml`. Do not change ViewModel, services, commands, data bindings, AI logic, scan logic, tray behavior, or file-moving behavior.

**Tech Stack:** WPF XAML, .NET 8, existing MVVM bindings, existing code-behind selection/scroll handling.

---

### Task 1: Theme Tokens And Control Primitives

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\App.xaml`

- [x] Refine palette tokens to a calm blue-gray workbench with muted status colors.
- [x] Add subtle shadow/elevation resources and compact surface colors.
- [x] Reduce card radius from oversized rounded blocks to restrained desktop radii.
- [x] Improve button hover, pressed, disabled, keyboard focus, and selection readability.
- [x] Add minimal scrollbar style: hidden by default until interaction, narrow thumb, no heavy chrome.
- [x] Keep all existing resource keys used by `MainWindow.xaml` so bindings and styles continue to resolve.

### Task 2: Main Workbench Layout

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\MainWindow.xaml`

- [x] Keep the active three-pane structure and existing bindings.
- [x] Tighten outer margins and pane gaps to match the preview.
- [x] Make left configuration panel clearer and less visually heavy.
- [x] Make middle results panel the primary scanning surface.
- [x] Make right progress/log/preview panel feel like one review workspace.
- [x] Do not modify `MainWindow.xaml.cs`.

### Task 3: Results, Focus, And Status States

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\App.xaml`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\MainWindow.xaml`

- [x] Strengthen `ListBoxItem` selected state with a visible accent rail/border.
- [x] Keep hover distinct from selected state.
- [x] Preserve virtualization and `ScrollIntoView` behavior.
- [x] Make filter buttons read as compact metric chips.
- [x] Keep all Chinese labels natural and unchanged unless they are purely visual helper copy.

### Task 4: Preview, Progress, Logs, And Scrollbars

**Files:**
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\MainWindow.xaml`
- Modify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\App.xaml`

- [x] Give image preview a calm canvas with clear empty state.
- [x] Improve progress card scanability: percentage, stage pill, current file, thin progress bar.
- [x] Make operation logs compact and readable without strong competing colors.
- [x] Set nonessential `ScrollViewer` controls to avoid always-visible scrollbars while keeping mouse wheel, touchpad, and keyboard scrolling.

### Task 5: Verification

**Files:**
- Verify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\App.xaml`
- Verify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\MainWindow.xaml`
- Verify: `D:\MyCodes\@UploadToGithub\JudgePicSFW\JudgePicSFW\JudgePicSFW.csproj`

- [x] Run XAML parse/static inspection by checking XML well-formedness.
- [x] Search touched XAML for missing resource keys and accidental event/command removal.
- [x] Inspect dependency imports and project references; no new dependency should be added.
- [x] Do not run `dotnet build` unless the user separately sends `11`.
