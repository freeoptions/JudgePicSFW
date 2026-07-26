namespace JudgePicSFW.Models;

public sealed class WorkspaceProgressInfo
{
    public WorkspaceProgressStage Stage { get; init; }

    public string Title { get; init; } = string.Empty;

    public string Detail { get; init; } = string.Empty;

    public string CurrentItem { get; init; } = string.Empty;

    public int Current { get; init; }

    public int Total { get; init; }
}
