namespace JudgePicSFW.Models;

public enum FileMoveStatus
{
    Moved,
    RenamedDueToConflict,
    SourceMissing,
}

public sealed class FileMoveResult
{
    public string SourcePath { get; init; } = string.Empty;

    public string DestinationPath { get; init; } = string.Empty;

    public FileMoveStatus Status { get; init; }

    public string Message { get; init; } = string.Empty;
}

public sealed class ConfirmedMoveSummary
{
    public List<FileMoveResult> Results { get; } = [];
}
