namespace JudgePicSFW.Models;

public sealed class AiModelStatus
{
    public bool IsEnabled { get; init; }

    public bool IsInstalled { get; init; }

    public string ModelPath { get; init; } = string.Empty;

    public string ModelId { get; init; } = string.Empty;

    public string Message { get; init; } = string.Empty;
}
