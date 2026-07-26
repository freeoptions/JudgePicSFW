namespace JudgePicSFW.Models;

public sealed class PersonalAiPredictionResult
{
    public bool IsAvailable { get; init; }

    public string ContentId { get; init; } = string.Empty;

    public string FilePath { get; init; } = string.Empty;

    public double NsfwProbability { get; init; }

    public double Confidence { get; init; }

    public string ModelVersion { get; init; } = string.Empty;

    public string Message { get; init; } = string.Empty;
}
