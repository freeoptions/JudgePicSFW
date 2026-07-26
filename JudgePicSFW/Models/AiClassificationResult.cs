namespace JudgePicSFW.Models;

public sealed class AiClassificationResult
{
    public bool IsAvailable { get; init; }

    public string ModelId { get; init; } = string.Empty;

    public double NsfwScore { get; init; }

    public double SfwScore { get; init; }

    public IReadOnlyDictionary<string, double> LabelScores { get; init; } = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);

    public string Message { get; init; } = string.Empty;
}
