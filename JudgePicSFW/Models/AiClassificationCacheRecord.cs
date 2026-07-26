namespace JudgePicSFW.Models;

public sealed class AiClassificationCacheRecord
{
    public string ContentId { get; set; } = string.Empty;

    public string ModelId { get; set; } = string.Empty;

    public double NsfwScore { get; set; }

    public double SfwScore { get; set; }

    public Dictionary<string, double> LabelScores { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public DateTime UpdatedAtUtc { get; set; }
}
