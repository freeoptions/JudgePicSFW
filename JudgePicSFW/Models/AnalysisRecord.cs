namespace JudgePicSFW.Models;

public sealed class AnalysisRecord
{
    public string FilePath { get; init; } = string.Empty;

    public string FileName { get; init; } = string.Empty;

    public string ContentId { get; init; } = string.Empty;

    public string AverageHash { get; init; } = string.Empty;

    public ulong AverageHashBits { get; init; }

    public int Width { get; init; }

    public int Height { get; init; }

    public ClassificationTaskMode TaskMode { get; init; } = ClassificationTaskMode.ContentSafety;

    public ImageLabel SuggestedLabel { get; init; }

    public ImageLabel FinalLabel { get; init; }

    public LabelOrigin LabelOrigin { get; init; }

    public double Confidence { get; init; }

    public bool IsManualCorrection { get; init; }

    public bool IsHumanConfirmed { get; init; }

    public string Explanation { get; init; } = string.Empty;

    public bool UsedHistoryCache { get; init; }

    public bool UsedAiModel { get; init; }

    public bool UsedPersonalLearning { get; init; }

    public bool UsedPersonalLora { get; init; }

    public double GenericNsfwScore { get; init; }

    public double PersonalNsfwScore { get; init; }
}
