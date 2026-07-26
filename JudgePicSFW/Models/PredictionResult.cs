namespace JudgePicSFW.Models;

public sealed class PredictionResult
{
    public required ImageLabel SuggestedLabel { get; init; }

    public required ImageLabel FinalLabel { get; init; }

    public required LabelOrigin LabelOrigin { get; init; }

    public required double Confidence { get; init; }

    public required string Explanation { get; init; }

    public bool IsManualCorrection { get; init; }

    public bool IsHumanConfirmed { get; init; }

    public bool RequiresAiReview { get; init; }

    public bool UsedHistoryCache { get; init; }

    public bool UsedAiModel { get; init; }

    public bool UsedPersonalLearning { get; init; }

    public bool UsedPersonalLora { get; init; }

    public double GenericNsfwScore { get; init; }

    public double PersonalNsfwScore { get; init; }
}
