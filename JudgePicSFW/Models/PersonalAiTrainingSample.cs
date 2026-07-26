namespace JudgePicSFW.Models;

public sealed class PersonalAiTrainingSample
{
    public string FilePath { get; set; } = string.Empty;

    public string ContentId { get; set; } = string.Empty;

    public ImageLabel Label { get; set; }

    public string Source { get; set; } = string.Empty;

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}

public sealed class PersonalAiMistakeSample
{
    public string OriginalFilePath { get; set; } = string.Empty;

    public string TrainingAssetPath { get; set; } = string.Empty;

    public string ContentId { get; set; } = string.Empty;

    public ImageLabel CorrectedLabel { get; set; }

    public ImageLabel PreviousLabel { get; set; }

    public LabelOrigin PreviousOrigin { get; set; }

    public string Source { get; set; } = string.Empty;

    public int CorrectionCount { get; set; } = 1;

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;

    public DateTime UpdatedAtUtc { get; set; } = DateTime.UtcNow;
}
