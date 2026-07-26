namespace JudgePicSFW.Models;

public sealed class LearnedImageRecord
{
    public string ContentId { get; set; } = string.Empty;

    public string AverageHash { get; set; } = string.Empty;

    public ulong AverageHashBits { get; set; }

    public ClassificationTaskMode TaskMode { get; set; } = ClassificationTaskMode.ContentSafety;

    public ImageLabel CurrentLabel { get; set; } = ImageLabel.Unknown;

    public LabelOrigin LabelOrigin { get; set; } = LabelOrigin.None;

    public string LastKnownPath { get; set; } = string.Empty;

    public int Width { get; set; }

    public int Height { get; set; }

    public int CorrectionCount { get; set; }

    public bool IsHumanConfirmed { get; set; }

    public bool IsLocked { get; set; }

    public DateTime UpdatedAtUtc { get; set; }

    public string LastReason { get; set; } = string.Empty;
}
