namespace JudgePicSFW.Models;

public sealed class PersonalAiTrainingRoot
{
    public string FolderPath { get; set; } = string.Empty;

    public ImageLabel Label { get; set; }

    public bool IsEnabled { get; set; } = true;
}

public sealed class PersonalAiEvaluationSample
{
    public string ContentId { get; set; } = string.Empty;

    public string FilePath { get; set; } = string.Empty;

    public string AverageHash { get; set; } = string.Empty;

    public ulong AverageHashBits { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }

    public ImageLabel Label { get; set; }

    public string Source { get; set; } = string.Empty;

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;

    public DateTime UpdatedAtUtc { get; set; } = DateTime.UtcNow;
}

public sealed class PersonalAiEvaluationSummary
{
    public string ModelVersion { get; set; } = string.Empty;

    public int TotalSamples { get; set; }

    public int EvaluatedSamples { get; set; }

    public int CorrectCount { get; set; }

    public int SfwSamples { get; set; }

    public int NsfwSamples { get; set; }

    public int SfwFalsePositiveCount { get; set; }

    public int NsfwFalseNegativeCount { get; set; }

    public int UncertainCount { get; set; }

    public double Accuracy { get; set; }

    public double SfwFalsePositiveRate { get; set; }

    public double NsfwFalseNegativeRate { get; set; }

    public double UncertainRate { get; set; }

    public bool AllowsPersonalModelPrimary { get; set; }

    public DateTime UpdatedAtUtc { get; set; } = DateTime.UtcNow;
}

public sealed class PersonalAiEvaluationReport
{
    public PersonalAiEvaluationSummary Summary { get; set; } = new();

    public List<PersonalAiEvaluationSampleResult> Results { get; set; } = [];
}

public sealed class PersonalAiEvaluationSampleResult
{
    public string ContentId { get; set; } = string.Empty;

    public string FilePath { get; set; } = string.Empty;

    public ImageLabel ExpectedLabel { get; set; }

    public ImageLabel PredictedLabel { get; set; } = ImageLabel.Uncertain;

    public double NsfwProbability { get; set; }

    public double Confidence { get; set; }

    public bool IsAvailable { get; set; }

    public string Message { get; set; } = string.Empty;
}
