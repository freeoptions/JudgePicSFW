namespace JudgePicSFW.Models;

public sealed class PersistedAppState
{
    public WorkspaceSettings Settings { get; set; } = new();

    public Dictionary<string, FileCacheRecord> FileCacheByPath { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public Dictionary<string, LearnedImageRecord> LearnedImagesById { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public Dictionary<string, AiClassificationCacheRecord> AiClassificationsById { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public Dictionary<string, PersonalAiEvaluationSample> PersonalAiEvaluationSamplesById { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public PersonalAiEvaluationSummary PersonalAiEvaluationSummary { get; set; } = new();

    public PersonalAiEvaluationReport PersonalAiEvaluationReport { get; set; } = new();

    public Dictionary<string, PersonalAiEvaluationSample> PersonAiEvaluationSamplesById { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public PersonalAiEvaluationSummary PersonAiEvaluationSummary { get; set; } = new();

    public PersonalAiEvaluationReport PersonAiEvaluationReport { get; set; } = new();
}
