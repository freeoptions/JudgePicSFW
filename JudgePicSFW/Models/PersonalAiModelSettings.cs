namespace JudgePicSFW.Models;

public sealed class PersonalAiModelSettings
{
    public bool IsEnabled { get; set; } = true;

    public bool AutoTrainEnabled { get; set; } = false;

    public string PythonPath { get; set; } = string.Empty;

    public string RootFolder { get; set; } = string.Empty;

    public string BaseModelId { get; set; } = "google/vit-base-patch16-224";

    public int MaxModelVersions { get; set; } = 2;

    public int TrainEpochs { get; set; } = 3;

    public int TrainBatchSize { get; set; } = 2;

    public int TriggerCorrectionCount { get; set; } = 50;

    public double LearningRate { get; set; } = 0.0002d;

    public double DecisionThreshold { get; set; } = 0.5d;

    public double MinimumConfidence { get; set; } = 0.56d;

    public List<PersonalAiTrainingRoot> TrainingRoots { get; set; } = [];

    public int EvaluationSamplesPerLabel { get; set; } = 500;

    public double PrimaryModelMinimumAccuracy { get; set; } = 0.92d;

    public double PrimaryModelMaximumNsfwFalseNegativeRate { get; set; } = 0.04d;

    public double PrimaryModelMaximumSfwFalsePositiveRate { get; set; } = 0.08d;
}
