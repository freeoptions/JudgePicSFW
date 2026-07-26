namespace JudgePicSFW.Models;

public enum WorkspaceProgressStage
{
    Idle = 0,
    Loading = 1,
    RefreshingSamples = 2,
    ExtractingAiFeatures = 3,
    PreparingSource = 4,
    AnalyzingSource = 5,
    ApplyingCorrection = 6,
    MovingFiles = 7,
    DownloadingModel = 8,
    TrainingPersonalModel = 9,
    Completed = 10,
}
