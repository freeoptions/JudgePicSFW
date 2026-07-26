namespace JudgePicSFW.Models;

public sealed class PersonalAiTrainingStatus
{
    public bool IsEnabled { get; set; }

    public bool IsEnvironmentReady { get; set; }

    public bool HasActiveModel { get; set; }

    public bool HasCandidateModel { get; set; }

    public bool AllowsPersonalModelPrimary { get; set; }

    public int TrainingSampleCount { get; set; }

    public int SfwSampleCount { get; set; }

    public int NsfwSampleCount { get; set; }

    public string ActiveModelVersion { get; set; } = string.Empty;

    public double? ActiveModelEvaluationAccuracy { get; set; }

    public int ActiveModelEvaluatedSamples { get; set; }

    public string CandidateModelVersion { get; set; } = string.Empty;

    public double? CandidateModelEvaluationAccuracy { get; set; }

    public int CandidateModelEvaluatedSamples { get; set; }

    public List<PersonalAiModelVersionInfo> ModelVersions { get; set; } = [];

    public string RootFolder { get; set; } = string.Empty;

    public string PythonPath { get; set; } = string.Empty;

    public string TrainingDevice { get; set; } = string.Empty;

    public string TrainingDeviceDisplay { get; set; } = string.Empty;

    public bool PreferCudaInstall { get; set; }

    public string TorchCudaVersion { get; set; } = string.Empty;

    public bool HasResumeCheckpoint { get; set; }

    public int ResumeEpoch { get; set; }

    public int ResumeBatch { get; set; }

    public int CachedTensorCount { get; set; }

    public string Message { get; set; } = string.Empty;
}
