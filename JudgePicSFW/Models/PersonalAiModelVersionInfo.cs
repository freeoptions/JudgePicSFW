namespace JudgePicSFW.Models;

public sealed class PersonalAiModelVersionInfo
{
    public string Version { get; init; } = string.Empty;

    public int TrainingSamples { get; init; }

    public double? EvaluationAccuracy { get; init; }

    public int EvaluatedSamples { get; init; }

    public string CreatedAt { get; init; } = string.Empty;

    public bool IsActive { get; init; }

    public bool IsCandidate { get; init; }

    public bool CanActivate => !IsActive;

    public string StatusText => IsActive
        ? "当前使用"
        : IsCandidate
            ? "候选模型"
            : "可切换";

    public string SummaryText => EvaluationAccuracy.HasValue
        ? $"准确率 {EvaluationAccuracy:P1} · 训练 {TrainingSamples} 张"
        : $"暂无准确率 · 训练 {TrainingSamples} 张";

    public string ActionText => IsCandidate ? "启用" : "切换";
}
