namespace JudgePicSFW.Models;

public sealed class AiModelSettings
{
    public bool IsEnabled { get; set; } = true;

    public string ModelPath { get; set; } = string.Empty;

    public bool IsNudeDetectorEnabled { get; set; } = true;

    public bool IsGpuAccelerationEnabled { get; set; } = true;

    public string NudeDetectorModelPath { get; set; } = string.Empty;

    public int NudeDetectorInputSize { get; set; } = 320;

    public double DefaultNsfwThreshold { get; set; } = 0.62d;
}
