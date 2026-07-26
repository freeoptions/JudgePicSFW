namespace JudgePicSFW.Models;

public sealed class SampleFolderConfig
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");

    public string FolderPath { get; set; } = string.Empty;

    public ImageLabel Label { get; set; }

    public bool IsEnabled { get; set; } = true;
}
