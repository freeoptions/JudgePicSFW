namespace JudgePicSFW.Models;

public sealed class WorkspaceFolderConfig
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");

    public string FolderPath { get; set; } = string.Empty;

    public bool IsActive { get; set; } = true;
}
