namespace JudgePicSFW.Models;

public sealed class PersonTaskSettings
{
    public string SourceFolder { get; set; } = string.Empty;

    public string PersonTargetFolder { get; set; } = string.Empty;

    public string NonPersonTargetFolder { get; set; } = string.Empty;

    public List<WorkspaceFolderConfig> SourceFolders { get; set; } = [];

    public List<WorkspaceFolderConfig> PersonTargetFolders { get; set; } = [];

    public List<WorkspaceFolderConfig> NonPersonTargetFolders { get; set; } = [];

    public List<SampleFolderConfig> SampleFolders { get; set; } = [];

    public PersonalAiModelSettings PersonalAi { get; set; } = new();
}
