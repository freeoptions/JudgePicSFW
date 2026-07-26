namespace JudgePicSFW.Models;

public sealed class WorkspaceSettings
{
    public ClassificationTaskMode TaskMode { get; set; } = ClassificationTaskMode.ContentSafety;

    public string SourceFolder { get; set; } = string.Empty;

    public string SfwTargetFolder { get; set; } = string.Empty;

    public string NsfwTargetFolder { get; set; } = string.Empty;

    public AiModelSettings AiModel { get; set; } = new();

    public PersonalAiModelSettings PersonalAi { get; set; } = new();

    public List<WorkspaceFolderConfig> SourceFolders { get; set; } = [];

    public List<WorkspaceFolderConfig> SfwTargetFolders { get; set; } = [];

    public List<WorkspaceFolderConfig> NsfwTargetFolders { get; set; } = [];

    public List<SampleFolderConfig> SampleFolders { get; set; } = [];

    public PersonTaskSettings PersonTask { get; set; } = new();
}
