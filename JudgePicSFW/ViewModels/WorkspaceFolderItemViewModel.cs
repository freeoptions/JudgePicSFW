using System.IO;
using JudgePicSFW.Models;

namespace JudgePicSFW.ViewModels;

public enum WorkspaceFolderKind
{
    Source,
    SfwTarget,
    NsfwTarget,
}

public sealed class WorkspaceFolderItemViewModel : ObservableObject
{
    private bool _isActive = true;

    public required string Id { get; init; }

    public required string FolderPath { get; init; }

    public required WorkspaceFolderKind Kind { get; init; }

    public string FolderName
    {
        get
        {
            var trimmedPath = FolderPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var folderName = Path.GetFileName(trimmedPath);
            return string.IsNullOrWhiteSpace(folderName) ? trimmedPath : folderName;
        }
    }

    public string KindText => Kind switch
    {
        WorkspaceFolderKind.Source => "待分析源",
        WorkspaceFolderKind.SfwTarget => "主要目标",
        WorkspaceFolderKind.NsfwTarget => "次要目标",
        _ => "工作文件夹",
    };

    public string ActiveStateText => IsActive ? "当前启用" : "未启用";

    public bool IsActive
    {
        get => _isActive;
        set
        {
            if (SetProperty(ref _isActive, value))
            {
                RaisePropertyChanged(nameof(ActiveStateText));
            }
        }
    }

    public WorkspaceFolderConfig ToConfig()
    {
        return new WorkspaceFolderConfig
        {
            Id = Id,
            FolderPath = FolderPath,
            IsActive = IsActive,
        };
    }
}
