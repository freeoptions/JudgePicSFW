using System.IO;
using JudgePicSFW.Models;

namespace JudgePicSFW.ViewModels;

public sealed class SampleFolderItemViewModel : ObservableObject
{
    private bool _isEnabled = true;

    public required string Id { get; init; }

    public required string FolderPath { get; init; }

    public required ImageLabel Label { get; init; }

    public string FolderName => Path.GetFileName(FolderPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

    public string LabelText => Label switch
    {
        ImageLabel.Sfw => "SFW 样本",
        ImageLabel.Nsfw => "NSFW 样本",
        ImageLabel.Person => "人物样本",
        ImageLabel.NonPerson => "非人物样本",
        _ => "未标注样本",
    };

    public string LearningStateText => IsEnabled ? "学习中" : "已停用";

    public bool IsEnabled
    {
        get => _isEnabled;
        set
        {
            if (SetProperty(ref _isEnabled, value))
            {
                RaisePropertyChanged(nameof(LearningStateText));
            }
        }
    }

    public SampleFolderConfig ToConfig()
    {
        return new SampleFolderConfig
        {
            Id = Id,
            FolderPath = FolderPath,
            Label = Label,
            IsEnabled = IsEnabled,
        };
    }
}
