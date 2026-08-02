using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using JudgePicSFW.Models;
using JudgePicSFW.Services;

namespace JudgePicSFW.ViewModels;

public sealed class AnalysisItemViewModel : ObservableObject
{
    private const int PreviewDecodePixelWidth = 420;

    private readonly ImageDecodeCacheService? _imageDecodeCacheService;
    private ImageLabel _currentLabel;
    private LabelOrigin _labelOrigin;
    private double _confidence;
    private bool _isManualCorrection;
    private bool _isHumanConfirmed;
    private bool _hasPendingBatchConfirmation;
    private bool _isReviewedInCurrentBatch;
    private readonly string _fullExplanation;
    private string _explanation;
    private BitmapImage? _previewImage;
    private int _previewLoadVersion;

    public AnalysisItemViewModel(AnalysisRecord record, ImageDecodeCacheService? imageDecodeCacheService = null)
    {
        _imageDecodeCacheService = imageDecodeCacheService;
        FilePath = record.FilePath;
        FileName = record.FileName;
        ContentId = record.ContentId;
        AverageHash = record.AverageHash;
        AverageHashBits = record.AverageHashBits;
        Width = record.Width;
        Height = record.Height;
        TaskMode = record.TaskMode;
        SuggestedLabel = record.SuggestedLabel;
        _currentLabel = record.FinalLabel;
        _labelOrigin = record.LabelOrigin;
        _confidence = record.Confidence;
        _isManualCorrection = record.IsManualCorrection;
        _isHumanConfirmed = record.IsHumanConfirmed;
        _hasPendingBatchConfirmation = false;
        _isReviewedInCurrentBatch = false;
        _fullExplanation = record.Explanation;
        _explanation = string.Empty;
        UsedHistoryCache = record.UsedHistoryCache;
        UsedAiModel = record.UsedAiModel;
        UsedPersonalLearning = record.UsedPersonalLearning;
        UsedPersonalLora = record.UsedPersonalLora;
        GenericNsfwScore = record.GenericNsfwScore;
        PersonalNsfwScore = record.PersonalNsfwScore;
    }

    public string FilePath { get; }

    public string FileName { get; }

    public string ContentId { get; }

    public string AverageHash { get; }

    public ulong AverageHashBits { get; }

    public int Width { get; }

    public int Height { get; }

    public Models.ClassificationTaskMode TaskMode { get; }

    public ImageLabel SuggestedLabel { get; }

    public bool UsedHistoryCache { get; private set; }

    public bool UsedAiModel { get; private set; }

    public bool UsedPersonalLearning { get; private set; }

    public bool UsedPersonalLora { get; private set; }

    public double GenericNsfwScore { get; }

    public double PersonalNsfwScore { get; }

    public ImageLabel CurrentLabel
    {
        get => _currentLabel;
        private set
        {
            if (SetProperty(ref _currentLabel, value))
            {
                RaiseLabelPropertiesChanged();
            }
        }
    }

    public LabelOrigin LabelOrigin
    {
        get => _labelOrigin;
        private set
        {
            if (SetProperty(ref _labelOrigin, value))
            {
                RaiseLabelPropertiesChanged();
            }
        }
    }

    public double Confidence
    {
        get => _confidence;
        private set
        {
            if (SetProperty(ref _confidence, value))
            {
                RaisePropertyChanged(nameof(ConfidenceText));
            }
        }
    }

    public bool IsManualCorrection
    {
        get => _isManualCorrection;
        private set
        {
            if (SetProperty(ref _isManualCorrection, value))
            {
                RaiseLabelPropertiesChanged();
            }
        }
    }

    public bool IsHumanConfirmed
    {
        get => _isHumanConfirmed;
        private set
        {
            if (SetProperty(ref _isHumanConfirmed, value))
            {
                RaiseLabelPropertiesChanged();
            }
        }
    }

    public string Explanation
    {
        get => _explanation;
        private set => SetProperty(ref _explanation, value);
    }

    public string ExplanationSummary => string.IsNullOrWhiteSpace(_fullExplanation)
        ? string.Empty
        : _fullExplanation.Length <= 120
            ? _fullExplanation
            : $"{_fullExplanation[..120]}...";

    public bool HasPendingBatchConfirmation
    {
        get => _hasPendingBatchConfirmation;
        private set => SetProperty(ref _hasPendingBatchConfirmation, value);
    }

    public bool IsReviewedInCurrentBatch
    {
        get => _isReviewedInCurrentBatch;
        private set
        {
            if (SetProperty(ref _isReviewedInCurrentBatch, value))
            {
                RaiseLabelPropertiesChanged();
            }
        }
    }

    public BitmapImage? PreviewImage
    {
        get => _previewImage;
        private set => SetProperty(ref _previewImage, value);
    }

    public string ResolutionText => $"{Width} x {Height}";

    public string ConfidenceText => Confidence > 0d ? $"置信度 {Confidence:P0}" : "置信度 --";

    public string CurrentLabelText => CurrentLabel switch
    {
        ImageLabel.Sfw => "SFW",
        ImageLabel.Nsfw => "NSFW",
        ImageLabel.Person => "人物",
        ImageLabel.NonPerson => "非人物",
        ImageLabel.Uncertain => "不确定",
        _ => "未标注",
    };

    public string LabelOriginText => LabelOrigin switch
    {
        LabelOrigin.SampleLibrary => "来自样本库",
        LabelOrigin.ModelPrediction => "来自自动判断",
        LabelOrigin.AiModel => "来自 AI 视觉模型",
        LabelOrigin.PersonalAi => UsedPersonalLora ? "来自个人 LoRA 大模型" : "来自个人 AI 校准",
        LabelOrigin.ManualCorrection => "来自人工纠正",
        LabelOrigin.ConfirmedMove => "来自最终确认",
        _ => "暂无来源",
    };

    public string StatusText => IsManualCorrection
        ? "已人工纠正"
        : HasPendingBatchConfirmation
            ? "待确认写入"
        : IsReviewedInCurrentBatch
            ? "已确认入库"
            : LabelOriginText;

    public AnalysisRecord ToRecord()
    {
        return new AnalysisRecord
        {
            FilePath = FilePath,
            FileName = FileName,
            ContentId = ContentId,
            AverageHash = AverageHash,
            AverageHashBits = AverageHashBits,
            Width = Width,
            Height = Height,
            TaskMode = TaskMode,
            SuggestedLabel = SuggestedLabel,
            FinalLabel = CurrentLabel,
            LabelOrigin = LabelOrigin,
            Confidence = Confidence,
            IsManualCorrection = IsManualCorrection,
            IsHumanConfirmed = IsHumanConfirmed,
            Explanation = Explanation,
            UsedHistoryCache = UsedHistoryCache,
            UsedAiModel = UsedAiModel,
            UsedPersonalLearning = UsedPersonalLearning,
            UsedPersonalLora = UsedPersonalLora,
            GenericNsfwScore = GenericNsfwScore,
            PersonalNsfwScore = PersonalNsfwScore,
        };
    }

    public void ApplyManualCorrection(ImageLabel newLabel)
    {
        CurrentLabel = newLabel;
        LabelOrigin = LabelOrigin.ManualCorrection;
        IsManualCorrection = true;
        IsHumanConfirmed = true;
        HasPendingBatchConfirmation = false;
        IsReviewedInCurrentBatch = true;
        UsedHistoryCache = false;
        UsedAiModel = false;
        UsedPersonalLearning = false;
        UsedPersonalLora = false;
        _explanation = newLabel == ImageLabel.Uncertain
            ? "当前图片已被你标记为不确定，暂不写入个人学习样本，建议后续继续复核。"
            : "当前标签已被人工纠正，后续相似图片会优先参考这次修正结果。";
        RaisePropertyChanged(nameof(Explanation));
        RaisePropertyChanged(nameof(ExplanationSummary));
    }

    public void MarkAsConfirmed(bool stageForBatchConfirmation)
    {
        IsHumanConfirmed = true;
        HasPendingBatchConfirmation = stageForBatchConfirmation;
        IsReviewedInCurrentBatch = true;
    }

    public void CommitBatchConfirmation()
    {
        IsHumanConfirmed = true;
        if (!IsManualCorrection)
        {
            LabelOrigin = LabelOrigin.ConfirmedMove;
        }

        HasPendingBatchConfirmation = false;
        IsReviewedInCurrentBatch = true;
    }

    public Task LoadPreviewAsync(CancellationToken cancellationToken = default)
    {
        return LoadPreviewAsync(revealExplanation: true, cancellationToken);
    }

    public async Task PreloadPreviewAsync(CancellationToken cancellationToken = default)
    {
        await LoadPreviewAsync(revealExplanation: false, cancellationToken).ConfigureAwait(true);
    }

    private async Task LoadPreviewAsync(bool revealExplanation, CancellationToken cancellationToken)
    {
        if (revealExplanation && string.IsNullOrEmpty(Explanation) && !string.IsNullOrEmpty(_fullExplanation))
        {
            Explanation = _fullExplanation;
        }

        if (PreviewImage is not null || !File.Exists(FilePath))
        {
            return;
        }

        var loadVersion = Interlocked.Increment(ref _previewLoadVersion);
        var filePath = FilePath;
        if (ImageDecodeCacheService.RequiresDedicatedDecoder(filePath))
        {
            if (_imageDecodeCacheService is null)
            {
                return;
            }

            var cachedPath = _imageDecodeCacheService.TryGetCachedJpegPath(filePath, PreviewDecodePixelWidth);
            if (cachedPath is null)
            {
                _ = LoadDedicatedPreviewAfterDecodeAsync(filePath, loadVersion, cancellationToken);
                return;
            }

            filePath = cachedPath;
        }

        try
        {
            var bitmap = await Task.Run(() => CreatePreviewBitmap(filePath, cancellationToken), cancellationToken).ConfigureAwait(true);
            if (cancellationToken.IsCancellationRequested || loadVersion != _previewLoadVersion)
            {
                return;
            }

            PreviewImage = bitmap;
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
            if (!cancellationToken.IsCancellationRequested && loadVersion == _previewLoadVersion)
            {
                PreviewImage = null;
            }
        }
    }

    private async Task LoadDedicatedPreviewAfterDecodeAsync(
        string filePath,
        int loadVersion,
        CancellationToken cancellationToken)
    {
        try
        {
            var cachedPath = await _imageDecodeCacheService!
                .QueuePreviewDecodeAsync(filePath, cancellationToken)
                .ConfigureAwait(true);
            if (string.IsNullOrWhiteSpace(cachedPath) ||
                cancellationToken.IsCancellationRequested ||
                loadVersion != _previewLoadVersion)
            {
                return;
            }

            var bitmap = await Task.Run(() => CreatePreviewBitmap(cachedPath, cancellationToken), cancellationToken)
                .ConfigureAwait(true);
            if (cancellationToken.IsCancellationRequested || loadVersion != _previewLoadVersion)
            {
                return;
            }

            PreviewImage = bitmap;
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
            if (!cancellationToken.IsCancellationRequested && loadVersion == _previewLoadVersion)
            {
                PreviewImage = null;
            }
        }
    }

    public void ReleasePreview()
    {
        Interlocked.Increment(ref _previewLoadVersion);
        if (Explanation == _fullExplanation)
        {
            Explanation = string.Empty;
        }

        if (PreviewImage is null)
        {
            return;
        }

        PreviewImage = null;
    }

    private static BitmapImage? CreatePreviewBitmap(string filePath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!File.Exists(filePath))
        {
            return null;
        }

        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.CreateOptions = BitmapCreateOptions.IgnoreColorProfile | BitmapCreateOptions.IgnoreImageCache;
        bitmap.DecodePixelWidth = PreviewDecodePixelWidth;
        bitmap.UriSource = new Uri(filePath);
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    private void RaiseLabelPropertiesChanged()
    {
        RaisePropertyChanged(nameof(CurrentLabelText));
        RaisePropertyChanged(nameof(LabelOriginText));
        RaisePropertyChanged(nameof(StatusText));
    }
}
