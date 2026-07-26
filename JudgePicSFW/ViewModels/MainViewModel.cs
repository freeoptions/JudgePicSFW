using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Windows.Data;
using JudgePicSFW.Commands;
using JudgePicSFW.Models;
using JudgePicSFW.Services;
using Forms = System.Windows.Forms;

namespace JudgePicSFW.ViewModels;

public sealed class MainViewModel : ObservableObject, IDisposable
{
    private const int PreviewCacheRadius = 8;
    private const int PreviewPreloadNeighborCount = 2;
    private readonly WorkspaceService _workspaceService;
    private readonly ICollectionView _resultsView;
    private readonly CancellationTokenSource _lifetimeTokenSource = new();
    private CancellationTokenSource? _settingsSaveDebounceTokenSource;
    private CancellationTokenSource? _activeScanTokenSource;
    private CancellationTokenSource? _toastTokenSource;
    private CancellationTokenSource? _previewLoadTokenSource;
    private WorkspaceSettings _settings = new();
    private bool _isApplyingTaskSettings;

    private string _sourceFolder = string.Empty;
    private string _sfwTargetFolder = string.Empty;
    private string _nsfwTargetFolder = string.Empty;
    private string _statusTitle = "准备就绪";
    private string _statusStageText = "空闲";
    private string _statusMessage = "先补充样本库，再开始分析源文件夹中的图片。";
    private string _statusCurrentItem = "缓存会直接复用，新增图片会自动补充学习结果。";
    private string _aiModelStatusText = "AI 视觉模型状态待检查。";
    private string _aiModelButtonText = "下载 AI 模型包";
    private string _personalAiStatusText = "个人大模型状态待检查。";
    private string _personalAiButtonText = "训练个人大模型";
    private string _activateCandidateModelButtonText = "启用候选模型";
    private string _personalAiStatusBadgeText = "未启用";
    private string _personalAiStatusBadgeDetailText = "训练或启用后生效";
    private string _personalAiModePillText = "微调模型";
    private string _personalAiActiveModelEvaluationText = "评测准确率：暂无评测";
    private string _personalAiCandidateModelVersionText = "候选版本待生成";
    private string _personalAiCandidateModelEvaluationText = "评测准确率：暂无评测";
    private IReadOnlyList<PersonalAiModelVersionInfo> _personalAiModelVersions = [];
    private AiModelSettings _aiModelSettings = new();
    private PersonalAiModelSettings _personalAiSettings = new();
    private int _statusCurrentCount;
    private int _statusTotalCount;
    private int _manualCorrectionsSinceTrainingPrompt;
    private bool _isBusy;
    private bool _isAiModelInstalled;
    private bool _isAiModelDownloading;
    private bool _isPersonalAiTraining;
    private bool _hasActivePersonalAiModel;
    private bool _hasCandidatePersonalAiModel;
    private bool _showPersonalAiCurrentModelState;
    private bool _isToastVisible;
    private AnalysisItemViewModel? _selectedResult;
    private ResultFilter _activeFilter = ResultFilter.All;
    private ClassificationTaskMode _activeTaskMode = ClassificationTaskMode.ContentSafety;
    private double _progressValue;
    private string _toastMessage = string.Empty;
    private OperationLogLevel _toastLevel = OperationLogLevel.Info;
    private int _lastMoveProgressLogCount;
    private int _lastTrainingProgressLogCount;
    private string _lastTrainingProgressLogSignature = string.Empty;
    private WorkspaceProgressStage _lastGeneralProgressLogStage = WorkspaceProgressStage.Idle;
    private int _lastGeneralProgressLogCount = -1;
    private string _lastGeneralProgressLogSignature = string.Empty;

    public MainViewModel(WorkspaceService workspaceService)
    {
        _workspaceService = workspaceService;

        Results = [];
        SourceFolders = [];
        SfwTargetFolders = [];
        NsfwTargetFolders = [];
        SfwSampleFolders = [];
        NsfwSampleFolders = [];
        OperationLogs = [];

        _resultsView = CollectionViewSource.GetDefaultView(Results);
        _resultsView.Filter = FilterResult;

        InitializeCommand = new AsyncRelayCommand(InitializeAsync);
        StartAnalysisCommand = new AsyncRelayCommand(StartAnalysisAsync, CanStartAnalysis);
        ConfirmMoveCommand = new AsyncRelayCommand(ConfirmMoveAsync, CanConfirmMove);
        RefreshSamplesCommand = new AsyncRelayCommand(RefreshSamplesAsync, CanRefreshSamples);
        InstallAiModelCommand = new AsyncRelayCommand(InstallAiModelAsync, CanInstallAiModel);
        TrainPersonalAiCommand = new AsyncRelayCommand(TrainPersonalAiAsync, CanTrainPersonalAi);
        ActivateCandidateModelCommand = new AsyncRelayCommand(ActivateCandidateModelAsync, CanActivateCandidateModel);
        ActivatePersonalAiModelCommand = new AsyncRelayCommand(ActivatePersonalAiModelAsync, CanActivatePersonalAiModel);
        StopScanCommand = new RelayCommand(StopCurrentScan, CanStopCurrentScan);

        AddSourceFolderCommand = new RelayCommand(() => AddWorkspaceFolder(WorkspaceFolderKind.Source));
        AddSfwTargetFolderCommand = new RelayCommand(() => AddWorkspaceFolder(WorkspaceFolderKind.SfwTarget));
        AddNsfwTargetFolderCommand = new RelayCommand(() => AddWorkspaceFolder(WorkspaceFolderKind.NsfwTarget));
        ActivateWorkspaceFolderCommand = new RelayCommand(parameter => ActivateWorkspaceFolder(parameter as WorkspaceFolderItemViewModel));
        RemoveWorkspaceFolderCommand = new RelayCommand(parameter => RemoveWorkspaceFolder(parameter as WorkspaceFolderItemViewModel));
        AddSfwSampleFolderCommand = new RelayCommand(() => AddSampleFolder(PrimaryLabel));
        AddNsfwSampleFolderCommand = new RelayCommand(() => AddSampleFolder(SecondaryLabel));
        RemoveSampleFolderCommand = new RelayCommand(parameter => RemoveSampleFolder(parameter as SampleFolderItemViewModel));
        CopyOperationLogCommand = new RelayCommand(parameter => CopyOperationLog(parameter as OperationLogItemViewModel));

        MarkAsSfwCommand = new AsyncRelayCommand(() => ApplyManualCorrectionAsync(PrimaryLabel), () => CanApplyManualCorrection(PrimaryLabel));
        MarkAsNsfwCommand = new AsyncRelayCommand(() => ApplyManualCorrectionAsync(SecondaryLabel), () => CanApplyManualCorrection(SecondaryLabel));
        MarkAsUncertainCommand = new AsyncRelayCommand(() => ApplyManualCorrectionAsync(ImageLabel.Uncertain), () => SelectedResult is not null && !IsBusy);

        ShowAllCommand = new RelayCommand(() => SetFilter(ResultFilter.All));
        ShowSfwCommand = new RelayCommand(() => SetFilter(ResultFilter.Sfw));
        ShowNsfwCommand = new RelayCommand(() => SetFilter(ResultFilter.Nsfw));
        ShowUncertainCommand = new RelayCommand(() => SetFilter(ResultFilter.Uncertain));
        ShowCorrectedCommand = new RelayCommand(() => SetFilter(ResultFilter.Corrected));
        ShowContentSafetyTaskCommand = new RelayCommand(() => SwitchTaskMode(ClassificationTaskMode.ContentSafety), () => !IsBusy);
        ShowPersonTaskCommand = new RelayCommand(() => SwitchTaskMode(ClassificationTaskMode.PersonPresence), () => !IsBusy);
    }

    public RangeObservableCollection<AnalysisItemViewModel> Results { get; }
    public ObservableCollection<WorkspaceFolderItemViewModel> SourceFolders { get; }
    public ObservableCollection<WorkspaceFolderItemViewModel> SfwTargetFolders { get; }
    public ObservableCollection<WorkspaceFolderItemViewModel> NsfwTargetFolders { get; }
    public ObservableCollection<SampleFolderItemViewModel> SfwSampleFolders { get; }
    public ObservableCollection<SampleFolderItemViewModel> NsfwSampleFolders { get; }
    public ObservableCollection<OperationLogItemViewModel> OperationLogs { get; }
    public RelayCommand AddSourceFolderCommand { get; }
    public RelayCommand AddSfwTargetFolderCommand { get; }
    public RelayCommand AddNsfwTargetFolderCommand { get; }
    public RelayCommand ActivateWorkspaceFolderCommand { get; }
    public RelayCommand RemoveWorkspaceFolderCommand { get; }
    public RelayCommand AddSfwSampleFolderCommand { get; }
    public RelayCommand AddNsfwSampleFolderCommand { get; }
    public RelayCommand RemoveSampleFolderCommand { get; }
    public RelayCommand CopyOperationLogCommand { get; }
    public AsyncRelayCommand InitializeCommand { get; }
    public AsyncRelayCommand StartAnalysisCommand { get; }
    public AsyncRelayCommand ConfirmMoveCommand { get; }
    public AsyncRelayCommand RefreshSamplesCommand { get; }
    public AsyncRelayCommand InstallAiModelCommand { get; }
    public AsyncRelayCommand TrainPersonalAiCommand { get; }
    public AsyncRelayCommand ActivateCandidateModelCommand { get; }
    public AsyncRelayCommand ActivatePersonalAiModelCommand { get; }
    public RelayCommand StopScanCommand { get; }
    public AsyncRelayCommand MarkAsSfwCommand { get; }
    public AsyncRelayCommand MarkAsNsfwCommand { get; }
    public AsyncRelayCommand MarkAsUncertainCommand { get; }
    public RelayCommand ShowAllCommand { get; }
    public RelayCommand ShowSfwCommand { get; }
    public RelayCommand ShowNsfwCommand { get; }
    public RelayCommand ShowUncertainCommand { get; }
    public RelayCommand ShowCorrectedCommand { get; }
    public RelayCommand ShowContentSafetyTaskCommand { get; }
    public RelayCommand ShowPersonTaskCommand { get; }

    public ClassificationTaskMode ActiveTaskMode
    {
        get => _activeTaskMode;
        private set
        {
            if (SetProperty(ref _activeTaskMode, value))
            {
                RaiseTaskModePropertiesChanged();
            }
        }
    }

    public bool IsContentSafetyTaskActive => ActiveTaskMode == ClassificationTaskMode.ContentSafety;
    public bool IsPersonTaskActive => ActiveTaskMode == ClassificationTaskMode.PersonPresence;
    public ImageLabel PrimaryLabel => GetPrimaryLabel(ActiveTaskMode);
    public ImageLabel SecondaryLabel => GetSecondaryLabel(ActiveTaskMode);
    public string AppSubtitle => IsPersonTaskActive ? "离线图片人物/非人物分类工具 · 四区工作台" : "离线图片内容安全分类工具 · 四区工作台";
    public string TaskModeTitle => IsPersonTaskActive ? "人物非人物分类" : "内容安全分类";
    public string TaskModeDescription => IsPersonTaskActive
        ? "只判断画面里是否有明确真人或真人元素；二次元、3D、风景、玩偶/雕像都归为非人物。"
        : "判断图片应进入 SFW 还是 NSFW，结合样本库、AI 视觉模型和你的人工纠错。";
    public string TaskConfigDescription => IsPersonTaskActive
        ? "管理人物分类任务的源文件夹、人物/非人物目标与样本库。"
        : "源文件夹、目标文件夹与样本库集中管理。";
    public string SourceFolderTitle => IsPersonTaskActive ? "人物分类源文件夹" : "待分析源文件夹";
    public string PrimaryLabelText => GetLabelDisplayText(PrimaryLabel);
    public string SecondaryLabelText => GetLabelDisplayText(SecondaryLabel);
    public string PrimarySampleCountText => $"{PrimaryLabelText} 样本 {SfwSampleFolders.Count} 个";
    public string SecondarySampleCountText => $"{SecondaryLabelText} 样本 {NsfwSampleFolders.Count} 个";
    public string PrimarySampleTitle => $"{PrimaryLabelText} 样本文件夹";
    public string SecondarySampleTitle => $"{SecondaryLabelText} 样本文件夹";
    public string AddPrimarySampleButtonText => $"添加 {PrimaryLabelText} 样本";
    public string AddSecondarySampleButtonText => $"添加 {SecondaryLabelText} 样本";
    public string PrimaryTargetTitle => $"{PrimaryLabelText} 目标文件夹";
    public string SecondaryTargetTitle => $"{SecondaryLabelText} 目标文件夹";
    public string AddPrimaryTargetButtonText => $"添加 {PrimaryLabelText} 目标";
    public string AddSecondaryTargetButtonText => $"添加 {SecondaryLabelText} 目标";
    public string PrimaryFilterLabel => PrimaryLabelText;
    public string SecondaryFilterLabel => SecondaryLabelText;
    public string StartAnalysisButtonText => "开始分析";
    public string PersonalAiPanelTitle => IsPersonTaskActive ? "人物模型详情" : "个人大模型详情";
    public string PersonalAiModelTitle => IsPersonTaskActive ? "人物模型" : "个人大模型";
    public string ConfirmMoveButtonText => ReadyToMoveCount > 0
        ? $"确认并移动 ({ReadyToMoveCount})"
        : "没有可移动图片";
    public string MarkAsPrimaryButtonText => $"改判为 {PrimaryLabelText}";
    public string MarkAsSecondaryButtonText => $"改判为 {SecondaryLabelText}";
    public string MarkAsUncertainButtonText => "改判为不确定";

    public string SourceFolder
    {
        get => _sourceFolder;
        set
        {
            if (SetProperty(ref _sourceFolder, value))
            {
                RaiseCommandStates();
                if (!_isApplyingTaskSettings)
                {
                    QueueSettingsSave();
                }
            }
        }
    }

    public string SfwTargetFolder
    {
        get => _sfwTargetFolder;
        set
        {
            if (SetProperty(ref _sfwTargetFolder, value))
            {
                RaiseCommandStates();
                if (!_isApplyingTaskSettings)
                {
                    QueueSettingsSave();
                }
            }
        }
    }

    public string NsfwTargetFolder
    {
        get => _nsfwTargetFolder;
        set
        {
            if (SetProperty(ref _nsfwTargetFolder, value))
            {
                RaiseCommandStates();
                if (!_isApplyingTaskSettings)
                {
                    QueueSettingsSave();
                }
            }
        }
    }

    public string StatusTitle
    {
        get => _statusTitle;
        private set => SetProperty(ref _statusTitle, value);
    }

    public string StatusStageText
    {
        get => _statusStageText;
        private set => SetProperty(ref _statusStageText, value);
    }

    public string StatusMessage
    {
        get => _statusMessage;
        private set => SetProperty(ref _statusMessage, value);
    }

    public string StatusCurrentItem
    {
        get => _statusCurrentItem;
        private set => SetProperty(ref _statusCurrentItem, value);
    }

    public string AiModelStatusText
    {
        get => _aiModelStatusText;
        private set => SetProperty(ref _aiModelStatusText, value);
    }

    public string AiModelButtonText
    {
        get => _aiModelButtonText;
        private set => SetProperty(ref _aiModelButtonText, value);
    }

    public string PersonalAiStatusText
    {
        get => _personalAiStatusText;
        private set => SetProperty(ref _personalAiStatusText, value);
    }

    public string PersonalAiButtonText
    {
        get => _personalAiButtonText;
        private set => SetProperty(ref _personalAiButtonText, value);
    }

    public string ActivateCandidateModelButtonText
    {
        get => _activateCandidateModelButtonText;
        private set => SetProperty(ref _activateCandidateModelButtonText, value);
    }

    public string PersonalAiStatusBadgeText
    {
        get => _personalAiStatusBadgeText;
        private set => SetProperty(ref _personalAiStatusBadgeText, value);
    }

    public string PersonalAiStatusBadgeDetailText
    {
        get => _personalAiStatusBadgeDetailText;
        private set => SetProperty(ref _personalAiStatusBadgeDetailText, value);
    }

    public string PersonalAiModePillText
    {
        get => _personalAiModePillText;
        private set => SetProperty(ref _personalAiModePillText, value);
    }

    public string PersonalAiActiveModelEvaluationText
    {
        get => _personalAiActiveModelEvaluationText;
        private set => SetProperty(ref _personalAiActiveModelEvaluationText, value);
    }

    public string PersonalAiCandidateModelVersionText
    {
        get => _personalAiCandidateModelVersionText;
        private set => SetProperty(ref _personalAiCandidateModelVersionText, value);
    }

    public string PersonalAiCandidateModelEvaluationText
    {
        get => _personalAiCandidateModelEvaluationText;
        private set => SetProperty(ref _personalAiCandidateModelEvaluationText, value);
    }

    public IReadOnlyList<PersonalAiModelVersionInfo> PersonalAiModelVersions
    {
        get => _personalAiModelVersions;
        private set => SetProperty(ref _personalAiModelVersions, value);
    }

    public bool HasActivePersonalAiModel
    {
        get => _hasActivePersonalAiModel;
        private set => SetProperty(ref _hasActivePersonalAiModel, value);
    }

    public bool HasCandidatePersonalAiModel
    {
        get => _hasCandidatePersonalAiModel;
        private set => SetProperty(ref _hasCandidatePersonalAiModel, value);
    }

    public bool ShowPersonalAiCurrentModelState
    {
        get => _showPersonalAiCurrentModelState;
        private set => SetProperty(ref _showPersonalAiCurrentModelState, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                RaisePropertyChanged(nameof(IsStatusBusy));
                RaisePropertyChanged(nameof(ProgressPercentText));
                RaiseCommandStates();
            }
        }
    }

    public double ProgressValue
    {
        get => _progressValue;
        private set
        {
            if (SetProperty(ref _progressValue, value))
            {
                RaisePropertyChanged(nameof(ProgressPercentText));
            }
        }
    }

    public AnalysisItemViewModel? SelectedResult
    {
        get => _selectedResult;
        set
        {
            if (SetProperty(ref _selectedResult, value))
            {
                StartPreviewLoading(_selectedResult);
                TrimPreviewCacheAroundSelection();
                RaisePropertyChanged(nameof(SelectedPreviewPositionText));
                RaiseCommandStates();
            }
        }
    }

    public int TotalCount => Results.Count;
    public int SfwCount => Results.Count(item => item.CurrentLabel == PrimaryLabel);
    public int NsfwCount => Results.Count(item => item.CurrentLabel == SecondaryLabel);
    public int UncertainCount => Results.Count(item => item.CurrentLabel == ImageLabel.Uncertain);
    public int CorrectedCount => Results.Count(item => item.IsReviewedInCurrentBatch);
    public int PendingConfirmationCount => Results.Count(item => item.HasPendingBatchConfirmation);
    public int ReadyToMoveCount => Results.Count(item => item.CurrentLabel == PrimaryLabel || item.CurrentLabel == SecondaryLabel);
    public bool IsAllFilterActive => _activeFilter == ResultFilter.All;
    public bool IsSfwFilterActive => _activeFilter == ResultFilter.Sfw;
    public bool IsNsfwFilterActive => _activeFilter == ResultFilter.Nsfw;
    public bool IsUncertainFilterActive => _activeFilter == ResultFilter.Uncertain;
    public bool IsCorrectedFilterActive => _activeFilter == ResultFilter.Corrected;
    public string ProgressPercentText => _statusTotalCount > 0 ? $"{Math.Round(ProgressValue * 100d):0}%" : (IsBusy ? "处理中" : "待命");
    public string ProgressCountText => _statusTotalCount > 0 ? $"{_statusCurrentCount} / {_statusTotalCount}" : "等待任务";
    public bool HasProgressNumbers => _statusTotalCount > 0;
    public bool IsStatusBusy => IsBusy;
    public string ResultHeadline => Results.Count == 0 ? "分析结果列表" : $"分析结果列表 · {Results.Count} 张";
    public string ResultSubheadline => Results.Count == 0 ? "开始分析后，结果会按你的个人标准逐步生成。" : "单击列表项即可预览和改判，清空不确定后再确认移动。";
    public string SelectedPreviewPositionText => GetSelectedPreviewPositionText();

    public bool IsToastVisible
    {
        get => _isToastVisible;
        private set => SetProperty(ref _isToastVisible, value);
    }

    public string ToastMessage
    {
        get => _toastMessage;
        private set => SetProperty(ref _toastMessage, value);
    }

    public OperationLogLevel ToastLevel
    {
        get => _toastLevel;
        private set => SetProperty(ref _toastLevel, value);
    }

    public string SampleLibrarySummary
    {
        get
        {
            var enabledCount = SfwSampleFolders.Concat(NsfwSampleFolders).Count(item => item.IsEnabled);
            var totalCount = SfwSampleFolders.Count + NsfwSampleFolders.Count;
            return totalCount == 0
                ? $"还没有样本库，建议先导入你已经分好类的 {PrimaryLabelText} / {SecondaryLabelText} 文件夹。"
                : $"当前共挂载 {totalCount} 个样本文件夹，其中 {enabledCount} 个正在参与学习。";
        }
    }

    public bool SelectAdjacentVisibleResult(int direction)
    {
        var visibleResults = _resultsView.Cast<AnalysisItemViewModel>().ToList();
        if (visibleResults.Count == 0)
        {
            return false;
        }

        var currentIndex = SelectedResult is null ? -1 : visibleResults.IndexOf(SelectedResult);
        var nextIndex = currentIndex < 0
            ? 0
            : Math.Clamp(currentIndex + Math.Sign(direction), 0, visibleResults.Count - 1);

        var nextResult = visibleResults[nextIndex];
        if (ReferenceEquals(nextResult, SelectedResult))
        {
            return false;
        }

        SelectedResult = nextResult;
        return true;
    }

    public void Dispose()
    {
        _settingsSaveDebounceTokenSource?.Cancel();
        _settingsSaveDebounceTokenSource?.Dispose();
        _activeScanTokenSource?.Cancel();
        _activeScanTokenSource?.Dispose();
        _toastTokenSource?.Cancel();
        _toastTokenSource?.Dispose();
        _previewLoadTokenSource?.Cancel();
        _previewLoadTokenSource?.Dispose();
        _lifetimeTokenSource.Cancel();
        _lifetimeTokenSource.Dispose();
    }

    private async Task InitializeAsync()
    {
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "正在加载配置", "正在读取本地缓存和上一次的工作区。", showToast: false);
        SetStatusSnapshot("正在加载", "初始化", "正在读取本地缓存和上一次的工作区配置。", "准备恢复上次使用的样本库和目标路径。", 0d, 0, 0);

        try
        {
            var settings = await _workspaceService.InitializeAsync();
            _settings = settings;
            ActiveTaskMode = settings.TaskMode;
            _aiModelSettings = settings.AiModel ?? new AiModelSettings();

            LoadActiveTaskSettings();

            SetStatusSnapshot("准备就绪", "空闲", "缓存已经就绪，可以开始分析。", "你可以继续补充样本库，或者直接对源文件夹生成判断列表。", 0d, 0, 0);
            RefreshAiModelStatus(settings);
            RefreshPersonalAiStatus(settings);
            RaiseSampleFolderSummaryProperties();
            AddOperationLog(OperationLogLevel.Success, "软件已就绪", "配置和缓存加载完成。");
        }
        catch (Exception exception)
        {
            SetStatusSnapshot("加载失败", "需要处理", "读取本地配置时遇到问题，软件已保留窗口，方便继续排查。", exception.Message, 0d, 0, 0);
            AddOperationLog(OperationLogLevel.Error, "加载配置失败", exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task StartAnalysisAsync()
    {
        var scanTokenSource = BeginCancellableScan();
        var cancellationToken = scanTokenSource.Token;
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "开始分析", "正在刷新样本库并分析源文件夹。");
        SetStatusSnapshot("正在准备分析", "启动", "正在为本轮分析准备样本数据。", "这一步会先检查样本缓存和新增图片。", 0.03d, 0, 0);

        try
        {
            await SaveSettingsAsync();
            var settings = BuildSettings();
            var progress = new Progress<WorkspaceProgressInfo>(HandleWorkspaceProgress);
            var summary = await _workspaceService.RefreshSampleLibraryAsync(settings, cancellationToken, progress);
            ApplyPersonalAiSettings(GetActivePersonalAiSettings(settings));

            SetStatusSnapshot("样本库已同步", "样本扫描完成", $"共检查 {summary.ScannedFiles} 张样本，复用缓存 {summary.ReusedCacheFiles} 张，新增导入 {summary.ImportedFiles} 张。", "接下来开始分析源文件夹中的图片。", 0.48d, summary.ScannedFiles, summary.ScannedFiles);
            AddOperationLog(OperationLogLevel.Success, "样本库已同步", $"检查 {summary.ScannedFiles} 张，新增 {summary.ImportedFiles} 张。", showToast: false);

            var records = await _workspaceService.AnalyzeAsync(settings, cancellationToken, progress);
            SetStatusSnapshot("正在装载结果", "结果列表", "扫描判断已完成，正在批量生成界面列表。", "大量图片会一次性刷新，界面会更顺。", 0.96d, records.Count, records.Count);
            var resultItems = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return records.Select(record => new AnalysisItemViewModel(record)).ToList();
            }, cancellationToken);

            ReleaseAllResultPreviews();
            Results.ReplaceAll(resultItems);
            SelectFirstVisibleResult();
            RefreshSummaries();

            SetStatusSnapshot("分析完成", "等待你检查", $"本轮共生成 {Results.Count} 条结果，你可以直接筛查、改判，清空不确定后确认移动。", "人工纠正会直接写回缓存，后续相似图片会优先参考这些经验。", 1d, Results.Count, Results.Count);
            AddOperationLog(OperationLogLevel.Success, "分析完成", BuildAnalysisCompletionDetail(records));
        }
        catch (OperationCanceledException)
        {
            SetStatusSnapshot("任务已取消", "已停止", "当前分析任务已经停止。", "可以重新开始一次新的分析。", ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Warning, "分析已停止", "当前分析任务已取消。");
        }
        catch (Exception exception)
        {
            SetStatusSnapshot("分析失败", "需要处理", "分析过程中遇到错误，已经停止本轮任务。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "分析失败", exception.Message);
        }
        finally
        {
            EndCancellableScan(scanTokenSource);
            IsBusy = false;
        }
    }

    private async Task RefreshSamplesAsync()
    {
        var scanTokenSource = BeginCancellableScan();
        var cancellationToken = scanTokenSource.Token;
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "开始预扫描样本", "正在刷新样本缓存。");
        SetStatusSnapshot("正在预扫描", "样本扫描", "正在刷新样本缓存。", "只会复用旧缓存，并补入新增的样本图片。", 0.03d, 0, 0);

        try
        {
            await SaveSettingsAsync();
            var settings = BuildSettings();
            var summary = await _workspaceService.RefreshSampleLibraryAsync(settings, cancellationToken, new Progress<WorkspaceProgressInfo>(HandleWorkspaceProgress));
            ApplyPersonalAiSettings(GetActivePersonalAiSettings(settings));
            SetStatusSnapshot("样本库已刷新", "完成", $"共检查 {summary.ScannedFiles} 张样本，命中缓存 {summary.ReusedCacheFiles} 张，新增导入 {summary.ImportedFiles} 张。", "现在可以直接开始分析源文件夹。", 1d, summary.ScannedFiles, summary.ScannedFiles);
            AddOperationLog(OperationLogLevel.Success, "样本库已刷新", $"检查 {summary.ScannedFiles} 张，新增 {summary.ImportedFiles} 张。");
        }
        catch (OperationCanceledException)
        {
            SetStatusSnapshot("任务已取消", "已停止", "样本扫描已经停止。", "可以重新刷新样本库，或者直接开始新的分析。", ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Warning, "样本扫描已停止", "当前样本扫描任务已取消。");
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            SetStatusSnapshot("刷新失败", "需要处理", "刷新样本库时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "刷新样本库失败", exception.Message);
        }
        finally
        {
            EndCancellableScan(scanTokenSource);
            IsBusy = false;
        }
    }

    private async Task InstallAiModelAsync()
    {
        var settings = BuildSettings();
        var aiModelStatus = _workspaceService.GetAiModelStatus(settings);
        if (aiModelStatus.IsInstalled)
        {
            RefreshAiModelStatus(settings);
            SetStatusSnapshot("AI 模型已就绪", "无需下载", "本地 AI 模型包已经存在，本次不会重复下载。", aiModelStatus.ModelPath, 1d, 100, 100);
            AddOperationLog(OperationLogLevel.Info, "AI 模型包已安装", "本次不会重复下载。");
            return;
        }

        var scanTokenSource = BeginCancellableScan();
        var cancellationToken = scanTokenSource.Token;
        _isAiModelDownloading = true;
        AiModelButtonText = "正在下载 AI 模型包";
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "开始下载 AI 模型包", "下载完成后会自动参与扫描判断。");
        SetStatusSnapshot("正在准备 AI 模型包", "模型下载", "正在下载通用 NSFW 分类模型和 NudeNet 裸露检测模型。", "模型较大，下载完成后会自动参与扫描判断。", 0.02d, 0, 100);

        try
        {
            await SaveSettingsAsync();
            await _workspaceService.DownloadAiModelAsync(settings, cancellationToken, new Progress<WorkspaceProgressInfo>(HandleWorkspaceProgress));
            RefreshAiModelStatus(settings);
            SetStatusSnapshot("AI 模型已就绪", "完成", "本地 AI 模型包已经安装完成。", "下一次扫描会结合通用分类、NudeNet 裸露检测、样本库和你的人工纠错。", 1d, 100, 100);
            AddOperationLog(OperationLogLevel.Success, "AI 模型包安装完成", "下一次扫描会结合本地视觉判断和个人学习。");
        }
        catch (OperationCanceledException)
        {
            RefreshAiModelStatus(settings);
            SetStatusSnapshot("任务已取消", "已停止", "AI 模型包下载已经停止。", "可以稍后重新下载。", ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Warning, "AI 模型包下载已停止", "可以稍后重新下载。");
        }
        catch (Exception exception)
        {
            RefreshAiModelStatus(settings);
            SetStatusSnapshot("模型下载失败", "需要处理", "下载 AI 模型包时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "AI 模型包下载失败", exception.Message);
        }
        finally
        {
            EndCancellableScan(scanTokenSource);
            _isAiModelDownloading = false;
            RefreshAiModelStatus(settings);
            IsBusy = false;
        }
    }

    private async Task TrainPersonalAiAsync()
    {
        var scanTokenSource = BeginCancellableScan();
        var cancellationToken = scanTokenSource.Token;
        var settings = BuildSettings();
        var trainingStatus = _workspaceService.GetPersonalAiTrainingStatus(settings);
        var modelText = IsPersonTaskActive ? "人物模型" : "个人大模型";
        var loraText = IsPersonTaskActive ? "人物 LoRA" : "LoRA 个人模型";
        var deviceText = string.IsNullOrWhiteSpace(trainingStatus.TrainingDeviceDisplay)
            ? "当前设备待检测"
            : trainingStatus.TrainingDeviceDisplay;
        IsBusy = true;
        _isPersonalAiTraining = true;
        _lastTrainingProgressLogCount = -1;
        _lastTrainingProgressLogSignature = string.Empty;
        PersonalAiButtonText = $"正在训练{modelText}";
        AddOperationLog(
            OperationLogLevel.Info,
            trainingStatus.HasResumeCheckpoint ? $"开始断点续训{modelText}" : $"开始训练{modelText}",
            trainingStatus.HasResumeCheckpoint
                ? $"将从第 {trainingStatus.ResumeEpoch} 轮、第 {trainingStatus.ResumeBatch} 批附近继续，训练设备：{deviceText}。"
                : $"正在基于上一版模型继续训练 {loraText}，并混入新增样本与历史回放样本，训练设备：{deviceText}。");
        SetStatusSnapshot(
            trainingStatus.HasResumeCheckpoint ? $"正在断点续训{modelText}" : $"正在训练{modelText}",
            "个人 AI",
            trainingStatus.HasResumeCheckpoint
                ? $"正在从最近一次 checkpoint 继续训练 {loraText}，当前设备：{deviceText}。"
                : $"正在基于上一版模型继续训练 {loraText}。当前设备：{deviceText}。首次训练会准备基础视觉模型。",
            "训练期间软件会保持响应，训练完成后会先生成候选模型，等待你确认启用。",
            0.05d,
            0,
            1);

        try
        {
            await SaveSettingsAsync();
            var status = await _workspaceService.TrainPersonalAiAsync(settings, cancellationToken, new Progress<WorkspaceProgressInfo>(HandleWorkspaceProgress));
            RefreshPersonalAiStatus(settings);
            var versionText = string.IsNullOrWhiteSpace(status.CandidateModelVersion) ? status.ActiveModelVersion : status.CandidateModelVersion;
            SetStatusSnapshot($"{modelText}训练完成", "完成", status.Message, versionText, 1d, 1, 1);
            AddOperationLog(OperationLogLevel.Success, $"{modelText}训练完成", string.IsNullOrWhiteSpace(status.CandidateModelVersion) ? status.Message : $"候选模型 {status.CandidateModelVersion} 已生成，确认体验后再启用。");
        }
        catch (OperationCanceledException)
        {
            RefreshPersonalAiStatus(settings);
            SetStatusSnapshot("训练已停止", "已停止", $"{modelText}训练已经取消。", "可以稍后继续训练。", ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Warning, $"{modelText}训练已停止", "可以稍后继续训练。");
        }
        catch (Exception exception)
        {
            RefreshPersonalAiStatus(settings);
            SetStatusSnapshot("训练失败", "需要处理", $"{modelText}训练时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, $"{modelText}训练失败", exception.Message);
        }
        finally
        {
            EndCancellableScan(scanTokenSource);
            _isPersonalAiTraining = false;
            IsBusy = false;
            RefreshPersonalAiStatus(settings);
        }
    }

    private async Task ActivateCandidateModelAsync()
    {
        var settings = BuildSettings();
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "正在启用候选模型", "正在把最新训练出的候选模型切换为当前启用版本。");

        try
        {
            var status = await _workspaceService.ActivateCandidatePersonalAiModelAsync(settings, _lifetimeTokenSource.Token);
            RefreshPersonalAiStatus(settings);
            SetStatusSnapshot("候选模型已启用", "完成", status.Message, status.ActiveModelVersion, 1d, 1, 1);
            AddOperationLog(OperationLogLevel.Success, "候选模型已启用", $"当前启用版本：{status.ActiveModelVersion}");
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            SetStatusSnapshot("启用候选模型失败", "需要处理", "切换候选模型时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "启用候选模型失败", exception.Message);
        }
        finally
        {
            IsBusy = false;
            RefreshPersonalAiStatus(settings);
        }
    }

    private async Task ActivatePersonalAiModelAsync(object? parameter)
    {
        if (parameter is not PersonalAiModelVersionInfo model)
        {
            return;
        }

        var settings = BuildSettings();
        IsBusy = true;
        AddOperationLog(OperationLogLevel.Info, "正在切换模型", $"正在切换到 {model.Version}。", showToast: false);

        try
        {
            var status = await _workspaceService.ActivatePersonalAiModelAsync(
                settings,
                model.Version,
                _lifetimeTokenSource.Token);
            RefreshPersonalAiStatus(settings);
            SetStatusSnapshot("模型已切换", "完成", $"当前使用：{status.ActiveModelVersion}", "后续分析将使用这个版本。", 1d, 1, 1);
            AddOperationLog(OperationLogLevel.Success, "模型已切换", $"当前使用版本：{status.ActiveModelVersion}");
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            SetStatusSnapshot("模型切换失败", "需要处理", "切换个人模型时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "模型切换失败", exception.Message);
        }
        finally
        {
            IsBusy = false;
            RefreshPersonalAiStatus(settings);
        }
    }

    private bool CanActivatePersonalAiModel(object? parameter)
    {
        return !IsBusy && parameter is PersonalAiModelVersionInfo { CanActivate: true };
    }

    private async Task ConfirmMoveAsync()
    {
        var movableItems = GetMovableResults();
        if (movableItems.Count == 0)
        {
            SetStatusSnapshot("无需移动", "已检查", "当前没有可移动的已分类图片。", "请先完成分析，或先把图片改判到明确分类。", 1d, 0, 0);
            AddOperationLog(OperationLogLevel.Info, "无需移动", "当前没有可移动的已分类图片。");
            return;
        }

        if (UncertainCount > 0)
        {
            const string message = "请先处理完不确定栏中的图片，清空不确定后才能开始移动。";
            SetStatusSnapshot("暂不能移动", "等待处理", message, $"当前还有 {UncertainCount} 张不确定图片。", ProgressValue, 0, movableItems.Count);
            AddOperationLog(OperationLogLevel.Warning, "暂不能移动", message);
            return;
        }

        var missingTargetLabel = GetMissingMoveTargetLabel(movableItems);
        if (!string.IsNullOrWhiteSpace(missingTargetLabel))
        {
            var detail = $"请先设置启用的 {missingTargetLabel} 目标文件夹，再执行移动。";
            SetStatusSnapshot("缺少目标文件夹", "需要处理", detail, "目标目录未配置完整。", ProgressValue, 0, movableItems.Count);
            AddOperationLog(OperationLogLevel.Warning, "缺少目标文件夹", detail);
            return;
        }

        IsBusy = true;
        _lastMoveProgressLogCount = 0;
        AddOperationLog(OperationLogLevel.Info, "开始移动文件", "正在把当前已分类图片移动到对应目标目录。");
        SetStatusSnapshot("正在移动文件", "移动文件", "正在把当前已分类图片移动到对应目标目录。", "遇到同名文件时会自动改名后继续移动。", 0.1d, 0, movableItems.Count);

        try
        {
            await SaveSettingsAsync();
            var moveSummary = await _workspaceService.MoveConfirmedAsync(
                movableItems.Select(item => item.ToRecord()).ToList(),
                BuildSettings(),
                _lifetimeTokenSource.Token,
                new Progress<WorkspaceProgressInfo>(HandleMoveWorkspaceProgress));

            var movedPaths = moveSummary.Results
                .Where(item => item.Status is FileMoveStatus.Moved or FileMoveStatus.RenamedDueToConflict)
                .Select(item => item.SourcePath)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            var renamedResults = moveSummary.Results
                .Where(item => item.Status == FileMoveStatus.RenamedDueToConflict)
                .ToList();
            var missingPaths = moveSummary.Results
                .Where(item => item.Status == FileMoveStatus.SourceMissing)
                .Select(item => item.SourcePath)
                .ToList();

            foreach (var result in Results.Where(item => movedPaths.Contains(item.FilePath)).ToList())
            {
                Results.Remove(result);
            }

            RefreshSummaries();
            _resultsView.Refresh();

            if (SelectedResult is null || !Results.Contains(SelectedResult))
            {
                SelectFirstVisibleResult();
            }

            var movedCount = movedPaths.Count;
            var renamedCount = renamedResults.Count;
            var missingCount = missingPaths.Count;
            var detail = BuildMoveSummaryDetail(movedCount, renamedResults, missingPaths);
            var level = missingCount > 0 ? OperationLogLevel.Warning : OperationLogLevel.Success;
            var title = movedCount > 0 ? "移动完成" : "没有文件被移动";
            var stageText = missingCount > 0 ? "部分完成" : "完成";
            var summaryText = BuildMoveSummaryText(movedCount, renamedCount, missingCount);

            SetStatusSnapshot(title, stageText, summaryText, detail, 1d, movedCount, movableItems.Count);
            AddOperationLog(level, title, detail);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            SetStatusSnapshot("移动失败", "需要处理", "移动当前图片时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "移动失败", exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task ApplyManualCorrectionAsync(ImageLabel targetLabel)
    {
        var correctedResult = SelectedResult;
        if (correctedResult is null)
        {
            return;
        }

        var previousVisibleIndex = GetVisibleResultIndex(correctedResult);
        if (correctedResult.CurrentLabel == targetLabel)
        {
            var stageForBatchConfirmation = targetLabel == PrimaryLabel || targetLabel == SecondaryLabel;
            correctedResult.MarkAsConfirmed(stageForBatchConfirmation);
            RefreshSummaries();
            _resultsView.Refresh();

            var nextResult = SelectNextResultAfterCorrection(correctedResult, previousVisibleIndex);
            var isUncertainReview = targetLabel == ImageLabel.Uncertain;
            SetStatusSnapshot(
                "已记录人工确认",
                nextResult is not null && !ReferenceEquals(nextResult, correctedResult) ? "已切到下一张" : "已更新",
                isUncertainReview
                    ? "当前图片仍保留在不确定队列，这次操作只记为人工确认，暂不会写入训练样本。"
                    : "当前图片标签没有变化，这次操作已记为人工确认。清空不确定队列后，就可以直接确认移动这批图片。",
                nextResult?.FileName ?? correctedResult.FileName,
                1d,
                1,
                1);
            AddOperationLog(OperationLogLevel.Success, "已记录人工确认", correctedResult.FileName);
            return;
        }

        IsBusy = true;
        var isPersistentLabel = targetLabel == PrimaryLabel || targetLabel == SecondaryLabel;
        SetStatusSnapshot(
            "正在写入纠正",
            "人工修正",
            isPersistentLabel ? "正在把这次改判结果写入缓存。" : "正在把这次改判结果应用到当前复核列表。",
            correctedResult.FileName,
            0.2d,
            1,
            1);

        try
        {
            if (isPersistentLabel)
            {
                await _workspaceService.ApplyManualCorrectionAsync(correctedResult.ToRecord(), targetLabel, _lifetimeTokenSource.Token);
                _manualCorrectionsSinceTrainingPrompt++;
            }
            correctedResult.ApplyManualCorrection(targetLabel);
            RefreshSummaries();
            _resultsView.Refresh();
            MaybeSuggestTraining();

            var nextResult = SelectNextResultAfterCorrection(correctedResult, previousVisibleIndex);
            var hasAdvanced = nextResult is not null && !ReferenceEquals(nextResult, correctedResult);
            SetStatusSnapshot(
                "纠正已写入",
                hasAdvanced ? "已切到下一张" : "已更新",
                hasAdvanced
                    ? isPersistentLabel
                        ? "这次改判已写回缓存，已经自动补上下一张待复查图片。"
                        : "这次改判已切到不确定队列，已经自动补上下一张待复查图片。"
                    : isPersistentLabel
                        ? "这次改判已写回缓存，当前列表已经没有下一张了。"
                        : "这次改判已应用到当前列表，当前列表已经没有下一张了。",
                nextResult?.FileName ?? correctedResult.FileName,
                1d,
                1,
                1);
            AddOperationLog(OperationLogLevel.Success, $"已改判为 {correctedResult.CurrentLabelText}", correctedResult.FileName);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            SetStatusSnapshot("纠正失败", "需要处理", "写入人工纠正时遇到错误。", exception.Message, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Error, "纠正失败", exception.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private List<AnalysisItemViewModel> GetMovableResults()
    {
        return Results
            .Where(item => item.CurrentLabel == PrimaryLabel || item.CurrentLabel == SecondaryLabel)
            .ToList();
    }

    private string? GetMissingMoveTargetLabel(IReadOnlyCollection<AnalysisItemViewModel> movableItems)
    {
        var primaryTargetFolder = ResolveActiveWorkspaceFolder(SfwTargetFolders, SfwTargetFolder);
        if (movableItems.Any(item => item.CurrentLabel == PrimaryLabel) &&
            string.IsNullOrWhiteSpace(primaryTargetFolder))
        {
            return PrimaryLabelText;
        }

        var secondaryTargetFolder = ResolveActiveWorkspaceFolder(NsfwTargetFolders, NsfwTargetFolder);
        if (movableItems.Any(item => item.CurrentLabel == SecondaryLabel) &&
            string.IsNullOrWhiteSpace(secondaryTargetFolder))
        {
            return SecondaryLabelText;
        }

        return null;
    }

    private static string BuildMoveSummaryText(int movedCount, int renamedCount, int missingCount)
    {
        var parts = new List<string>();
        if (movedCount > 0)
        {
            parts.Add($"已移动 {movedCount} 张");
        }

        if (renamedCount > 0)
        {
            parts.Add($"重名自动改名 {renamedCount} 张");
        }

        if (missingCount > 0)
        {
            parts.Add($"源文件不存在跳过 {missingCount} 张");
        }

        return parts.Count > 0
            ? string.Join("，", parts) + "。"
            : "本次没有文件被移动。";
    }

    private static string BuildMoveSummaryDetail(int movedCount, IReadOnlyList<FileMoveResult> renamedResults, IReadOnlyList<string> missingPaths)
    {
        var detailParts = new List<string>();
        if (movedCount > 0)
        {
            detailParts.Add($"成功移动 {movedCount} 张图片。");
        }

        if (renamedResults.Count > 0)
        {
            var renamedNames = string.Join(
                "、",
                renamedResults
                    .Select(item => $"{Path.GetFileName(item.SourcePath)} -> {Path.GetFileName(item.DestinationPath)}")
                    .Where(name => !string.IsNullOrWhiteSpace(name))
                    .Take(5));
            detailParts.Add(string.IsNullOrWhiteSpace(renamedNames)
                ? $"有 {renamedResults.Count} 张图片因重名被自动改名。"
                : $"重名自动改名 {renamedResults.Count} 张：{renamedNames}{(renamedResults.Count > 5 ? " 等" : string.Empty)}。");
        }

        if (missingPaths.Count > 0)
        {
            var missingNames = string.Join("、", missingPaths.Select(Path.GetFileName).Where(name => !string.IsNullOrWhiteSpace(name)).Take(5));
            detailParts.Add(string.IsNullOrWhiteSpace(missingNames)
                ? $"有 {missingPaths.Count} 张图片在移动前已不存在。"
                : $"移动前已不存在 {missingPaths.Count} 张：{missingNames}{(missingPaths.Count > 5 ? " 等" : string.Empty)}。");
        }

        return detailParts.Count > 0
            ? string.Join(" ", detailParts)
            : "本次没有文件被移动。";
    }

    private int GetVisibleResultIndex(AnalysisItemViewModel result)
    {
        var index = 0;
        foreach (AnalysisItemViewModel visibleResult in _resultsView)
        {
            if (ReferenceEquals(visibleResult, result))
            {
                return index;
            }

            index++;
        }

        return -1;
    }

    private void MaybeSuggestTraining()
    {
        var triggerCount = Math.Max(1, _personalAiSettings.TriggerCorrectionCount <= 0 ? 50 : _personalAiSettings.TriggerCorrectionCount);
        if (_manualCorrectionsSinceTrainingPrompt < triggerCount)
        {
            return;
        }

        _manualCorrectionsSinceTrainingPrompt = 0;
        var primaryCorrections = Results.Count(item => item.IsManualCorrection && item.CurrentLabel == PrimaryLabel);
        var secondaryCorrections = Results.Count(item => item.IsManualCorrection && item.CurrentLabel == SecondaryLabel);
        var modelText = IsPersonTaskActive ? "人物模型" : "个人大模型";
        AddOperationLog(
            OperationLogLevel.Info,
            "已累计足够改判样本",
            $"已累计新增 {triggerCount} 张改判，可开始训练{modelText}。本轮改判中 {PrimaryLabelText} {primaryCorrections} 张，{SecondaryLabelText} {secondaryCorrections} 张。");
    }

    private AnalysisItemViewModel? SelectNextResultAfterCorrection(AnalysisItemViewModel correctedResult, int previousVisibleIndex)
    {
        var visibleResults = _resultsView.Cast<AnalysisItemViewModel>().ToList();
        if (visibleResults.Count == 0)
        {
            SelectedResult = null;
            return null;
        }

        var currentIndex = visibleResults.IndexOf(correctedResult);
        var nextIndex = currentIndex >= 0
            ? currentIndex + 1
            : Math.Max(previousVisibleIndex, 0);

        if (nextIndex >= visibleResults.Count)
        {
            nextIndex = currentIndex >= 0 ? currentIndex : visibleResults.Count - 1;
        }

        SelectedResult = visibleResults[nextIndex];
        return SelectedResult;
    }

    private void SwitchTaskMode(ClassificationTaskMode taskMode)
    {
        if (ActiveTaskMode == taskMode)
        {
            return;
        }

        if (IsBusy)
        {
            AddOperationLog(OperationLogLevel.Warning, "暂不能切换任务", "当前任务正在运行，请停止或等待完成后再切换。");
            return;
        }

        UpdateCurrentTaskSettings();
        ActiveTaskMode = taskMode;
        _settings.TaskMode = taskMode;
        LoadActiveTaskSettings();

        ReleaseAllResultPreviews();
        Results.Clear();
        _resultsView.Refresh();
        SetFilter(ResultFilter.All);
        RefreshSummaries();
        RefreshPersonalAiStatus(_settings);

        SetStatusSnapshot("任务类型已切换", TaskModeTitle, TaskModeDescription, SourceFolderTitle, 0d, 0, 0);
        AddOperationLog(OperationLogLevel.Info, "任务类型已切换", TaskModeTitle);
        _ = SaveSettingsAsync();
    }

    private void LoadActiveTaskSettings()
    {
        _isApplyingTaskSettings = true;
        try
        {
            ClearTaskCollections();

            if (ActiveTaskMode == ClassificationTaskMode.PersonPresence)
            {
                var personTask = _settings.PersonTask ?? new PersonTaskSettings();
                _settings.PersonTask = personTask;
                LoadWorkspaceFolders(personTask.SourceFolders, WorkspaceFolderKind.Source);
                LoadWorkspaceFolders(personTask.PersonTargetFolders, WorkspaceFolderKind.SfwTarget);
                LoadWorkspaceFolders(personTask.NonPersonTargetFolders, WorkspaceFolderKind.NsfwTarget);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.Source);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.SfwTarget);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.NsfwTarget);
                LoadSampleFolders(personTask.SampleFolders);
                _personalAiSettings = personTask.PersonalAi ?? new PersonalAiModelSettings();
            }
            else
            {
                LoadWorkspaceFolders(_settings.SourceFolders, WorkspaceFolderKind.Source);
                LoadWorkspaceFolders(_settings.SfwTargetFolders, WorkspaceFolderKind.SfwTarget);
                LoadWorkspaceFolders(_settings.NsfwTargetFolders, WorkspaceFolderKind.NsfwTarget);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.Source);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.SfwTarget);
                SyncActiveWorkspaceFolderPath(WorkspaceFolderKind.NsfwTarget);
                LoadSampleFolders(_settings.SampleFolders);
                _personalAiSettings = _settings.PersonalAi ?? new PersonalAiModelSettings();
            }
        }
        finally
        {
            _isApplyingTaskSettings = false;
        }

        RaiseTaskModePropertiesChanged();
    }

    private void UpdateCurrentTaskSettings()
    {
        var sourceFolder = ResolveActiveWorkspaceFolder(SourceFolders, SourceFolder);
        var primaryTargetFolder = ResolveActiveWorkspaceFolder(SfwTargetFolders, SfwTargetFolder);
        var secondaryTargetFolder = ResolveActiveWorkspaceFolder(NsfwTargetFolders, NsfwTargetFolder);
        var sourceFolders = BuildWorkspaceFolderConfigs(SourceFolders, sourceFolder);
        var primaryTargetFolders = BuildWorkspaceFolderConfigs(SfwTargetFolders, primaryTargetFolder);
        var secondaryTargetFolders = BuildWorkspaceFolderConfigs(NsfwTargetFolders, secondaryTargetFolder);
        var sampleFolders = BuildSampleFolderConfigs();

        if (ActiveTaskMode == ClassificationTaskMode.PersonPresence)
        {
            var personTask = _settings.PersonTask ?? new PersonTaskSettings();
            _settings.PersonTask = personTask;
            personTask.SourceFolder = sourceFolder;
            personTask.PersonTargetFolder = primaryTargetFolder;
            personTask.NonPersonTargetFolder = secondaryTargetFolder;
            personTask.SourceFolders = sourceFolders;
            personTask.PersonTargetFolders = primaryTargetFolders;
            personTask.NonPersonTargetFolders = secondaryTargetFolders;
            personTask.SampleFolders = sampleFolders;
            personTask.PersonalAi = BuildPersonalAiSettings(ClassificationTaskMode.PersonPresence);
        }
        else
        {
            _settings.SourceFolder = sourceFolder;
            _settings.SfwTargetFolder = primaryTargetFolder;
            _settings.NsfwTargetFolder = secondaryTargetFolder;
            _settings.SourceFolders = sourceFolders;
            _settings.SfwTargetFolders = primaryTargetFolders;
            _settings.NsfwTargetFolders = secondaryTargetFolders;
            _settings.SampleFolders = sampleFolders;
            _settings.PersonalAi = BuildPersonalAiSettings(ClassificationTaskMode.ContentSafety);
        }
    }

    private void ClearTaskCollections()
    {
        foreach (var sampleFolder in SfwSampleFolders.Concat(NsfwSampleFolders).ToList())
        {
            sampleFolder.PropertyChanged -= OnSampleFolderPropertyChanged;
        }

        SourceFolders.Clear();
        SfwTargetFolders.Clear();
        NsfwTargetFolders.Clear();
        SfwSampleFolders.Clear();
        NsfwSampleFolders.Clear();
    }

    private void LoadSampleFolders(IEnumerable<SampleFolderConfig> sampleFolders)
    {
        foreach (var sampleFolder in sampleFolders)
        {
            if (string.IsNullOrWhiteSpace(sampleFolder.FolderPath) || !IsCurrentTaskLabel(sampleFolder.Label))
            {
                continue;
            }

            AttachSampleFolder(new SampleFolderItemViewModel
            {
                Id = string.IsNullOrWhiteSpace(sampleFolder.Id) ? Guid.NewGuid().ToString("N") : sampleFolder.Id,
                FolderPath = sampleFolder.FolderPath,
                Label = sampleFolder.Label,
                IsEnabled = sampleFolder.IsEnabled,
            });
        }
    }

    private void AddWorkspaceFolder(WorkspaceFolderKind kind)
    {
        var selectedPath = PickFolder(GetWorkspaceFolderPickerTitle(kind));
        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            return;
        }

        var folders = GetWorkspaceFolders(kind);
        var existingFolder = folders.FirstOrDefault(item => string.Equals(item.FolderPath, selectedPath, StringComparison.OrdinalIgnoreCase));
        if (existingFolder is not null)
        {
            ActivateWorkspaceFolder(existingFolder);
            SetStatusSnapshot("工作文件夹已切换", GetWorkspaceFolderKindText(kind), "这个文件夹已经在列表中，已直接切换为当前启用。", existingFolder.FolderName, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Info, "工作文件夹已切换", existingFolder.FolderPath);
            return;
        }

        var workspaceFolder = new WorkspaceFolderItemViewModel
        {
            Id = Guid.NewGuid().ToString("N"),
            FolderPath = selectedPath,
            Kind = kind,
            IsActive = false,
        };

        folders.Add(workspaceFolder);
        ActivateWorkspaceFolder(workspaceFolder);
        SetStatusSnapshot("工作文件夹已加入", GetWorkspaceFolderKindText(kind), "已加入列表，并切换为当前启用。", workspaceFolder.FolderName, ProgressValue, _statusCurrentCount, _statusTotalCount);
        AddOperationLog(OperationLogLevel.Success, "工作文件夹已加入", workspaceFolder.FolderPath);
    }

    private void ActivateWorkspaceFolder(WorkspaceFolderItemViewModel? workspaceFolder)
    {
        if (workspaceFolder is null)
        {
            return;
        }

        var folders = GetWorkspaceFolders(workspaceFolder.Kind);
        foreach (var folder in folders)
        {
            folder.IsActive = ReferenceEquals(folder, workspaceFolder);
        }

        SyncActiveWorkspaceFolderPath(workspaceFolder.Kind);
        _ = SaveSettingsAsync();
        AddOperationLog(OperationLogLevel.Info, "工作文件夹已启用", workspaceFolder.FolderPath, showToast: false);
        RaiseCommandStates();
    }

    private void RemoveWorkspaceFolder(WorkspaceFolderItemViewModel? workspaceFolder)
    {
        if (workspaceFolder is null)
        {
            return;
        }

        var folders = GetWorkspaceFolders(workspaceFolder.Kind);
        var wasActive = workspaceFolder.IsActive;
        folders.Remove(workspaceFolder);

        if (wasActive && folders.Count > 0)
        {
            folders[0].IsActive = true;
        }

        SyncActiveWorkspaceFolderPath(workspaceFolder.Kind);
        _ = SaveSettingsAsync();
        RaiseCommandStates();
        SetStatusSnapshot("工作文件夹已移除", GetWorkspaceFolderKindText(workspaceFolder.Kind), "只移除这个快捷入口，不会删除磁盘上的任何文件。", workspaceFolder.FolderName, ProgressValue, _statusCurrentCount, _statusTotalCount);
        AddOperationLog(OperationLogLevel.Warning, "工作文件夹已移除", workspaceFolder.FolderPath);
    }

    private void LoadWorkspaceFolders(IEnumerable<WorkspaceFolderConfig> folders, WorkspaceFolderKind kind)
    {
        foreach (var folder in folders)
        {
            if (string.IsNullOrWhiteSpace(folder.FolderPath))
            {
                continue;
            }

            GetWorkspaceFolders(kind).Add(new WorkspaceFolderItemViewModel
            {
                Id = string.IsNullOrWhiteSpace(folder.Id) ? Guid.NewGuid().ToString("N") : folder.Id,
                FolderPath = folder.FolderPath,
                Kind = kind,
                IsActive = folder.IsActive,
            });
        }
    }

    private ObservableCollection<WorkspaceFolderItemViewModel> GetWorkspaceFolders(WorkspaceFolderKind kind)
    {
        return kind switch
        {
            WorkspaceFolderKind.Source => SourceFolders,
            WorkspaceFolderKind.SfwTarget => SfwTargetFolders,
            WorkspaceFolderKind.NsfwTarget => NsfwTargetFolders,
            _ => SourceFolders,
        };
    }

    private void SyncActiveWorkspaceFolderPath(WorkspaceFolderKind kind)
    {
        var folders = GetWorkspaceFolders(kind);
        var activeFolder = folders.FirstOrDefault(item => item.IsActive) ?? folders.FirstOrDefault();

        foreach (var folder in folders)
        {
            folder.IsActive = ReferenceEquals(folder, activeFolder);
        }

        var activePath = activeFolder?.FolderPath ?? string.Empty;
        switch (kind)
        {
            case WorkspaceFolderKind.Source:
                SourceFolder = activePath;
                break;
            case WorkspaceFolderKind.SfwTarget:
                SfwTargetFolder = activePath;
                break;
            case WorkspaceFolderKind.NsfwTarget:
                NsfwTargetFolder = activePath;
                break;
        }
    }

    private string GetWorkspaceFolderPickerTitle(WorkspaceFolderKind kind)
    {
        return kind switch
        {
            WorkspaceFolderKind.Source => IsPersonTaskActive ? "选择人物/非人物待分析源文件夹" : "选择待分析源文件夹",
            WorkspaceFolderKind.SfwTarget => $"选择 {PrimaryLabelText} 目标文件夹",
            WorkspaceFolderKind.NsfwTarget => $"选择 {SecondaryLabelText} 目标文件夹",
            _ => "选择工作文件夹",
        };
    }

    private string GetWorkspaceFolderKindText(WorkspaceFolderKind kind)
    {
        return kind switch
        {
            WorkspaceFolderKind.Source => IsPersonTaskActive ? "人物任务源" : "待分析源",
            WorkspaceFolderKind.SfwTarget => $"{PrimaryLabelText} 目标",
            WorkspaceFolderKind.NsfwTarget => $"{SecondaryLabelText} 目标",
            _ => "工作文件夹",
        };
    }

    private void AddSampleFolder(ImageLabel label)
    {
        var labelText = GetLabelDisplayText(label);
        var selectedPath = PickFolder($"选择 {labelText} 样本文件夹");
        if (string.IsNullOrWhiteSpace(selectedPath))
        {
            return;
        }

        if (SfwSampleFolders.Concat(NsfwSampleFolders).Any(item => string.Equals(item.FolderPath, selectedPath, StringComparison.OrdinalIgnoreCase)))
        {
            SetStatusSnapshot("样本库已存在", "无需重复添加", "这个样本文件夹之前已经加入过了。", selectedPath, ProgressValue, _statusCurrentCount, _statusTotalCount);
            AddOperationLog(OperationLogLevel.Info, "样本库已存在", selectedPath);
            return;
        }

        var sampleFolder = new SampleFolderItemViewModel
        {
            Id = Guid.NewGuid().ToString("N"),
            FolderPath = selectedPath,
            Label = label,
            IsEnabled = true,
        };

        AttachSampleFolder(sampleFolder);
        _ = SaveSettingsAsync();
        RaiseSampleFolderSummaryProperties();
        SetStatusSnapshot("样本库已加入", $"{labelText} 样本", $"{labelText} 样本文件夹已加入待学习列表。", Path.GetFileName(selectedPath), ProgressValue, _statusCurrentCount, _statusTotalCount);
        AddOperationLog(OperationLogLevel.Success, "样本库已加入", selectedPath);
    }

    private void RemoveSampleFolder(SampleFolderItemViewModel? sampleFolder)
    {
        if (sampleFolder is null)
        {
            return;
        }

        sampleFolder.PropertyChanged -= OnSampleFolderPropertyChanged;

        if (sampleFolder.Label == PrimaryLabel)
        {
            SfwSampleFolders.Remove(sampleFolder);
        }
        else
        {
            NsfwSampleFolders.Remove(sampleFolder);
        }

        _ = SaveSettingsAsync();
        RaiseSampleFolderSummaryProperties();
        SetStatusSnapshot("样本库已移除", "仅移除挂载", "这个文件夹已从工作区移除，但已经学到的缓存经验会继续保留。", sampleFolder.FolderName, ProgressValue, _statusCurrentCount, _statusTotalCount);
        AddOperationLog(OperationLogLevel.Warning, "样本库已移除", sampleFolder.FolderPath);
    }

    private void AttachSampleFolder(SampleFolderItemViewModel sampleFolder)
    {
        sampleFolder.PropertyChanged += OnSampleFolderPropertyChanged;
        if (sampleFolder.Label == PrimaryLabel)
        {
            SfwSampleFolders.Add(sampleFolder);
        }
        else
        {
            NsfwSampleFolders.Add(sampleFolder);
        }

        RaiseSampleFolderSummaryProperties();
    }

    private async void OnSampleFolderPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(SampleFolderItemViewModel.IsEnabled))
        {
            await SaveSettingsAsync();
            RaiseSampleFolderSummaryProperties();
            if (sender is SampleFolderItemViewModel sampleFolder)
            {
                AddOperationLog(
                    sampleFolder.IsEnabled ? OperationLogLevel.Success : OperationLogLevel.Warning,
                    sampleFolder.IsEnabled ? "样本文件夹已启用" : "样本文件夹已停用",
                    sampleFolder.FolderPath);
            }
        }
    }

    private bool FilterResult(object item)
    {
        if (item is not AnalysisItemViewModel result)
        {
            return false;
        }

        return _activeFilter switch
        {
            ResultFilter.Sfw => result.CurrentLabel == PrimaryLabel,
            ResultFilter.Nsfw => result.CurrentLabel == SecondaryLabel,
            ResultFilter.Uncertain => result.CurrentLabel == ImageLabel.Uncertain,
            ResultFilter.Corrected => result.IsReviewedInCurrentBatch,
            _ => true,
        };
    }

    private async Task SaveSettingsAsync()
    {
        await _workspaceService.SaveSettingsAsync(BuildSettings());
    }

    private void QueueSettingsSave()
    {
        _settingsSaveDebounceTokenSource?.Cancel();
        _settingsSaveDebounceTokenSource?.Dispose();

        _settingsSaveDebounceTokenSource = new CancellationTokenSource();
        var cancellationToken = _settingsSaveDebounceTokenSource.Token;
        _ = SaveSettingsDebouncedAsync(cancellationToken);
    }

    private async Task SaveSettingsDebouncedAsync(CancellationToken cancellationToken)
    {
        try
        {
            await Task.Delay(350, cancellationToken);
            await SaveSettingsAsync();
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
        }
    }

    private WorkspaceSettings BuildSettings()
    {
        UpdateCurrentTaskSettings();
        _settings.TaskMode = ActiveTaskMode;
        _settings.AiModel = new AiModelSettings
        {
            IsEnabled = _aiModelSettings.IsEnabled,
            ModelPath = _aiModelSettings.ModelPath,
            IsNudeDetectorEnabled = _aiModelSettings.IsNudeDetectorEnabled,
            IsGpuAccelerationEnabled = _aiModelSettings.IsGpuAccelerationEnabled,
            NudeDetectorModelPath = _aiModelSettings.NudeDetectorModelPath,
            NudeDetectorInputSize = _aiModelSettings.NudeDetectorInputSize,
            DefaultNsfwThreshold = _aiModelSettings.DefaultNsfwThreshold,
        };
        return _settings;
    }

    private PersonalAiModelSettings BuildPersonalAiSettings(ClassificationTaskMode taskMode)
    {
        var primaryLabel = GetPrimaryLabel(taskMode);
        var secondaryLabel = GetSecondaryLabel(taskMode);
        return new PersonalAiModelSettings
        {
            IsEnabled = _personalAiSettings.IsEnabled,
            AutoTrainEnabled = _personalAiSettings.AutoTrainEnabled,
            PythonPath = _personalAiSettings.PythonPath ?? string.Empty,
            RootFolder = _personalAiSettings.RootFolder ?? string.Empty,
            BaseModelId = string.IsNullOrWhiteSpace(_personalAiSettings.BaseModelId) ||
                          string.Equals(_personalAiSettings.BaseModelId, "google/vit-base-patch16-224-in21k", StringComparison.OrdinalIgnoreCase)
                ? "google/vit-base-patch16-224"
                : _personalAiSettings.BaseModelId,
            MaxModelVersions = _personalAiSettings.MaxModelVersions <= 0 ? 2 : Math.Clamp(_personalAiSettings.MaxModelVersions, 1, 2),
            TrainEpochs = _personalAiSettings.TrainEpochs <= 0 ? 3 : _personalAiSettings.TrainEpochs,
            TrainBatchSize = PersonalAiTrainingService.NormalizePersonalAiBatchSize(_personalAiSettings.TrainBatchSize),
            TriggerCorrectionCount = _personalAiSettings.TriggerCorrectionCount <= 0 ? 50 : _personalAiSettings.TriggerCorrectionCount,
            LearningRate = _personalAiSettings.LearningRate <= 0d ? 0.0002d : _personalAiSettings.LearningRate,
            DecisionThreshold = _personalAiSettings.DecisionThreshold <= 0d ? 0.5d : _personalAiSettings.DecisionThreshold,
            MinimumConfidence = _personalAiSettings.MinimumConfidence <= 0d ? 0.56d : _personalAiSettings.MinimumConfidence,
            TrainingRoots = (_personalAiSettings.TrainingRoots ?? [])
                .Where(root => !string.IsNullOrWhiteSpace(root.FolderPath) &&
                               (root.Label == primaryLabel || root.Label == secondaryLabel))
                .Select(root => new PersonalAiTrainingRoot
                {
                    FolderPath = root.FolderPath,
                    Label = root.Label,
                    IsEnabled = root.IsEnabled,
                })
                .ToList(),
            EvaluationSamplesPerLabel = _personalAiSettings.EvaluationSamplesPerLabel <= 0 ? 500 : _personalAiSettings.EvaluationSamplesPerLabel,
            PrimaryModelMinimumAccuracy = _personalAiSettings.PrimaryModelMinimumAccuracy <= 0d ? 0.92d : _personalAiSettings.PrimaryModelMinimumAccuracy,
            PrimaryModelMaximumNsfwFalseNegativeRate = _personalAiSettings.PrimaryModelMaximumNsfwFalseNegativeRate <= 0d ? 0.04d : _personalAiSettings.PrimaryModelMaximumNsfwFalseNegativeRate,
            PrimaryModelMaximumSfwFalsePositiveRate = _personalAiSettings.PrimaryModelMaximumSfwFalsePositiveRate <= 0d ? 0.08d : _personalAiSettings.PrimaryModelMaximumSfwFalsePositiveRate,
        };
    }

    private void ApplyPersonalAiSettings(PersonalAiModelSettings? settings)
    {
        if (settings is null)
        {
            return;
        }

        _personalAiSettings = settings;
        if (IsPersonTaskActive)
        {
            _settings.PersonTask ??= new PersonTaskSettings();
            _settings.PersonTask.PersonalAi = settings;
        }
        else
        {
            _settings.PersonalAi = settings;
        }
        RefreshPersonalAiStatus(_settings);
    }

    private PersonalAiModelSettings GetActivePersonalAiSettings(WorkspaceSettings settings)
    {
        return IsPersonTaskActive
            ? settings.PersonTask?.PersonalAi ?? new PersonalAiModelSettings()
            : settings.PersonalAi ?? new PersonalAiModelSettings();
    }

    private static string ResolveActiveWorkspaceFolder(IEnumerable<WorkspaceFolderItemViewModel> folders, string fallbackPath)
    {
        return folders.FirstOrDefault(item => item.IsActive)?.FolderPath ?? fallbackPath;
    }

    private static List<WorkspaceFolderConfig> BuildWorkspaceFolderConfigs(IEnumerable<WorkspaceFolderItemViewModel> folders, string activePath)
    {
        var configs = folders
            .Where(item => !string.IsNullOrWhiteSpace(item.FolderPath))
            .Select(item => item.ToConfig())
            .ToList();

        if (!string.IsNullOrWhiteSpace(activePath) &&
            configs.All(item => !string.Equals(item.FolderPath, activePath, StringComparison.OrdinalIgnoreCase)))
        {
            configs.Insert(0, new WorkspaceFolderConfig
            {
                FolderPath = activePath,
                IsActive = true,
            });
        }

        var activeConfig = !string.IsNullOrWhiteSpace(activePath)
            ? configs.FirstOrDefault(item => string.Equals(item.FolderPath, activePath, StringComparison.OrdinalIgnoreCase))
            : configs.FirstOrDefault(item => item.IsActive) ?? configs.FirstOrDefault();

        foreach (var config in configs)
        {
            config.IsActive = ReferenceEquals(config, activeConfig);
        }

        return configs;
    }

    private List<SampleFolderConfig> BuildSampleFolderConfigs()
    {
        return SfwSampleFolders
            .Concat(NsfwSampleFolders)
            .Where(item => IsCurrentTaskLabel(item.Label))
            .Select(item => item.ToConfig())
            .ToList();
    }

    private void RefreshAiModelStatus(WorkspaceSettings settings)
    {
        var status = _workspaceService.GetAiModelStatus(settings);
        _isAiModelInstalled = status.IsInstalled;
        AiModelStatusText = status.Message;
        AiModelButtonText = _isAiModelDownloading
            ? "正在下载 AI 模型包"
            : status.IsInstalled
                ? "AI 模型包已安装"
                : "下载 AI 模型包";
        RaiseCommandStates();
    }

    private void RefreshPersonalAiStatus(WorkspaceSettings settings)
    {
        _personalAiSettings = IsPersonTaskActive
            ? settings.PersonTask?.PersonalAi ?? new PersonalAiModelSettings()
            : settings.PersonalAi ?? new PersonalAiModelSettings();
        var status = _workspaceService.GetPersonalAiTrainingStatus(settings);
        var modelText = IsPersonTaskActive ? "人物模型" : "个人大模型";
        PersonalAiStatusText = BuildPersonalAiSummary(status);
        PersonalAiModelVersions = status.ModelVersions;
        HasActivePersonalAiModel = status.HasActiveModel;
        HasCandidatePersonalAiModel = status.HasCandidateModel;
        ShowPersonalAiCurrentModelState = status.HasActiveModel && !status.HasCandidateModel;
        PersonalAiStatusBadgeText = BuildPersonalAiStatusBadgeText(status);
        PersonalAiStatusBadgeDetailText = BuildPersonalAiStatusBadgeDetailText(status);
        PersonalAiModePillText = BuildPersonalAiModePillText(status);
        PersonalAiActiveModelEvaluationText = BuildPersonalAiEvaluationText(
            status.ActiveModelEvaluationAccuracy,
            status.ActiveModelEvaluatedSamples);
        PersonalAiCandidateModelVersionText = string.IsNullOrWhiteSpace(status.CandidateModelVersion)
            ? "候选版本待生成"
            : $"候选版本：{status.CandidateModelVersion}";
        PersonalAiCandidateModelEvaluationText = BuildPersonalAiEvaluationText(
            status.CandidateModelEvaluationAccuracy,
            status.CandidateModelEvaluatedSamples);
        PersonalAiButtonText = _isPersonalAiTraining
            ? $"正在训练{modelText}"
            : status.HasResumeCheckpoint
                ? $"继续断点训练{modelText}"
            : status.HasActiveModel
                ? $"继续训练{modelText}"
                : $"训练{modelText}";
        ActivateCandidateModelButtonText = status.HasCandidateModel
            ? "启用候选模型"
            : "当前已启用";
        RaiseCommandStates();
    }

    private static string BuildPersonalAiSummary(PersonalAiTrainingStatus status)
    {
        var activeModel = status.ModelVersions.FirstOrDefault(model => model.IsActive);
        if (activeModel is not null)
        {
            return $"已就绪 · {activeModel.SummaryText}";
        }

        var candidateModel = status.ModelVersions.FirstOrDefault(model => model.IsCandidate);
        if (candidateModel is not null)
        {
            return $"候选模型待启用 · {candidateModel.SummaryText}";
        }

        return status.TrainingSampleCount > 0
            ? $"等待训练 · 已收集 {status.TrainingSampleCount} 张样本"
            : "等待训练";
    }

    private static string BuildPersonalAiStatusBadgeText(PersonalAiTrainingStatus status)
    {
        if (status.HasActiveModel)
        {
            return status.AllowsPersonalModelPrimary ? "已启用主判" : "已启用个人模型";
        }

        return status.HasCandidateModel ? "候选模型待启用" : "个人模型未启用";
    }

    private static string BuildPersonalAiStatusBadgeDetailText(PersonalAiTrainingStatus status)
    {
        if (status.HasActiveModel)
        {
            return string.IsNullOrWhiteSpace(status.ActiveModelVersion)
                ? "当前启用版本已生效"
                : $"当前启用版本：{status.ActiveModelVersion}";
        }

        if (status.HasCandidateModel)
        {
            return string.IsNullOrWhiteSpace(status.CandidateModelVersion)
                ? "训练已生成候选模型，启用后才会参与正式判断。"
                : $"候选版本：{status.CandidateModelVersion}，启用后才会参与正式判断。";
        }

        return "训练并启用候选模型后，个人模型才会参与正式判断。";
    }

    private static string BuildPersonalAiModePillText(PersonalAiTrainingStatus status)
    {
        if (status.HasActiveModel)
        {
            return status.AllowsPersonalModelPrimary ? "主判生效" : "辅助生效";
        }

        return status.HasCandidateModel ? "等待启用" : "微调模型";
    }

    private static string BuildPersonalAiEvaluationText(double? accuracy, int evaluatedSamples)
    {
        return accuracy.HasValue && evaluatedSamples > 0
            ? $"评测准确率：{accuracy.Value:P1}（{evaluatedSamples} 张）"
            : "评测准确率：暂无评测";
    }

    private static string? PickFolder(string description)
    {
        using var dialog = new Forms.FolderBrowserDialog
        {
            Description = description,
            UseDescriptionForTitle = true,
            ShowNewFolderButton = true,
        };

        return dialog.ShowDialog() == Forms.DialogResult.OK ? dialog.SelectedPath : null;
    }

    private void SetFilter(ResultFilter filter)
    {
        _activeFilter = filter;
        RaisePropertyChanged(nameof(IsAllFilterActive));
        RaisePropertyChanged(nameof(IsSfwFilterActive));
        RaisePropertyChanged(nameof(IsNsfwFilterActive));
        RaisePropertyChanged(nameof(IsUncertainFilterActive));
        RaisePropertyChanged(nameof(IsCorrectedFilterActive));
        _resultsView.Refresh();
        SelectFirstVisibleResult();
        RaisePropertyChanged(nameof(SelectedPreviewPositionText));
        RaiseCommandStates();
    }

    private void SelectFirstVisibleResult()
    {
        SelectedResult = _resultsView.Cast<AnalysisItemViewModel>().FirstOrDefault();
    }

    private void ReleaseAllResultPreviews()
    {
        _previewLoadTokenSource?.Cancel();
        _previewLoadTokenSource?.Dispose();
        _previewLoadTokenSource = null;

        foreach (var result in Results)
        {
            result.ReleasePreview();
        }
    }

    private void StartPreviewLoading(AnalysisItemViewModel? selectedResult)
    {
        _previewLoadTokenSource?.Cancel();
        _previewLoadTokenSource?.Dispose();
        _previewLoadTokenSource = null;

        if (selectedResult is null)
        {
            return;
        }

        _previewLoadTokenSource = CancellationTokenSource.CreateLinkedTokenSource(_lifetimeTokenSource.Token);
        _ = LoadSelectedAndNeighborPreviewsAsync(selectedResult, _previewLoadTokenSource.Token);
    }

    private async Task LoadSelectedAndNeighborPreviewsAsync(AnalysisItemViewModel selectedResult, CancellationToken cancellationToken)
    {
        try
        {
            await selectedResult.LoadPreviewAsync(cancellationToken).ConfigureAwait(true);
            foreach (var neighbor in GetPreviewPreloadItems(selectedResult))
            {
                cancellationToken.ThrowIfCancellationRequested();
                await neighbor.PreloadPreviewAsync(cancellationToken).ConfigureAwait(true);
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    private IReadOnlyList<AnalysisItemViewModel> GetPreviewPreloadItems(AnalysisItemViewModel selectedResult)
    {
        var visibleResults = _resultsView.Cast<AnalysisItemViewModel>().ToList();
        var selectedIndex = visibleResults.IndexOf(selectedResult);
        if (selectedIndex < 0)
        {
            return [];
        }

        var preloadItems = new List<AnalysisItemViewModel>(PreviewPreloadNeighborCount * 2);
        for (var offset = 1; offset <= PreviewPreloadNeighborCount; offset++)
        {
            var nextIndex = selectedIndex + offset;
            if (nextIndex < visibleResults.Count)
            {
                preloadItems.Add(visibleResults[nextIndex]);
            }

            var previousIndex = selectedIndex - offset;
            if (previousIndex >= 0)
            {
                preloadItems.Add(visibleResults[previousIndex]);
            }
        }

        return preloadItems;
    }

    private void TrimPreviewCacheAroundSelection()
    {
        if (Results.Count == 0)
        {
            return;
        }

        var retainedItems = new HashSet<AnalysisItemViewModel>();
        if (_selectedResult is not null)
        {
            retainedItems.Add(_selectedResult);

            var visibleResults = _resultsView.Cast<AnalysisItemViewModel>().ToList();
            var selectedIndex = visibleResults.IndexOf(_selectedResult);
            if (selectedIndex >= 0)
            {
                var startIndex = Math.Max(0, selectedIndex - PreviewCacheRadius);
                var endIndex = Math.Min(visibleResults.Count - 1, selectedIndex + PreviewCacheRadius);
                for (var index = startIndex; index <= endIndex; index++)
                {
                    retainedItems.Add(visibleResults[index]);
                }
            }
        }

        foreach (var result in Results)
        {
            if (!retainedItems.Contains(result))
            {
                result.ReleasePreview();
            }
        }
    }

    private void RefreshSummaries()
    {
        RaisePropertyChanged(nameof(TotalCount));
        RaisePropertyChanged(nameof(SfwCount));
        RaisePropertyChanged(nameof(NsfwCount));
        RaisePropertyChanged(nameof(UncertainCount));
        RaisePropertyChanged(nameof(CorrectedCount));
        RaisePropertyChanged(nameof(PendingConfirmationCount));
        RaisePropertyChanged(nameof(ReadyToMoveCount));
        RaisePropertyChanged(nameof(ResultHeadline));
        RaisePropertyChanged(nameof(ResultSubheadline));
        RaisePropertyChanged(nameof(SelectedPreviewPositionText));
        RaisePropertyChanged(nameof(ConfirmMoveButtonText));
        RaiseCommandStates();
    }

    private string GetSelectedPreviewPositionText()
    {
        var visibleResults = _resultsView.Cast<AnalysisItemViewModel>().ToList();
        if (visibleResults.Count == 0)
        {
            return "0 / 0";
        }

        if (SelectedResult is null)
        {
            return $"0 / {visibleResults.Count}";
        }

        var selectedIndex = visibleResults.IndexOf(SelectedResult);
        return selectedIndex < 0
            ? $"未在当前筛选 / {visibleResults.Count}"
            : $"{selectedIndex + 1} / {visibleResults.Count}";
    }

    private bool CanStartAnalysis()
    {
        return !IsBusy &&
               !string.IsNullOrWhiteSpace(SourceFolder) &&
               Directory.Exists(SourceFolder) &&
               SfwSampleFolders.Concat(NsfwSampleFolders).Any(folder => folder.IsEnabled);
    }

    private bool CanApplyManualCorrection(ImageLabel targetLabel)
    {
        if (SelectedResult is null || IsBusy)
        {
            return false;
        }

        return !((targetLabel == PrimaryLabel && _activeFilter == ResultFilter.Sfw) ||
                  (targetLabel == SecondaryLabel && _activeFilter == ResultFilter.Nsfw));
    }

    private bool CanConfirmMove()
    {
        return !IsBusy &&
               Results.Any(item => item.CurrentLabel == PrimaryLabel || item.CurrentLabel == SecondaryLabel);
    }

    private bool CanRefreshSamples()
    {
        return !IsBusy && SfwSampleFolders.Concat(NsfwSampleFolders).Any(folder => folder.IsEnabled);
    }

    private bool CanInstallAiModel()
    {
        return IsContentSafetyTaskActive && !IsBusy && !_isAiModelDownloading && !_isAiModelInstalled;
    }

    private bool CanTrainPersonalAi()
    {
        return !IsBusy && _personalAiSettings.IsEnabled;
    }

    private bool CanActivateCandidateModel()
    {
        return !IsBusy &&
               !string.IsNullOrWhiteSpace(_workspaceService.GetPersonalAiTrainingStatus(_settings).CandidateModelVersion);
    }

    private bool CanStopCurrentScan()
    {
        return IsBusy && _activeScanTokenSource is { IsCancellationRequested: false };
    }

    private CancellationTokenSource BeginCancellableScan()
    {
        _activeScanTokenSource?.Dispose();
        _activeScanTokenSource = CancellationTokenSource.CreateLinkedTokenSource(_lifetimeTokenSource.Token);
        ResetProgressLogState();
        RaiseCommandStates();
        return _activeScanTokenSource;
    }

    private void ResetProgressLogState()
    {
        _lastMoveProgressLogCount = 0;
        _lastTrainingProgressLogCount = -1;
        _lastTrainingProgressLogSignature = string.Empty;
        _lastGeneralProgressLogStage = WorkspaceProgressStage.Idle;
        _lastGeneralProgressLogCount = -1;
        _lastGeneralProgressLogSignature = string.Empty;
    }

    private void EndCancellableScan(CancellationTokenSource scanTokenSource)
    {
        if (ReferenceEquals(_activeScanTokenSource, scanTokenSource))
        {
            _activeScanTokenSource = null;
        }

        scanTokenSource.Dispose();
        RaiseCommandStates();
    }

    private void StopCurrentScan()
    {
        if (_activeScanTokenSource is null || _activeScanTokenSource.IsCancellationRequested)
        {
            return;
        }

        _activeScanTokenSource.Cancel();
        SetStatusSnapshot("正在停止", "取消任务", "已发送停止请求，当前正在收尾。", StatusCurrentItem, ProgressValue, _statusCurrentCount, _statusTotalCount);
        AddOperationLog(OperationLogLevel.Warning, "正在停止任务", "已发送停止请求，等待当前步骤收尾。");
        RaiseCommandStates();
    }

    private void RaiseCommandStates()
    {
        StartAnalysisCommand.RaiseCanExecuteChanged();
        ConfirmMoveCommand.RaiseCanExecuteChanged();
        RefreshSamplesCommand.RaiseCanExecuteChanged();
        InstallAiModelCommand.RaiseCanExecuteChanged();
        TrainPersonalAiCommand.RaiseCanExecuteChanged();
        ActivateCandidateModelCommand.RaiseCanExecuteChanged();
        ActivatePersonalAiModelCommand.RaiseCanExecuteChanged();
        StopScanCommand.RaiseCanExecuteChanged();
        MarkAsSfwCommand.RaiseCanExecuteChanged();
        MarkAsNsfwCommand.RaiseCanExecuteChanged();
        MarkAsUncertainCommand.RaiseCanExecuteChanged();
        ShowContentSafetyTaskCommand.RaiseCanExecuteChanged();
        ShowPersonTaskCommand.RaiseCanExecuteChanged();
    }

    private void HandleWorkspaceProgress(WorkspaceProgressInfo progress)
    {
        var ratio = progress.Total > 0 ? (double)progress.Current / progress.Total : 0d;
        var scaledRatio = progress.Stage switch
        {
            WorkspaceProgressStage.RefreshingSamples => 0.08d + (ratio * 0.14d),
            WorkspaceProgressStage.ExtractingAiFeatures when progress.Title.Contains("源图", StringComparison.OrdinalIgnoreCase) => 0.46d + (ratio * 0.30d),
            WorkspaceProgressStage.ExtractingAiFeatures => 0.22d + (ratio * 0.20d),
            WorkspaceProgressStage.PreparingSource => 0.42d + (ratio * 0.04d),
            WorkspaceProgressStage.AnalyzingSource => 0.78d + (ratio * 0.16d),
            WorkspaceProgressStage.MovingFiles => 0.10d + (ratio * 0.84d),
            WorkspaceProgressStage.DownloadingModel => ratio,
            WorkspaceProgressStage.TrainingPersonalModel => ratio,
            WorkspaceProgressStage.Completed => 1d,
            _ => ratio,
        };

        SetStatusSnapshot(
            progress.Title,
            GetStageDisplayName(progress.Stage),
            progress.Detail,
            string.IsNullOrWhiteSpace(progress.CurrentItem) ? "正在处理本轮任务..." : progress.CurrentItem,
            scaledRatio,
            progress.Current,
            progress.Total);

        HandlePersonalTrainingWorkspaceProgress(progress);
        HandleGeneralWorkspaceProgress(progress);
    }

    private void HandleGeneralWorkspaceProgress(WorkspaceProgressInfo progress)
    {
        if (progress.Stage is WorkspaceProgressStage.TrainingPersonalModel or WorkspaceProgressStage.MovingFiles)
        {
            return;
        }

        if (!ShouldLogGeneralWorkspaceProgress(progress))
        {
            return;
        }

        _lastGeneralProgressLogStage = progress.Stage;
        _lastGeneralProgressLogCount = progress.Current;
        _lastGeneralProgressLogSignature = BuildProgressLogSignature(progress);
        var title = progress.Total > 0
            ? $"{GetStageDisplayName(progress.Stage)} {progress.Current} / {progress.Total}"
            : GetStageDisplayName(progress.Stage);
        var detail = string.IsNullOrWhiteSpace(progress.CurrentItem)
            ? progress.Detail
            : $"{progress.Detail} 当前：{progress.CurrentItem}";
        AddOperationLog(OperationLogLevel.Info, title, detail, showToast: false);
    }

    private void HandlePersonalTrainingWorkspaceProgress(WorkspaceProgressInfo progress)
    {
        if (progress.Stage != WorkspaceProgressStage.TrainingPersonalModel)
        {
            return;
        }

        var signature = $"{progress.Current}/{progress.Total}|{progress.Title}|{progress.CurrentItem}|{progress.Detail}";
        if (string.Equals(signature, _lastTrainingProgressLogSignature, StringComparison.Ordinal))
        {
            return;
        }

        if (!ShouldLogPersonalTrainingProgress(progress))
        {
            return;
        }

        _lastTrainingProgressLogSignature = signature;
        _lastTrainingProgressLogCount = progress.Current;
        var progressText = progress.Total > 0
            ? $"训练进度 {progress.Current} / {progress.Total}"
            : "个人训练进度";
        AddOperationLog(
            OperationLogLevel.Info,
            progressText,
            $"{progress.Detail} 当前步骤：{progress.CurrentItem}",
            showToast: false);
    }

    private bool ShouldLogPersonalTrainingProgress(WorkspaceProgressInfo progress)
    {
        if (progress.Total <= 20)
        {
            return true;
        }

        if (progress.Current <= 0)
        {
            return _lastTrainingProgressLogCount != progress.Current;
        }

        if (progress.Current >= progress.Total)
        {
            return true;
        }

        var step = Math.Max(1, progress.Total / 20);
        return progress.Current != _lastTrainingProgressLogCount &&
               progress.Current % step == 0;
    }

    private bool ShouldLogGeneralWorkspaceProgress(WorkspaceProgressInfo progress)
    {
        var signature = BuildProgressLogSignature(progress);
        if (string.Equals(signature, _lastGeneralProgressLogSignature, StringComparison.Ordinal))
        {
            return false;
        }

        if (progress.Stage != _lastGeneralProgressLogStage)
        {
            return true;
        }

        if (progress.Total <= 0)
        {
            return true;
        }

        if (progress.Current <= 0)
        {
            return true;
        }

        if (progress.Current >= progress.Total)
        {
            return true;
        }

        if (progress.Current == _lastGeneralProgressLogCount)
        {
            return false;
        }

        var step = progress.Stage == WorkspaceProgressStage.ExtractingAiFeatures
            ? Math.Max(1, Math.Min(128, progress.Total / 12))
            : Math.Max(1, progress.Total / 10);
        return progress.Current % step == 0;
    }

    private static string BuildProgressLogSignature(WorkspaceProgressInfo progress)
    {
        return $"{progress.Stage}|{progress.Current}/{progress.Total}|{progress.Title}|{progress.CurrentItem}|{progress.Detail}";
    }

    private void HandleMoveWorkspaceProgress(WorkspaceProgressInfo progress)
    {
        HandleWorkspaceProgress(progress);

        if (progress.Stage != WorkspaceProgressStage.MovingFiles || progress.Current <= 0)
        {
            return;
        }

        if (progress.Current == _lastMoveProgressLogCount)
        {
            return;
        }

        _lastMoveProgressLogCount = progress.Current;
        AddOperationLog(
            OperationLogLevel.Info,
            $"移动进度 {progress.Current} / {progress.Total}",
            $"{progress.Detail} 当前文件：{progress.CurrentItem}",
            showToast: false);
    }

    private void SetStatusSnapshot(string title, string stageText, string detail, string currentItem, double progressValue, int currentCount, int totalCount)
    {
        StatusTitle = title;
        StatusStageText = stageText;
        StatusMessage = detail;
        StatusCurrentItem = currentItem;
        _statusCurrentCount = currentCount;
        _statusTotalCount = totalCount;
        ProgressValue = Math.Clamp(progressValue, 0d, 1d);
        RaisePropertyChanged(nameof(ProgressPercentText));
        RaisePropertyChanged(nameof(ProgressCountText));
        RaisePropertyChanged(nameof(HasProgressNumbers));
    }

    private void AddOperationLog(OperationLogLevel level, string message, string detail, bool showToast = true)
    {
        OperationLogs.Insert(0, new OperationLogItemViewModel(level, message, detail));
        while (OperationLogs.Count > 80)
        {
            OperationLogs.RemoveAt(OperationLogs.Count - 1);
        }

        if (showToast)
        {
            ShowToast(message, level);
        }
    }

    private static string BuildAnalysisCompletionDetail(IReadOnlyList<AnalysisRecord> records)
    {
        var aiModelCount = records.Count(item => item.UsedAiModel);
        var personalLearningCount = records.Count(item => item.UsedPersonalLearning);
        var personalLoraCount = records.Count(item => item.UsedPersonalLora);
        var historyCacheCount = records.Count(item => item.UsedHistoryCache);
        var uncertainCount = records.Count(item => item.FinalLabel == ImageLabel.Uncertain);
        var aiFinalCount = records.Count(item => item.LabelOrigin is LabelOrigin.AiModel or LabelOrigin.PersonalAi);
        var manualCacheCount = records.Count(item => item.LabelOrigin == LabelOrigin.ManualCorrection);
        var sampleCacheCount = records.Count(item => item.LabelOrigin == LabelOrigin.SampleLibrary);
        var originSummary = string.Join("，", records
            .GroupBy(item => item.LabelOrigin)
            .OrderByDescending(group => group.Count())
            .Select(group => $"{GetLabelOriginSummaryText(group.Key)} {group.Count()} 张"));

        return $"本轮生成 {records.Count} 条结果，其中不确定 {uncertainCount} 张。最终由 AI 主判 {aiFinalCount} 张，其中通用 AI 视觉模型参与 {aiModelCount} 张，个人 AI 校准参与 {personalLearningCount} 张，个人 LoRA 大模型参与 {personalLoraCount} 张；历史缓存直接命中 {historyCacheCount} 张，其中人工纠正缓存 {manualCacheCount} 张，样本库缓存 {sampleCacheCount} 张。来源分布：{originSummary}。";
    }

    private static string GetLabelOriginSummaryText(LabelOrigin labelOrigin)
    {
        return labelOrigin switch
        {
            LabelOrigin.SampleLibrary => "样本库",
            LabelOrigin.ModelPrediction => "自动判断",
            LabelOrigin.ManualCorrection => "人工纠正",
            LabelOrigin.ConfirmedMove => "最终确认",
            LabelOrigin.AiModel => "AI 视觉模型",
            LabelOrigin.PersonalAi => "个人 AI",
            _ => "未知来源",
        };
    }

    private void CopyOperationLog(OperationLogItemViewModel? logItem)
    {
        if (logItem is null)
        {
            return;
        }

        try
        {
            System.Windows.Clipboard.SetText(logItem.CopyText);
            ShowToast("日志已复制", OperationLogLevel.Success);
        }
        catch (Exception exception)
        {
            ShowToast($"复制失败：{exception.Message}", OperationLogLevel.Error);
        }
    }

    private void ShowToast(string message, OperationLogLevel level)
    {
        _toastTokenSource?.Cancel();
        _toastTokenSource?.Dispose();
        _toastTokenSource = CancellationTokenSource.CreateLinkedTokenSource(_lifetimeTokenSource.Token);

        ToastMessage = message;
        ToastLevel = level;
        IsToastVisible = true;

        _ = HideToastLaterAsync(_toastTokenSource.Token);
    }

    private async Task HideToastLaterAsync(CancellationToken cancellationToken)
    {
        try
        {
            await Task.Delay(2400, cancellationToken);
            IsToastVisible = false;
        }
        catch (OperationCanceledException)
        {
        }
    }

    private void RaiseTaskModePropertiesChanged()
    {
        RaisePropertyChanged(nameof(IsContentSafetyTaskActive));
        RaisePropertyChanged(nameof(IsPersonTaskActive));
        RaisePropertyChanged(nameof(PrimaryLabel));
        RaisePropertyChanged(nameof(SecondaryLabel));
        RaisePropertyChanged(nameof(AppSubtitle));
        RaisePropertyChanged(nameof(TaskModeTitle));
        RaisePropertyChanged(nameof(TaskModeDescription));
        RaisePropertyChanged(nameof(TaskConfigDescription));
        RaisePropertyChanged(nameof(SourceFolderTitle));
        RaisePropertyChanged(nameof(PrimaryLabelText));
        RaisePropertyChanged(nameof(SecondaryLabelText));
        RaisePropertyChanged(nameof(PrimarySampleTitle));
        RaisePropertyChanged(nameof(SecondarySampleTitle));
        RaisePropertyChanged(nameof(AddPrimarySampleButtonText));
        RaisePropertyChanged(nameof(AddSecondarySampleButtonText));
        RaisePropertyChanged(nameof(PrimaryTargetTitle));
        RaisePropertyChanged(nameof(SecondaryTargetTitle));
        RaisePropertyChanged(nameof(AddPrimaryTargetButtonText));
        RaisePropertyChanged(nameof(AddSecondaryTargetButtonText));
        RaisePropertyChanged(nameof(PrimaryFilterLabel));
        RaisePropertyChanged(nameof(SecondaryFilterLabel));
        RaisePropertyChanged(nameof(StartAnalysisButtonText));
        RaisePropertyChanged(nameof(PersonalAiPanelTitle));
        RaisePropertyChanged(nameof(PersonalAiModelTitle));
        RaisePropertyChanged(nameof(ConfirmMoveButtonText));
        RaisePropertyChanged(nameof(MarkAsPrimaryButtonText));
        RaisePropertyChanged(nameof(MarkAsSecondaryButtonText));
        RaiseSampleFolderSummaryProperties();
        RefreshSummaries();
    }

    private void RaiseSampleFolderSummaryProperties()
    {
        RaisePropertyChanged(nameof(PrimarySampleCountText));
        RaisePropertyChanged(nameof(SecondarySampleCountText));
        RaisePropertyChanged(nameof(SampleLibrarySummary));
    }

    private bool IsCurrentTaskLabel(ImageLabel label)
    {
        return label == PrimaryLabel || label == SecondaryLabel;
    }

    private static ImageLabel GetPrimaryLabel(ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence ? ImageLabel.Person : ImageLabel.Sfw;
    }

    private static ImageLabel GetSecondaryLabel(ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence ? ImageLabel.NonPerson : ImageLabel.Nsfw;
    }

    private static string GetLabelDisplayText(ImageLabel label)
    {
        return label switch
        {
            ImageLabel.Sfw => "SFW",
            ImageLabel.Nsfw => "NSFW",
            ImageLabel.Person => "人物",
            ImageLabel.NonPerson => "非人物",
            ImageLabel.Uncertain => "不确定",
            _ => "未标注",
        };
    }

    private string GetStageDisplayName(WorkspaceProgressStage stage)
    {
        return stage switch
        {
            WorkspaceProgressStage.Loading => "初始化",
            WorkspaceProgressStage.RefreshingSamples => "样本扫描",
            WorkspaceProgressStage.ExtractingAiFeatures => "AI 特征",
            WorkspaceProgressStage.PreparingSource => "准备源图",
            WorkspaceProgressStage.AnalyzingSource => IsPersonTaskActive ? "人物分析" : "内容分析",
            WorkspaceProgressStage.ApplyingCorrection => "人工修正",
            WorkspaceProgressStage.MovingFiles => "移动文件",
            WorkspaceProgressStage.DownloadingModel => "模型下载",
            WorkspaceProgressStage.TrainingPersonalModel => "个人训练",
            WorkspaceProgressStage.Completed => "完成",
            _ => "空闲",
        };
    }

    private enum ResultFilter
    {
        All,
        Sfw,
        Nsfw,
        Uncertain,
        Corrected,
    }
}
