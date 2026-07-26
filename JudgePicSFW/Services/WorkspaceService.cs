using System.Diagnostics;
using System.IO;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class WorkspaceService
{
    private const double NsfwScoreBias = 1.12d;
    private const double ManualCorrectionWeight = 2.5d;
    private const double ConservativeVoteGapThreshold = 0.18d;
    private const double ConservativeConfidenceThreshold = 0.55d;
    private const int ManualDuplicateDistanceThreshold = 2;
    private const int NearbyNsfwDistanceThreshold = 8;
    private const int NearbyLabelWinningDistance = 4;
    private const int UnknownPatternDistanceThreshold = 32;
    private const int PersonalCalibrationMinimumTrainingSamples = 12;
    private const int PersonalCalibrationMinimumSamplesPerLabel = 3;
    private const double PersonalCalibrationMinimumScoreGap = 0.08d;
    private const int PersonalAiMinimumTrainingSamples = 32;
    private const int PersonalAiMinimumSamplesPerLabel = 6;
    private const int PersonalAiEarlyLearningMinimumSamples = 2;
    private const double PersonalAiMaximumClosestDistance = 0.16d;
    private const double PersonalAiRelevantNeighborDistance = 0.2d;
    private const double PersonalAiMinimumConfidence = 0.7d;
    private const double UserPreferenceAiMinimumConfidence = 0.42d;
    private const double UserPreferenceNsfwOverrideConfidence = 0.46d;
    private const double PersonalAiStrongConfidence = 0.9d;
    private const double PersonalAiStrongClosestDistance = 0.07d;
    private const int PersonalAiMinimumSameLabelCloseNeighbors = 2;
    private const int PersonalAiStrongSameLabelCloseNeighbors = 3;
    private const double PersonalAiMaximumBlendWeight = 0.2d;
    private const double PersonalAiModelMaximumBlendWeight = 0.24d;
    private const double PersonalAiLimitedNsfwBlendWeight = 0.04d;
    private const double LowExplicitNsfwGuardThreshold = 0.08d;
    private const double LowSuggestiveGuardThreshold = 0.18d;
    private const double PersonalAiNsfwPromotionExplicitThreshold = 0.18d;
    private const double PersonalAiNsfwPromotionSuggestiveThreshold = 0.55d;
    private const double PersonalAiSuggestiveRequiresExplicitThreshold = 0.08d;
    private const int HardNsfwSampleDistanceThreshold = 3;
    private const int HardNsfwSampleWinningDistance = 10;
    private const int PersonalAiModelEpochs = 140;
    private const int PersonalAiModelMaxSamplesPerLabel = 2500;
    private const double PersonalAiModelLearningRate = 0.08d;
    private const double PersonalAiModelL2 = 0.004d;
    private const int AiFeatureExtractionBatchSize = 64;
    private const int AiFeatureExtractionPersistInterval = 512;
    private const int SmallLearnedIndexFullScanThreshold = 384;
    private const int SampleNeighborMinimumCandidateCount = 48;
    private const int AiCorrectionFeatureBucketCount = 32;
    private const int AiCorrectionCandidateFullScanThreshold = 800;
    private const int AiCorrectionMinimumCandidateCount = 192;
    private const int AiCorrectionDistanceEstimateWindow = 96;
    private static readonly double[] AiFeatureWeights =
    [
        2.0d, 2.5d, 1.7d, 3.0d, 3.2d, 3.4d, 3.0d, 2.0d, 1.5d,
        1.2d, 1.4d, 1.6d, 1.6d, 0.9d, 0.7d, 0.9d, 0.6d, 0.5d,
    ];

    private readonly AppStateStore _appStateStore;
    private readonly ImageFingerprintService _imageFingerprintService;
    private readonly AiNsfwClassifierService _aiNsfwClassifierService;
    private readonly PersonalAiTrainingService _personalAiTrainingService;
    private readonly WindowsShellFileMoveService _fileMoveService;
    private readonly PerformanceLogService _performanceLogService;
    private PersistedAppState? _state;

    public WorkspaceService(
        AppStateStore appStateStore,
        ImageFingerprintService imageFingerprintService,
        AiNsfwClassifierService aiNsfwClassifierService,
        PersonalAiTrainingService personalAiTrainingService,
        WindowsShellFileMoveService fileMoveService,
        PerformanceLogService performanceLogService)
    {
        _appStateStore = appStateStore;
        _imageFingerprintService = imageFingerprintService;
        _aiNsfwClassifierService = aiNsfwClassifierService;
        _personalAiTrainingService = personalAiTrainingService;
        _fileMoveService = fileMoveService;
        _performanceLogService = performanceLogService;
    }

    public async Task<WorkspaceSettings> InitializeAsync()
    {
        _state = await _appStateStore.LoadAsync();
        return CloneSettings(_state.Settings);
    }

    public async Task SaveSettingsAsync(WorkspaceSettings settings)
    {
        var state = await GetStateAsync();
        state.Settings = CloneSettings(settings);
        await PersistAsync();
    }

    public AiModelStatus GetAiModelStatus(WorkspaceSettings settings)
    {
        return _aiNsfwClassifierService.GetStatus(settings.AiModel ?? new AiModelSettings());
    }

    public PersonalAiTrainingStatus GetPersonalAiTrainingStatus(WorkspaceSettings settings)
    {
        var taskMode = settings.TaskMode;
        var personalAiSettings = GetActivePersonalAiSettings(settings);
        var sampleFolders = GetActiveSampleFolders(settings)
            .Where(folder => folder.IsEnabled && Directory.Exists(folder.FolderPath))
            .ToList();
        personalAiSettings = MergeTrainingRootsFromSampleFolders(personalAiSettings, sampleFolders, taskMode);
        SetActivePersonalAiSettings(settings, personalAiSettings);

        var status = _personalAiTrainingService.GetStatus(personalAiSettings, taskMode);
        var summary = GetActivePersonalAiEvaluationSummary(_state, taskMode);
        if (summary is not null && summary.TotalSamples > 0)
        {
            status.AllowsPersonalModelPrimary = summary.AllowsPersonalModelPrimary;
            var primaryText = summary.AllowsPersonalModelPrimary
                ? "个人模型已通过评测，可作为主判"
                : "个人模型尚未通过主判评测";
            var primaryLabelText = GetLabelDisplayText(GetPrimaryLabel(taskMode));
            var secondaryLabelText = GetLabelDisplayText(GetSecondaryLabel(taskMode));
            status.Message += $" 最近评测：{summary.EvaluatedSamples}/{summary.TotalSamples} 张，准确率 {summary.Accuracy:P1}，{primaryLabelText} 误判为 {secondaryLabelText} {summary.SfwFalsePositiveRate:P1}，{secondaryLabelText} 误判为 {primaryLabelText} {summary.NsfwFalseNegativeRate:P1}，{primaryText}。";
        }

        return status;
    }

    public async Task<PersonalAiTrainingStatus> TrainPersonalAiAsync(WorkspaceSettings settings, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress = null)
    {
        var taskMode = settings.TaskMode;
        var sampleFolders = GetActiveSampleFolders(settings)
            .Where(folder => folder.IsEnabled && Directory.Exists(folder.FolderPath))
            .ToList();
        var activePersonalAiSettings = MergeTrainingRootsFromSampleFolders(GetActivePersonalAiSettings(settings), sampleFolders, taskMode);
        SetActivePersonalAiSettings(settings, activePersonalAiSettings);
        var state = await GetStateAsync().ConfigureAwait(false);
        state.Settings = CloneSettings(settings);
        await PersistAsync().ConfigureAwait(false);

        await BackfillManualCorrectionsIntoMistakeBookAsync(
                state,
                activePersonalAiSettings,
                cancellationToken,
                progress,
                taskMode)
            .ConfigureAwait(false);

        var status = await _personalAiTrainingService.TrainAsync(activePersonalAiSettings, cancellationToken, progress, taskMode).ConfigureAwait(false);
        await EvaluatePersonalAiCandidateAsync(settings, status, cancellationToken, progress).ConfigureAwait(false);
        var prunedModelCount = await _personalAiTrainingService.PruneModelVersionsAsync(activePersonalAiSettings, cancellationToken, taskMode).ConfigureAwait(false);
        if (prunedModelCount > 0)
        {
            var modelText = taskMode == ClassificationTaskMode.PersonPresence ? "人物模型" : "个人大模型";
            ReportProgress(
                progress,
                WorkspaceProgressStage.TrainingPersonalModel,
                $"已清理旧{modelText}",
                $"已按成功率保留最好的 2 个{modelText}版本，清理 {prunedModelCount} 个旧版本，为磁盘腾出空间。",
                $"{modelText}版本整理",
                1,
                1);
            await _performanceLogService.LogAsync("PersonalAiPrune", $"任务 {taskMode}，已按成功率保留最好的 2 个{modelText}版本，清理 {prunedModelCount} 个旧版本。").ConfigureAwait(false);
        }

        return GetPersonalAiTrainingStatus(settings);
    }

    internal async Task<int> BackfillManualCorrectionsIntoMistakeBookAsync(
        PersistedAppState state,
        PersonalAiModelSettings personalAiSettings,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress = null,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        if (!personalAiSettings.IsEnabled)
        {
            return 0;
        }

        var candidates = state.LearnedImagesById.Values
            .Where(record => ShouldBackfillManualCorrectionIntoMistakeBook(record, taskMode))
            .OrderBy(record => record.UpdatedAtUtc)
            .ToList();
        if (candidates.Count == 0)
        {
            return 0;
        }

        ReportProgress(
            progress,
            WorkspaceProgressStage.TrainingPersonalModel,
            "正在同步历史错题本",
            $"正在把历史人工改判补入轻量错题本，候选 {candidates.Count} 条。",
            string.Empty,
            0,
            candidates.Count);

        var backfilledCount = 0;
        for (var index = 0; index < candidates.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var record = candidates[index];
            if (_personalAiTrainingService.HasMistakeBookSample(
                    personalAiSettings,
                    record.LastKnownPath,
                    record.ContentId,
                    record.CurrentLabel,
                    taskMode))
            {
                continue;
            }

            await _personalAiTrainingService.RecordManualCorrectionSampleAsync(
                    personalAiSettings,
                    record.LastKnownPath,
                    record.ContentId,
                    record.CurrentLabel,
                    ImageLabel.Uncertain,
                    LabelOrigin.ManualCorrection,
                    cancellationToken,
                    correctionCount: Math.Max(1, record.CorrectionCount),
                    incrementExistingMistake: false,
                    taskMode: taskMode)
                .ConfigureAwait(false);
            backfilledCount++;

            if (ShouldReportProgress(index + 1, candidates.Count))
            {
                ReportProgress(
                    progress,
                    WorkspaceProgressStage.TrainingPersonalModel,
                    "正在同步历史错题本",
                    $"已补入 {backfilledCount} 条历史人工改判，已检查 {index + 1} / {candidates.Count} 条。",
                    Path.GetFileName(record.LastKnownPath),
                    index + 1,
                    candidates.Count);
            }
        }

        if (backfilledCount > 0)
        {
            ReportProgress(
                progress,
                WorkspaceProgressStage.TrainingPersonalModel,
                "历史错题本同步完成",
                $"已补入 {backfilledCount} 条历史人工改判，本次增量训练会优先使用这些轻量错题素材。",
                "错题本",
                candidates.Count,
                candidates.Count);
        }

        return backfilledCount;
    }

    public Task<PersonalAiTrainingStatus> ActivateCandidatePersonalAiModelAsync(WorkspaceSettings settings, CancellationToken cancellationToken)
    {
        return _personalAiTrainingService.ActivateCandidateModelAsync(GetActivePersonalAiSettings(settings), cancellationToken, settings.TaskMode);
    }

    public Task<PersonalAiTrainingStatus> ActivatePersonalAiModelAsync(
        WorkspaceSettings settings,
        string modelVersion,
        CancellationToken cancellationToken)
    {
        return _personalAiTrainingService.ActivateModelAsync(
            GetActivePersonalAiSettings(settings),
            modelVersion,
            cancellationToken,
            settings.TaskMode);
    }

    public Task DownloadAiModelAsync(WorkspaceSettings settings, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress = null)
    {
        return _aiNsfwClassifierService.DownloadDefaultModelAsync(settings.AiModel ?? new AiModelSettings(), cancellationToken, progress);
    }

    public async Task<SampleRefreshSummary> RefreshSampleLibraryAsync(WorkspaceSettings settings, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress = null)
    {
        var overallStopwatch = Stopwatch.StartNew();
        var state = await GetStateAsync();

        var summary = new SampleRefreshSummary();
        var sampleFolders = GetActiveSampleFolders(settings)
            .Where(folder => folder.IsEnabled && Directory.Exists(folder.FolderPath))
            .ToList();
        SetActivePersonalAiSettings(
            settings,
            MergeTrainingRootsFromSampleFolders(GetActivePersonalAiSettings(settings), sampleFolders, settings.TaskMode));

        state.Settings = CloneSettings(settings);

        var sampleFiles = sampleFolders
            .SelectMany(folder => EnumerateSupportedImages(folder.FolderPath).Select(filePath => (folder, filePath)))
            .ToList();

        var totalFiles = sampleFiles.Count;
        var scannedFiles = 0;
        var reusedCacheFiles = 0;
        var importedFiles = 0;
        var ignoredFiles = 0;
        var sampleAiCacheReuseCount = 0;
        var sampleAiInferenceCount = 0;
        var sampleAiSkippedCount = 0;
        var fingerprintElapsedMilliseconds = 0L;
        var sampleAiElapsedMilliseconds = 0L;
        var stateLock = new object();
        var aiFeatureRequestLock = new object();
        var aiFeatureRequests = new List<AiFeatureRequest>();
        var aiFeatureRequestIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var personalAiTrainingSamplesLock = new object();
        var personalAiTrainingSamples = new List<(string FilePath, string ContentId, ImageLabel Label, string Source)>();
        var scanParallelism = GetScanParallelism();
        var aiModelStatus = GetAiModelStatus(settings);
        var canRunSampleAiInference = settings.TaskMode == ClassificationTaskMode.ContentSafety && aiModelStatus.IsEnabled && aiModelStatus.IsInstalled;

        var sampleScanDetail = canRunSampleAiInference
            ? "正在检查样本缓存、新增图片，并为缺少新版 AI 特征的样本排队。"
            : "正在检查样本缓存与新增图片。";
        ReportProgress(progress, WorkspaceProgressStage.RefreshingSamples, "正在扫描样本库", sampleScanDetail, string.Empty, 0, totalFiles);

        await Parallel.ForEachAsync(sampleFiles, new ParallelOptions
        {
            CancellationToken = cancellationToken,
            MaxDegreeOfParallelism = scanParallelism,
        }, async (sampleFile, token) =>
        {
            var (sampleFolder, filePath) = sampleFile;
            var normalizedPath = NormalizePath(filePath);
            try
            {
                token.ThrowIfCancellationRequested();

                FileCacheRecord? cacheRecord = null;
                lock (stateLock)
                {
                    if (TryReuseFileCache(state, normalizedPath, out var reusedCacheRecord))
                    {
                        cacheRecord = reusedCacheRecord;
                    }
                }

                if (cacheRecord is null)
                {
                    var fingerprintStopwatch = Stopwatch.StartNew();
                    var fingerprint = await _imageFingerprintService.CreateFingerprintAsync(normalizedPath, token).ConfigureAwait(false);
                    fingerprintStopwatch.Stop();

                    Interlocked.Add(ref fingerprintElapsedMilliseconds, fingerprintStopwatch.ElapsedMilliseconds);
                    lock (stateLock)
                    {
                        cacheRecord = UpsertFileCacheRecord(state, normalizedPath, fingerprint);
                    }

                    Interlocked.Increment(ref importedFiles);
                }
                else
                {
                    Interlocked.Increment(ref reusedCacheFiles);
                }

                lock (stateLock)
                {
                    UpsertLearnedRecord(state, cacheRecord, sampleFolder.Label, LabelOrigin.SampleLibrary, "来自样本库导入", settings.TaskMode);
                }

                if (IsTaskLabel(sampleFolder.Label, settings.TaskMode))
                {
                    lock (personalAiTrainingSamplesLock)
                    {
                        personalAiTrainingSamples.Add((normalizedPath, cacheRecord.ContentId, sampleFolder.Label, "sample_library"));
                    }
                }

                if (canRunSampleAiInference)
                {
                    var cachedAiClassification = TryGetCachedAiClassification(state, cacheRecord, stateLock);
                    if (cachedAiClassification is not null)
                    {
                        Interlocked.Increment(ref sampleAiCacheReuseCount);
                    }
                    else
                    {
                        lock (aiFeatureRequestLock)
                        {
                            if (aiFeatureRequestIds.Add(cacheRecord.ContentId))
                            {
                                aiFeatureRequests.Add(new AiFeatureRequest(cacheRecord.ContentId, normalizedPath));
                            }
                        }
                    }
                }

            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                Interlocked.Increment(ref ignoredFiles);
            }

            var current = Interlocked.Increment(ref scannedFiles);
            if (ShouldReportProgress(current, totalFiles))
            {
                var detail = canRunSampleAiInference
                    ? $"已处理 {current} 张样本，复用图片缓存 {reusedCacheFiles} 张，新增 {importedFiles} 张，AI 特征复用 {sampleAiCacheReuseCount} 张。"
                    : $"已处理 {current} 张样本，复用缓存 {reusedCacheFiles} 张，新增 {importedFiles} 张。";
                ReportProgress(progress, WorkspaceProgressStage.RefreshingSamples, "正在扫描样本库", detail, Path.GetFileName(normalizedPath), current, totalFiles);
            }
        });

        if (personalAiTrainingSamples.Count > 0)
        {
            await _personalAiTrainingService.RecordTrainingSamplesAsync(
                GetActivePersonalAiSettings(settings),
                personalAiTrainingSamples,
                cancellationToken,
                settings.TaskMode).ConfigureAwait(false);
        }

        if (canRunSampleAiInference && aiFeatureRequests.Count > 0)
        {
            var sampleFeatureDetail = $"启用样本库共 {totalFiles} 张，已复用 AI 特征 {sampleAiCacheReuseCount} 张；还有 {aiFeatureRequests.Count} 张样本缺少新版 AI 特征，正在补齐缓存。";
            var aiFeatureSummary = await ExtractAiFeaturesInBatchesAsync(
                state,
                aiFeatureRequests,
                settings,
                stateLock,
                cancellationToken,
                progress,
                WorkspaceProgressStage.ExtractingAiFeatures,
                "正在提取样本 AI 特征",
                sampleFeatureDetail).ConfigureAwait(false);
            sampleAiInferenceCount = aiFeatureSummary.InferenceCount;
            sampleAiSkippedCount = aiFeatureSummary.SkippedCount;
            sampleAiElapsedMilliseconds = aiFeatureSummary.ElapsedMilliseconds;
        }

        summary.ScannedFiles = scannedFiles;
        summary.ReusedCacheFiles = reusedCacheFiles;
        summary.ImportedFiles = importedFiles;
        summary.IgnoredFiles = ignoredFiles;
        RefreshPersonalAiEvaluationSet(state, settings, sampleFolders);

        await PersistAsync();
        overallStopwatch.Stop();

        var completedDetail = canRunSampleAiInference
            ? $"共检查 {summary.ScannedFiles} 张，复用缓存 {summary.ReusedCacheFiles} 张，新增导入 {summary.ImportedFiles} 张；样本 AI 特征新增 {sampleAiInferenceCount} 张，复用 {sampleAiCacheReuseCount} 张。"
            : $"共检查 {summary.ScannedFiles} 张，复用缓存 {summary.ReusedCacheFiles} 张，新增导入 {summary.ImportedFiles} 张。";
        ReportProgress(progress, canRunSampleAiInference ? WorkspaceProgressStage.ExtractingAiFeatures : WorkspaceProgressStage.RefreshingSamples, "样本库扫描完成", completedDetail, string.Empty, 1, 1);
        var modelText = settings.TaskMode == ClassificationTaskMode.PersonPresence ? "人物模型" : "个人大模型";
        await _performanceLogService.LogAsync("SampleRefresh", $"任务 {settings.TaskMode}，总耗时 {overallStopwatch.Elapsed.TotalSeconds:F2}s，并发 {scanParallelism}，样本数 {summary.ScannedFiles}，复用缓存 {summary.ReusedCacheFiles}，新增指纹 {summary.ImportedFiles}，忽略 {summary.IgnoredFiles}，{modelText}训练样本同步 {personalAiTrainingSamples.Count}，指纹计算耗时 {fingerprintElapsedMilliseconds}ms，样本AI缓存 {sampleAiCacheReuseCount}，样本AI推理 {sampleAiInferenceCount}，样本AI跳过 {sampleAiSkippedCount}，样本AI耗时 {sampleAiElapsedMilliseconds}ms。");

        return summary;
    }

    public async Task<IReadOnlyList<AnalysisRecord>> AnalyzeAsync(WorkspaceSettings settings, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress = null)
    {
        var overallStopwatch = Stopwatch.StartNew();
        var state = await GetStateAsync();
        state.Settings = CloneSettings(settings);

        var sourceFolder = GetActiveSourceFolder(settings);
        if (string.IsNullOrWhiteSpace(sourceFolder) || !Directory.Exists(sourceFolder))
        {
            return [];
        }

        ReportProgress(
            progress,
            WorkspaceProgressStage.PreparingSource,
            "正在枚举源图片",
            "正在扫描源文件夹及其子文件夹，统计可分析的图片数量。",
            sourceFolder,
            0,
            1);
        var sourceFiles = EnumerateSupportedImages(sourceFolder).ToList();
        var totalFiles = sourceFiles.Count;
        ReportProgress(
            progress,
            WorkspaceProgressStage.PreparingSource,
            "源图片枚举完成",
            $"已找到 {totalFiles} 张可分析图片，开始加载样本库和人工纠错记录。",
            Path.GetFileName(sourceFolder),
            0,
            totalFiles);
        var results = new List<AnalysisRecord>(totalFiles);
        ReportProgress(
            progress,
            WorkspaceProgressStage.Loading,
            "正在加载学习记录",
            "正在整理样本库、人工纠错、AI 校准和个人模型状态。",
            string.Empty,
            0,
            1);
        var activePersonalAiSettings = GetActivePersonalAiSettings(settings);
        var primaryLabel = GetPrimaryLabel(settings.TaskMode);
        var secondaryLabel = GetSecondaryLabel(settings.TaskMode);
        var activeSampleFolders = GetActiveSampleFolders(settings)
            .Where(folder => folder.IsEnabled &&
                             IsTaskLabel(folder.Label, settings.TaskMode) &&
                             !string.IsNullOrWhiteSpace(folder.FolderPath))
            .ToList();
        var learnedIndex = BuildLearnedSampleIndex(state, settings.TaskMode, activePersonalAiSettings, activeSampleFolders);
        var aiCalibration = BuildPersonalAiCalibration(
            state,
            settings.AiModel ?? new AiModelSettings(),
            activePersonalAiSettings);
        var aiCorrectionProfile = BuildPersonalAiCorrectionProfile(state, activePersonalAiSettings);
        var personalAiModel = BuildPersonalAiModel(aiCorrectionProfile);
        var activeLearningSampleCount = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? learnedIndex.All.Count
            : aiCorrectionProfile.TrainingSampleCount;
        ReportProgress(
            progress,
            WorkspaceProgressStage.Loading,
            "学习记录加载完成",
            $"已整理 {learnedIndex.All.Count} 条历史学习记录，当前任务训练参考 {activeLearningSampleCount} 条。",
            activePersonalAiSettings.IsEnabled ? "个人模型已启用" : "个人模型未启用",
            1,
            1);

        var processedCount = 0;
        var cacheReuseCount = 0;
        var fingerprintCount = 0;
        var ignoredCount = 0;
        var aiCacheReuseCount = 0;
        var aiInferenceCount = 0;
        var aiSkippedCount = 0;
        var fingerprintElapsedMilliseconds = 0L;
        var predictionElapsedMilliseconds = 0L;
        var aiElapsedMilliseconds = 0L;
        var stateLock = new object();
        var resultsLock = new object();
        var scanParallelism = GetScanParallelism();
        var aiModelStatus = GetAiModelStatus(settings);
        var canRunAiInference = settings.TaskMode == ClassificationTaskMode.ContentSafety && aiModelStatus.IsEnabled && aiModelStatus.IsInstalled;

        var analysisDetail = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? "正在根据人物/非人物样本库和你的纠错记录进行判断。真人、真人元素、真人海报算人物；二次元、风景、玩偶雕像不算。"
            : "正在先用缓存、人工纠错和样本相似度预筛；缺少新版 AI 特征的图片会补充通用分类与 NudeNet 裸露检测。";
        ReportProgress(progress, WorkspaceProgressStage.PreparingSource, "正在准备源图片", analysisDetail, string.Empty, 0, totalFiles);

        var sourceItems = new List<SourceAnalysisInput>(totalFiles);
        var sourceItemsLock = new object();
        var preparedCount = 0;
        await Parallel.ForEachAsync(sourceFiles, new ParallelOptions
        {
            CancellationToken = cancellationToken,
            MaxDegreeOfParallelism = scanParallelism,
        }, async (filePath, token) =>
        {
            var normalizedPath = NormalizePath(filePath);
            var hasFailed = false;
            try
            {
                token.ThrowIfCancellationRequested();

                FileCacheRecord? cacheRecord = null;
                lock (stateLock)
                {
                    if (TryReuseFileCache(state, normalizedPath, out var existingCacheRecord))
                    {
                        cacheRecord = existingCacheRecord;
                    }
                }

                if (cacheRecord is null)
                {
                    var fingerprintStopwatch = Stopwatch.StartNew();
                    var fingerprint = await _imageFingerprintService.CreateFingerprintAsync(normalizedPath, token).ConfigureAwait(false);
                    fingerprintStopwatch.Stop();

                    Interlocked.Add(ref fingerprintElapsedMilliseconds, fingerprintStopwatch.ElapsedMilliseconds);
                    lock (stateLock)
                    {
                        cacheRecord = UpsertFileCacheRecord(state, normalizedPath, fingerprint);
                    }

                    Interlocked.Increment(ref fingerprintCount);
                }
                else
                {
                    Interlocked.Increment(ref cacheReuseCount);
                }

                lock (sourceItemsLock)
                {
                    sourceItems.Add(new SourceAnalysisInput(normalizedPath, cacheRecord));
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                hasFailed = true;
                Interlocked.Increment(ref ignoredCount);
            }

            var current = Interlocked.Increment(ref preparedCount);
            if (ShouldReportProgress(current, totalFiles))
            {
                var detail = hasFailed
                    ? "正在准备源图片缓存，部分损坏图片已自动跳过。"
                    : $"正在准备源图片缓存，复用图片缓存 {cacheReuseCount} 张，新增指纹 {fingerprintCount} 张。";
                ReportProgress(progress, WorkspaceProgressStage.PreparingSource, "正在准备源图片", detail, Path.GetFileName(normalizedPath), current, totalFiles);
            }
        });

        if (canRunAiInference && sourceItems.Count > 0)
        {
            ReportProgress(
                progress,
                WorkspaceProgressStage.ExtractingAiFeatures,
                "正在检查源图 AI 特征缓存",
                $"正在检查 {sourceItems.Count} 张源图是否已有新版 AI 特征缓存。",
                string.Empty,
                0,
                sourceItems.Count);
            var sourceAiFeatureRequests = new List<AiFeatureRequest>();
            var requestedContentIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var checkedAiCacheCount = 0;
            foreach (var sourceItem in sourceItems)
            {
                checkedAiCacheCount++;
                var aiClassification = TryGetCachedAiClassification(state, sourceItem.CacheRecord, stateLock);
                if (aiClassification is not null)
                {
                    aiCacheReuseCount++;
                }
                else if (!string.IsNullOrWhiteSpace(sourceItem.CacheRecord.ContentId) &&
                         requestedContentIds.Add(sourceItem.CacheRecord.ContentId))
                {
                    sourceAiFeatureRequests.Add(new AiFeatureRequest(sourceItem.CacheRecord.ContentId, sourceItem.FilePath));
                }

                if (ShouldReportProgress(checkedAiCacheCount, sourceItems.Count))
                {
                    ReportProgress(
                        progress,
                        WorkspaceProgressStage.ExtractingAiFeatures,
                        "正在检查源图 AI 特征缓存",
                        $"已检查 {checkedAiCacheCount} / {sourceItems.Count} 张，复用 AI 缓存 {aiCacheReuseCount} 张，待新增复核 {sourceAiFeatureRequests.Count} 张。",
                        Path.GetFileName(sourceItem.FilePath),
                        checkedAiCacheCount,
                        sourceItems.Count);
                }
            }

            if (sourceAiFeatureRequests.Count > 0)
            {
                var sourceFeatureDetail = $"源文件夹共 {sourceItems.Count} 张，已复用 AI 特征 {aiCacheReuseCount} 张；还有 {sourceAiFeatureRequests.Count} 张源图缺少新版 AI 特征，正在补齐缓存。";
                var aiFeatureSummary = await ExtractAiFeaturesInBatchesAsync(
                    state,
                    sourceAiFeatureRequests,
                    settings,
                    stateLock,
                    cancellationToken,
                    progress,
                    WorkspaceProgressStage.ExtractingAiFeatures,
                    "正在提取源图 AI 特征",
                    sourceFeatureDetail).ConfigureAwait(false);
                aiInferenceCount = aiFeatureSummary.InferenceCount;
                aiElapsedMilliseconds = aiFeatureSummary.ElapsedMilliseconds;
            }
        }

        IReadOnlyDictionary<string, PersonalAiPredictionResult> personalAiPredictions;
        if (activePersonalAiSettings.IsEnabled)
        {
            var modelText = settings.TaskMode == ClassificationTaskMode.PersonPresence ? "人物个人模型" : "个人大模型";
            var currentItemText = settings.TaskMode == ClassificationTaskMode.PersonPresence ? "人物 LoRA 主判" : "个人 LoRA 主判";
            ReportProgress(
                progress,
                WorkspaceProgressStage.AnalyzingSource,
                $"正在执行{modelText}预测",
                $"正在用当前启用的{modelText}批量预测 {sourceItems.Count} 张源图。",
                currentItemText,
                0,
                sourceItems.Count);
            personalAiPredictions = await _personalAiTrainingService
                .PredictBatchAsync(
                    activePersonalAiSettings,
                    sourceItems.Select(item => (item.CacheRecord.ContentId, item.FilePath)).ToList(),
                    cancellationToken,
                    progress,
                    settings.TaskMode)
                .ConfigureAwait(false);
            ReportProgress(
                progress,
                WorkspaceProgressStage.AnalyzingSource,
                $"{modelText}预测完成",
                $"{modelText}返回 {personalAiPredictions.Count} 条预测，开始合并样本库、AI 特征和人工经验。",
                currentItemText,
                sourceItems.Count,
                sourceItems.Count);
        }
        else
        {
            personalAiPredictions = new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }

        processedCount = 0;
        await Parallel.ForEachAsync(sourceItems, new ParallelOptions
        {
            CancellationToken = cancellationToken,
            MaxDegreeOfParallelism = scanParallelism,
        }, (sourceItem, token) =>
        {
            token.ThrowIfCancellationRequested();
            var normalizedPath = sourceItem.FilePath;
            var cacheRecord = sourceItem.CacheRecord;

            try
            {
                AiClassificationResult? aiClassification = null;
                if (settings.TaskMode == ClassificationTaskMode.ContentSafety && settings.AiModel?.IsEnabled == true)
                {
                    aiClassification = TryGetCachedAiClassification(state, cacheRecord, stateLock);
                    if (aiClassification is null && canRunAiInference)
                    {
                        Interlocked.Increment(ref aiSkippedCount);
                    }
                }

                personalAiPredictions.TryGetValue(cacheRecord.ContentId, out var personalAiPrediction);
                var predictionStopwatch = Stopwatch.StartNew();
                var prediction = settings.TaskMode == ClassificationTaskMode.PersonPresence
                    ? PredictBySampleTask(cacheRecord.ContentId, cacheRecord.AverageHashBits, cacheRecord.Width, cacheRecord.Height, learnedIndex, ImageLabel.Person, ImageLabel.NonPerson, "人物", "非人物", personalAiPrediction, activePersonalAiSettings)
                    : Predict(cacheRecord.ContentId, cacheRecord.AverageHashBits, cacheRecord.Width, cacheRecord.Height, learnedIndex, aiClassification, personalAiPrediction, aiCalibration, aiCorrectionProfile, personalAiModel, activePersonalAiSettings, state.PersonalAiEvaluationSummary);
                predictionStopwatch.Stop();
                Interlocked.Add(ref predictionElapsedMilliseconds, predictionStopwatch.ElapsedMilliseconds);

                var analysisRecord = new AnalysisRecord
                {
                    FilePath = normalizedPath,
                    FileName = Path.GetFileName(normalizedPath),
                    ContentId = cacheRecord.ContentId,
                    AverageHash = cacheRecord.AverageHash,
                    AverageHashBits = cacheRecord.AverageHashBits,
                    Width = cacheRecord.Width,
                    Height = cacheRecord.Height,
                    TaskMode = settings.TaskMode,
                    SuggestedLabel = prediction.SuggestedLabel,
                    FinalLabel = prediction.FinalLabel,
                    LabelOrigin = prediction.LabelOrigin,
                    Confidence = prediction.Confidence,
                    IsManualCorrection = prediction.IsManualCorrection,
                    IsHumanConfirmed = prediction.IsHumanConfirmed,
                    Explanation = prediction.Explanation,
                    UsedHistoryCache = prediction.UsedHistoryCache,
                    UsedAiModel = prediction.UsedAiModel,
                    UsedPersonalLearning = prediction.UsedPersonalLearning,
                    UsedPersonalLora = prediction.UsedPersonalLora,
                    GenericNsfwScore = prediction.GenericNsfwScore,
                    PersonalNsfwScore = prediction.PersonalNsfwScore,
                };

                lock (resultsLock)
                {
                    results.Add(analysisRecord);
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                Interlocked.Increment(ref ignoredCount);
            }

            var current = Interlocked.Increment(ref processedCount);
            if (ShouldReportProgress(current, sourceItems.Count))
            {
                int resultCount;
                lock (resultsLock)
                {
                    resultCount = results.Count;
                }

                var detail = settings.TaskMode == ClassificationTaskMode.PersonPresence
                    ? $"已生成 {resultCount} 条人物分类结果，正在参考样本库和纠错记录。"
                    : $"已生成 {resultCount} 条结果，AI 新增复核 {aiInferenceCount} 张，复用 AI 缓存 {aiCacheReuseCount} 张。";
                ReportProgress(progress, WorkspaceProgressStage.AnalyzingSource, "正在生成扫描结果", detail, Path.GetFileName(normalizedPath), current, sourceItems.Count);
            }

            return ValueTask.CompletedTask;
        });

        await PersistAsync();
        overallStopwatch.Stop();

        var primaryResultCount = results.Count(item => item.FinalLabel == primaryLabel);
        var secondaryResultCount = results.Count(item => item.FinalLabel == secondaryLabel);
        var originSummary = string.Join("，", results
            .GroupBy(item => item.LabelOrigin)
            .OrderByDescending(group => group.Count())
            .Select(group => $"{group.Key}:{group.Count()}"));

        ReportProgress(progress, WorkspaceProgressStage.Completed, "源文件夹分析完成", $"已完成 {results.Count} 条结果生成，可以开始二次筛查。", string.Empty, totalFiles, totalFiles);
        var calibrationStatus = aiCalibration.IsPersonalized ? "已启用" : "未启用";
        var taskText = settings.TaskMode == ClassificationTaskMode.PersonPresence ? "人物非人物" : "内容安全";
        var primaryText = GetLabelDisplayText(primaryLabel);
        var secondaryText = GetLabelDisplayText(secondaryLabel);
        var learningPrimaryCount = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? learnedIndex.All.Count(item => item.CurrentLabel == primaryLabel)
            : aiCorrectionProfile.SfwCount;
        var learningSecondaryCount = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? learnedIndex.All.Count(item => item.CurrentLabel == secondaryLabel)
            : aiCorrectionProfile.NsfwCount;
        var learningManualCount = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? learnedIndex.All.Count(item => item.LabelOrigin == LabelOrigin.ManualCorrection)
            : aiCorrectionProfile.ManualCorrectionCount;
        var learningSampleLibraryCount = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? learnedIndex.All.Count(item => item.LabelOrigin == LabelOrigin.SampleLibrary)
            : aiCorrectionProfile.SampleLibraryCount;
        var modelEnabledText = settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? (activePersonalAiSettings.IsEnabled ? "已启用" : "未启用")
            : (personalAiModel.IsAvailable ? "已启用" : "未启用");
        await _performanceLogService.LogAsync("SourceAnalyze", $"任务 {taskText}，总耗时 {overallStopwatch.Elapsed.TotalSeconds:F2}s，并发 {scanParallelism}，源图 {totalFiles}，结果 {results.Count}，{primaryText}:{primaryResultCount}，{secondaryText}:{secondaryResultCount}，来源 {originSummary}，缓存命中 {cacheReuseCount}，新增指纹 {fingerprintCount}，跳过 {ignoredCount}，指纹计算耗时 {fingerprintElapsedMilliseconds}ms，预测耗时 {predictionElapsedMilliseconds}ms，AI缓存 {aiCacheReuseCount}，AI推理 {aiInferenceCount}，AI跳过 {aiSkippedCount}，AI耗时 {aiElapsedMilliseconds}ms，已学习样本 {learnedIndex.All.Count}，当前任务训练参考 {activeLearningSampleCount}（{primaryText} {learningPrimaryCount}，{secondaryText} {learningSecondaryCount}，人工 {learningManualCount}，样本库 {learningSampleLibraryCount}），个人AI模型 {modelEnabledText} 阈值 {personalAiModel.DecisionThreshold:P0}，个人阈值 {calibrationStatus} {aiCalibration.DecisionThreshold:P0}。");

        return results
            .OrderBy(item => item.IsHumanConfirmed)
            .ThenByDescending(item => item.FinalLabel == ImageLabel.Uncertain)
            .ThenBy(item => item.Confidence)
            .ThenBy(item => item.FileName, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public async Task ApplyManualCorrectionAsync(AnalysisRecord record, ImageLabel correctedLabel, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var state = await GetStateAsync();
        var cacheRecord = ResolveFileCache(state, record.FilePath, record.ContentId, record.AverageHash, record.AverageHashBits, record.Width, record.Height);

        await EnsureAiClassificationForCorrectionAsync(state, cacheRecord, record.FilePath, record.TaskMode, cancellationToken).ConfigureAwait(false);
        UpsertLearnedRecord(state, cacheRecord, correctedLabel, LabelOrigin.ManualCorrection, "来自人工纠错", record.TaskMode, incrementCorrectionCount: true);
        if (IsTaskLabel(correctedLabel, record.TaskMode))
        {
            await _personalAiTrainingService.RecordManualCorrectionSampleAsync(
                GetPersonalAiSettings(state.Settings, record.TaskMode),
                cacheRecord.FilePath,
                cacheRecord.ContentId,
                correctedLabel,
                record.FinalLabel,
                record.LabelOrigin,
                cancellationToken,
                taskMode: record.TaskMode).ConfigureAwait(false);
        }

        await PersistAsync();
    }

    public async Task ConfirmResultsAsync(IEnumerable<AnalysisRecord> records, WorkspaceSettings settings, CancellationToken cancellationToken)
    {
        var state = await GetStateAsync();
        var primaryLabel = GetPrimaryLabel(settings.TaskMode);
        var secondaryLabel = GetSecondaryLabel(settings.TaskMode);

        foreach (var record in records)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!File.Exists(record.FilePath))
            {
                continue;
            }

            if (record.FinalLabel != primaryLabel && record.FinalLabel != secondaryLabel)
            {
                continue;
            }

            var cacheRecord = ResolveFileCache(state, record.FilePath, record.ContentId, record.AverageHash, record.AverageHashBits, record.Width, record.Height);
            UpsertLearnedRecord(
                state,
                cacheRecord,
                record.FinalLabel,
                record.IsManualCorrection ? LabelOrigin.ManualCorrection : LabelOrigin.ConfirmedMove,
                record.IsManualCorrection ? "人工纠正后确认本批次结果" : "人工确认本批次结果",
                settings.TaskMode);

            if (IsTaskLabel(record.FinalLabel, settings.TaskMode))
            {
                await _personalAiTrainingService.RecordTrainingSampleAsync(
                    GetActivePersonalAiSettings(settings),
                    cacheRecord.FilePath,
                    cacheRecord.ContentId,
                    record.FinalLabel,
                    record.IsManualCorrection ? "manual_confirmed_batch" : "confirmed_batch",
                    cancellationToken,
                    settings.TaskMode).ConfigureAwait(false);
            }
        }

        await PersistAsync();
    }

    public async Task<ConfirmedMoveSummary> MoveConfirmedAsync(
        IEnumerable<AnalysisRecord> records,
        WorkspaceSettings settings,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress = null)
    {
        var state = await GetStateAsync();
        var summary = new ConfirmedMoveSummary();
        var primaryLabel = GetPrimaryLabel(settings.TaskMode);
        var secondaryLabel = GetSecondaryLabel(settings.TaskMode);
        var movableRecords = records.Where(item => item.FinalLabel == primaryLabel || item.FinalLabel == secondaryLabel).ToList();
        var totalCount = movableRecords.Count;
        var processedCount = 0;
        var movedCount = 0;
        var renamedCount = 0;
        var missingCount = 0;

        foreach (var record in movableRecords)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!File.Exists(record.FilePath))
            {
                summary.Results.Add(new FileMoveResult
                {
                    SourcePath = record.FilePath,
                    Status = FileMoveStatus.SourceMissing,
                    Message = "源文件不存在，已跳过。",
                });
                processedCount++;
                missingCount++;
                ReportMoveProgress(progress, processedCount, totalCount, movedCount, renamedCount, missingCount, record.FileName);
                continue;
            }

            var destinationFolder = record.FinalLabel == primaryLabel
                ? GetActivePrimaryTargetFolder(settings)
                : GetActiveSecondaryTargetFolder(settings);
            if (string.IsNullOrWhiteSpace(destinationFolder))
            {
                continue;
            }

            FileMoveResult moveResult;
            try
            {
                moveResult = await _fileMoveService.MoveFileAsync(record.FilePath, destinationFolder, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (exception is not OperationCanceledException)
            {
                throw new IOException($"移动 {record.FileName} 时失败：{exception.Message}", exception);
            }

            summary.Results.Add(moveResult);
            processedCount++;
            switch (moveResult.Status)
            {
                case FileMoveStatus.Moved:
                    movedCount++;
                    break;
                case FileMoveStatus.RenamedDueToConflict:
                    renamedCount++;
                    break;
                case FileMoveStatus.SourceMissing:
                    missingCount++;
                    break;
            }
            ReportMoveProgress(progress, processedCount, totalCount, movedCount, renamedCount, missingCount, Path.GetFileName(moveResult.DestinationPath));
            if (moveResult.Status != FileMoveStatus.Moved)
            {
                if (moveResult.Status != FileMoveStatus.RenamedDueToConflict)
                {
                    continue;
                }
            }

            var targetPath = moveResult.DestinationPath;
            if (state.FileCacheByPath.Remove(record.FilePath, out var existingCache))
            {
                existingCache.FilePath = NormalizePath(targetPath);
                existingCache.LastWriteUtc = File.GetLastWriteTimeUtc(targetPath);
                existingCache.LastSeenUtc = DateTime.UtcNow;
                state.FileCacheByPath[existingCache.FilePath] = existingCache;

                UpsertLearnedRecord(state, existingCache, record.FinalLabel, record.IsManualCorrection ? LabelOrigin.ManualCorrection : LabelOrigin.ConfirmedMove, "在结果复核后确认移动", settings.TaskMode);
                if (IsTaskLabel(record.FinalLabel, settings.TaskMode))
                {
                    await _personalAiTrainingService.RecordTrainingSampleAsync(
                        GetActivePersonalAiSettings(settings),
                        existingCache.FilePath,
                        existingCache.ContentId,
                        record.FinalLabel,
                        record.IsManualCorrection ? "manual_confirmed_move" : "confirmed_move",
                        cancellationToken,
                        settings.TaskMode).ConfigureAwait(false);
                }
            }
        }

        await PersistAsync();
        return summary;
    }

    private async Task EvaluatePersonalAiCandidateAsync(
        WorkspaceSettings settings,
        PersonalAiTrainingStatus trainingStatus,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress)
    {
        var state = await GetStateAsync().ConfigureAwait(false);
        var taskMode = settings.TaskMode;
        var evaluationStore = GetActivePersonalAiEvaluationSamples(state, taskMode);
        var primaryLabel = GetPrimaryLabel(taskMode);
        var secondaryLabel = GetSecondaryLabel(taskMode);
        var primaryText = GetLabelDisplayText(primaryLabel);
        var secondaryText = GetLabelDisplayText(secondaryLabel);
        var modelText = taskMode == ClassificationTaskMode.PersonPresence ? "人物个人模型" : "个人大模型";
        var evaluationSamples = evaluationStore.Values
            .Where(sample => IsTaskLabel(sample.Label, taskMode) &&
                             !string.IsNullOrWhiteSpace(sample.ContentId) &&
                             File.Exists(sample.FilePath))
            .OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (evaluationSamples.Count == 0)
        {
            return;
        }

        ReportProgress(
            progress,
            WorkspaceProgressStage.TrainingPersonalModel,
            $"正在评测{modelText}",
            $"正在用固定评测集检查候选模型表现，本次评测 {evaluationSamples.Count} 张，不会复制原图。",
            "固定评测集",
            0,
            evaluationSamples.Count);

        var predictions = await _personalAiTrainingService.PredictCandidateBatchAsync(
                GetActivePersonalAiSettings(settings),
                evaluationSamples.Select(sample => (sample.ContentId, sample.FilePath)).ToList(),
                cancellationToken,
                progress,
                taskMode)
            .ConfigureAwait(false);

        var report = BuildPersonalAiEvaluationReport(
            string.IsNullOrWhiteSpace(trainingStatus.CandidateModelVersion)
                ? trainingStatus.ActiveModelVersion
                : trainingStatus.CandidateModelVersion,
            evaluationSamples,
            predictions,
            GetActivePersonalAiSettings(settings),
            taskMode);

        SetActivePersonalAiEvaluationReport(state, taskMode, report);
        await PersistAsync().ConfigureAwait(false);
        await _personalAiTrainingService.UpdateModelEvaluationSummaryAsync(
                GetActivePersonalAiSettings(settings),
                report.Summary,
                cancellationToken,
                taskMode)
            .ConfigureAwait(false);

        var resultText = report.Summary.AllowsPersonalModelPrimary
            ? "评测达标，个人模型后续可作为主判。"
            : "评测尚未达标，个人模型继续作为辅助判断。";
        ReportProgress(
            progress,
            WorkspaceProgressStage.TrainingPersonalModel,
            $"{modelText}评测完成",
            $"固定评测集 {report.Summary.EvaluatedSamples}/{report.Summary.TotalSamples} 张，准确率 {report.Summary.Accuracy:P1}，{primaryText} 误判为 {secondaryText} {report.Summary.SfwFalsePositiveRate:P1}，{secondaryText} 误判为 {primaryText} {report.Summary.NsfwFalseNegativeRate:P1}，不确定 {report.Summary.UncertainRate:P1}。{resultText}",
            report.Summary.ModelVersion,
            report.Summary.EvaluatedSamples,
            Math.Max(1, report.Summary.TotalSamples));
        await _performanceLogService.LogAsync(
                "PersonalAiEvaluate",
                $"任务 {taskMode}，模型 {report.Summary.ModelVersion}，评测 {report.Summary.EvaluatedSamples}/{report.Summary.TotalSamples}，准确率 {report.Summary.Accuracy:P2}，{primaryText} 误判为 {secondaryText} {report.Summary.SfwFalsePositiveRate:P2}，{secondaryText} 误判为 {primaryText} {report.Summary.NsfwFalseNegativeRate:P2}，不确定 {report.Summary.UncertainRate:P2}。{resultText}")
            .ConfigureAwait(false);
    }

    internal static PersonalAiEvaluationReport BuildPersonalAiEvaluationReport(
        string modelVersion,
        IReadOnlyList<PersonalAiEvaluationSample> samples,
        IReadOnlyDictionary<string, PersonalAiPredictionResult> predictions,
        PersonalAiModelSettings settings,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var primaryLabel = GetPrimaryLabel(taskMode);
        var secondaryLabel = GetSecondaryLabel(taskMode);
        var results = new List<PersonalAiEvaluationSampleResult>(samples.Count);
        var decisionThreshold = settings.DecisionThreshold <= 0d
            ? 0.5d
            : Math.Clamp(settings.DecisionThreshold, 0.01d, 0.99d);
        var minimumConfidence = settings.MinimumConfidence <= 0d
            ? 0.56d
            : Math.Clamp(settings.MinimumConfidence, 0.01d, 0.99d);
        foreach (var sample in samples)
        {
            predictions.TryGetValue(sample.ContentId, out var prediction);
            var isAvailable = prediction is { IsAvailable: true };
            var predictedLabel = ImageLabel.Uncertain;
            var confidence = prediction?.Confidence ?? 0d;
            var nsfwProbability = prediction?.NsfwProbability ?? 0d;
            if (isAvailable && confidence >= minimumConfidence)
            {
                predictedLabel = nsfwProbability >= decisionThreshold ? secondaryLabel : primaryLabel;
            }

            results.Add(new PersonalAiEvaluationSampleResult
            {
                ContentId = sample.ContentId,
                FilePath = sample.FilePath,
                ExpectedLabel = sample.Label,
                PredictedLabel = predictedLabel,
                NsfwProbability = nsfwProbability,
                Confidence = confidence,
                IsAvailable = isAvailable,
                Message = prediction?.Message ?? string.Empty,
            });
        }

        var evaluatedSamples = results.Count(item => item.IsAvailable);
        var sfwSamples = samples.Count(item => item.Label == primaryLabel);
        var nsfwSamples = samples.Count(item => item.Label == secondaryLabel);
        var correctCount = results.Count(item => item.PredictedLabel == item.ExpectedLabel);
        var sfwFalsePositiveCount = results.Count(item => item.ExpectedLabel == primaryLabel && item.PredictedLabel == secondaryLabel);
        var nsfwFalseNegativeCount = results.Count(item => item.ExpectedLabel == secondaryLabel && item.PredictedLabel == primaryLabel);
        var uncertainCount = results.Count(item => item.PredictedLabel == ImageLabel.Uncertain);
        var accuracy = evaluatedSamples <= 0 ? 0d : correctCount / (double)evaluatedSamples;
        var sfwFalsePositiveRate = sfwSamples <= 0 ? 1d : sfwFalsePositiveCount / (double)sfwSamples;
        var nsfwFalseNegativeRate = nsfwSamples <= 0 ? 1d : nsfwFalseNegativeCount / (double)nsfwSamples;
        var uncertainRate = samples.Count <= 0 ? 1d : uncertainCount / (double)samples.Count;
        var minimumSamplesPerLabel = Math.Min(50, Math.Max(10, (settings.EvaluationSamplesPerLabel <= 0 ? 500 : settings.EvaluationSamplesPerLabel) / 5));
        var minimumAccuracy = settings.PrimaryModelMinimumAccuracy <= 0d
            ? 0.92d
            : Math.Clamp(settings.PrimaryModelMinimumAccuracy, 0.5d, 0.995d);
        var maxSfwFalsePositiveRate = settings.PrimaryModelMaximumSfwFalsePositiveRate <= 0d
            ? 0.08d
            : Math.Clamp(settings.PrimaryModelMaximumSfwFalsePositiveRate, 0d, 0.5d);
        var maxNsfwFalseNegativeRate = settings.PrimaryModelMaximumNsfwFalseNegativeRate <= 0d
            ? 0.04d
            : Math.Clamp(settings.PrimaryModelMaximumNsfwFalseNegativeRate, 0d, 0.5d);

        var summary = new PersonalAiEvaluationSummary
        {
            ModelVersion = modelVersion,
            TotalSamples = samples.Count,
            EvaluatedSamples = evaluatedSamples,
            CorrectCount = correctCount,
            SfwSamples = sfwSamples,
            NsfwSamples = nsfwSamples,
            SfwFalsePositiveCount = sfwFalsePositiveCount,
            NsfwFalseNegativeCount = nsfwFalseNegativeCount,
            UncertainCount = uncertainCount,
            Accuracy = accuracy,
            SfwFalsePositiveRate = sfwFalsePositiveRate,
            NsfwFalseNegativeRate = nsfwFalseNegativeRate,
            UncertainRate = uncertainRate,
            AllowsPersonalModelPrimary = evaluatedSamples == samples.Count &&
                                         sfwSamples >= minimumSamplesPerLabel &&
                                         nsfwSamples >= minimumSamplesPerLabel &&
                                         accuracy >= minimumAccuracy &&
                                         sfwFalsePositiveRate <= maxSfwFalsePositiveRate &&
                                         nsfwFalseNegativeRate <= maxNsfwFalseNegativeRate,
            UpdatedAtUtc = DateTime.UtcNow,
        };

        return new PersonalAiEvaluationReport
        {
            Summary = summary,
            Results = results,
        };
    }

    private static void ReportMoveProgress(
        IProgress<WorkspaceProgressInfo>? progress,
        int current,
        int total,
        int movedCount,
        int renamedCount,
        int missingCount,
        string currentFileName)
    {
        if (progress is null || total <= 0)
        {
            return;
        }

        if (current != total && current % 100 != 0)
        {
            return;
        }

        progress.Report(new WorkspaceProgressInfo
        {
            Stage = WorkspaceProgressStage.MovingFiles,
            Title = "正在移动文件",
            Detail = $"已处理 {current} / {total}，成功 {movedCount}，重名改名 {renamedCount}，缺失跳过 {missingCount}。",
            CurrentItem = currentFileName,
            Current = current,
            Total = total,
        });
    }

    private PredictionResult Predict(
        string contentId,
        ulong averageHashBits,
        int width,
        int height,
        LearnedSampleIndex learnedIndex,
        AiClassificationResult? aiClassification,
        PersonalAiPredictionResult? personalAiPrediction,
        AiPersonalCalibration aiCalibration,
        AiCorrectionProfile aiCorrectionProfile,
        PersonalAiModel personalAiModel,
        PersonalAiModelSettings personalAiSettings,
        PersonalAiEvaluationSummary? aiEvaluationSummary)
    {
        if (learnedIndex.ByContentId.TryGetValue(contentId, out var exactMatch) && exactMatch.CurrentLabel is ImageLabel.Sfw or ImageLabel.Nsfw)
        {
            return new PredictionResult
            {
                SuggestedLabel = exactMatch.CurrentLabel,
                FinalLabel = exactMatch.CurrentLabel,
                LabelOrigin = exactMatch.LabelOrigin,
                Confidence = 1d,
                Explanation = "命中已学习图片缓存，直接沿用历史判断。",
                IsManualCorrection = exactMatch.LabelOrigin == LabelOrigin.ManualCorrection,
                IsHumanConfirmed = exactMatch.IsHumanConfirmed || exactMatch.LabelOrigin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove,
                UsedHistoryCache = true,
            };
        }

        var visualKey = BuildVisualKey(averageHashBits, width, height);
        if (learnedIndex.ByVisualKey.TryGetValue(visualKey, out var visualMatch) && visualMatch.CurrentLabel is ImageLabel.Sfw or ImageLabel.Nsfw)
        {
            return new PredictionResult
            {
                SuggestedLabel = visualMatch.CurrentLabel,
                FinalLabel = visualMatch.CurrentLabel,
                LabelOrigin = visualMatch.LabelOrigin,
                Confidence = 0.98d,
                Explanation = "命中同尺寸同指纹图片，直接沿用已学习结果。",
                IsManualCorrection = visualMatch.LabelOrigin == LabelOrigin.ManualCorrection,
                IsHumanConfirmed = visualMatch.IsHumanConfirmed || visualMatch.LabelOrigin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove,
                UsedHistoryCache = true,
            };
        }

        var nearestManualCorrection = FindNearestManualCorrection(learnedIndex, averageHashBits, ManualDuplicateDistanceThreshold);
        if (nearestManualCorrection.HasValue && nearestManualCorrection.Value.Sample.CurrentLabel is ImageLabel.Sfw or ImageLabel.Nsfw)
        {
            var (sample, distance) = nearestManualCorrection.Value;
            var manualConfidence = Math.Clamp(1d - (distance / 16d), 0.9d, 0.99d);
            return new PredictionResult
            {
                SuggestedLabel = sample.CurrentLabel,
                FinalLabel = sample.CurrentLabel,
                LabelOrigin = LabelOrigin.ManualCorrection,
                Confidence = manualConfidence,
                Explanation = $"命中近重复人工纠正图，相似距离 {distance}，沿用你的手动判断。普通相似图不会再被人工纠正硬覆盖。",
                IsHumanConfirmed = true,
                UsedHistoryCache = true,
            };
        }

        if (aiClassification is { IsAvailable: true })
        {
            var sampleSignal = learnedIndex.All.Count == 0
                ? SampleVoteSignal.Empty
                : CalculateSampleVoteSignal(learnedIndex, averageHashBits, 20);
            var sampleTotalScore = sampleSignal.SfwScore + sampleSignal.NsfwScore;
            var sampleNsfwProbability = sampleTotalScore <= 0d ? 0.5d : sampleSignal.NsfwScore / sampleTotalScore;
            var sampleCloseness = Math.Clamp(1d - (sampleSignal.ClosestDistance / 24d), 0d, 1d);
            var sampleWeight = sampleTotalScore <= 0d ? 0d : Math.Clamp(0.18d + (sampleCloseness * 0.44d), 0.18d, 0.62d);
            var explicitNsfwScore = GetAiExplicitNsfwScore(aiClassification);
            var suggestiveScore = GetAiScore(aiClassification, "sexy");
            var aiWeight = explicitNsfwScore >= 0.5d ? 1d - sampleWeight : Math.Min(0.34d, 1d - sampleWeight);
            var personalizedNsfwScore = (aiClassification.NsfwScore * aiWeight) + (sampleNsfwProbability * (1d - aiWeight));
            var hasPersonalLoraPrediction = personalAiPrediction is { IsAvailable: true } &&
                                            personalAiPrediction.Confidence >= personalAiSettings.MinimumConfidence;
            var personalModelCanLead = IsPersonalModelPrimaryEligible(aiEvaluationSummary, personalAiSettings);
            if (hasPersonalLoraPrediction)
            {
                var loraWeight = personalModelCanLead && personalAiPrediction!.Confidence >= Math.Max(0.72d, personalAiSettings.MinimumConfidence)
                    ? Math.Clamp(0.82d + (personalAiPrediction.Confidence * 0.14d), 0.82d, 0.96d)
                    : Math.Clamp(0.48d + (personalAiPrediction!.Confidence * 0.34d), 0.48d, 0.82d);
                personalizedNsfwScore = (personalizedNsfwScore * (1d - loraWeight)) + (personalAiPrediction.NsfwProbability * loraWeight);
            }

            var neighborAiSignal = MatchPersonalAiCorrection(aiCorrectionProfile, aiClassification);
            var modelAiSignal = PredictPersonalAiModel(personalAiModel, aiClassification);
            var personalAiSignal = MergePersonalAiSignals(neighborAiSignal, modelAiSignal);
            var hardNsfwSampleEvidence = HasHardNsfwSampleEvidence(sampleSignal);
            var strongLocalPersonalNsfw = personalAiSignal is not null &&
                                          personalAiSignal.Label == ImageLabel.Nsfw &&
                                          personalAiSignal.ManualNeighborCount > 0 &&
                                          personalAiSignal.SameLabelCloseNeighborCount > personalAiSignal.OppositeLabelCloseNeighborCount &&
                                          personalAiSignal.Confidence >= PersonalAiMinimumConfidence;
            var canPersonalAiPromoteNsfw = CanPersonalAiPromoteNsfwForScores(
                explicitNsfwScore,
                suggestiveScore,
                hardNsfwSampleEvidence,
                strongLocalPersonalNsfw);
            var personalCorrectionBlendWeight = CalculatePersonalCorrectionBlendWeight(personalAiSignal, canPersonalAiPromoteNsfw);
            var personalCorrectionWasLimited = personalAiSignal is not null &&
                                               personalAiSignal.Label == ImageLabel.Nsfw &&
                                               !canPersonalAiPromoteNsfw;
            if (personalAiSignal is not null && personalCorrectionBlendWeight > 0d)
            {
                personalizedNsfwScore = (personalizedNsfwScore * (1d - personalCorrectionBlendWeight)) + (personalAiSignal.NsfwProbability * personalCorrectionBlendWeight);
            }

            var nearbyNsfwBoost = sampleSignal.ClosestNsfwDistance <= NearbyNsfwDistanceThreshold &&
                                   sampleSignal.ClosestNsfwDistance + NearbyLabelWinningDistance <= sampleSignal.ClosestSfwDistance;
            if (nearbyNsfwBoost)
            {
                personalizedNsfwScore = Math.Max(personalizedNsfwScore, aiCalibration.DecisionThreshold + 0.08d);
            }

            var hasPersonalLoraNsfwOverride = hasPersonalLoraPrediction &&
                                             personalAiPrediction!.NsfwProbability >= personalAiSettings.DecisionThreshold &&
                                             (personalAiPrediction.Confidence >= 0.72d ||
                                              explicitNsfwScore >= PersonalAiNsfwPromotionExplicitThreshold ||
                                              (suggestiveScore >= PersonalAiNsfwPromotionSuggestiveThreshold &&
                                               explicitNsfwScore >= PersonalAiSuggestiveRequiresExplicitThreshold) ||
                                              hardNsfwSampleEvidence);
            if (hasPersonalLoraNsfwOverride)
            {
                personalizedNsfwScore = Math.Max(personalizedNsfwScore, aiCalibration.DecisionThreshold + 0.06d);
            }

            var hasStrongPersonalNsfw = personalAiSignal is not null &&
                                        personalAiSignal.Label == ImageLabel.Nsfw &&
                                        personalAiSignal.Confidence >= PersonalAiStrongConfidence &&
                                        personalAiSignal.ClosestDistance <= PersonalAiStrongClosestDistance &&
                                        personalAiSignal.SameLabelCloseNeighborCount >= PersonalAiStrongSameLabelCloseNeighbors &&
                                        canPersonalAiPromoteNsfw;
            var hasUserPreferenceNsfwOverride = personalAiSignal is not null &&
                                                personalAiSignal.Label == ImageLabel.Nsfw &&
                                                personalAiSignal.ManualNeighborCount > 0 &&
                                                personalAiSignal.SameLabelCloseNeighborCount > personalAiSignal.OppositeLabelCloseNeighborCount &&
                                                personalAiSignal.Confidence >= UserPreferenceNsfwOverrideConfidence &&
                                                canPersonalAiPromoteNsfw;
            var hasStrongPersonalSfw = personalAiSignal is not null &&
                                       personalAiSignal.Label == ImageLabel.Sfw &&
                                       personalAiSignal.Confidence >= PersonalAiStrongConfidence &&
                                       personalAiSignal.ClosestDistance <= PersonalAiStrongClosestDistance &&
                                       personalAiSignal.SameLabelCloseNeighborCount >= PersonalAiStrongSameLabelCloseNeighbors;
            if (hasStrongPersonalNsfw || hasUserPreferenceNsfwOverride)
            {
                personalizedNsfwScore = Math.Max(personalizedNsfwScore, aiCalibration.DecisionThreshold + (hasStrongPersonalNsfw ? 0.08d : 0.04d));
            }
            else if (hasStrongPersonalSfw)
            {
                personalizedNsfwScore = Math.Min(personalizedNsfwScore, aiCalibration.DecisionThreshold - 0.08d);
            }

            var finalLabel = personalizedNsfwScore >= aiCalibration.DecisionThreshold ? ImageLabel.Nsfw : ImageLabel.Sfw;
            var lowExplicitSafetyGuard = explicitNsfwScore <= LowExplicitNsfwGuardThreshold &&
                                         suggestiveScore <= LowSuggestiveGuardThreshold &&
                                         !hardNsfwSampleEvidence;
            var suggestiveOnlySfwGuard = ShouldApplySuggestiveOnlySfwGuard(
                finalLabel == ImageLabel.Nsfw,
                explicitNsfwScore,
                suggestiveScore,
                hardNsfwSampleEvidence,
                hasPersonalLoraNsfwOverride);
            if (finalLabel == ImageLabel.Nsfw && (lowExplicitSafetyGuard || suggestiveOnlySfwGuard))
            {
                finalLabel = ImageLabel.Sfw;
                var guardMargin = suggestiveOnlySfwGuard
                    ? Math.Max(0.08d, aiCalibration.ReviewMargin + 0.02d)
                    : 0.12d;
                personalizedNsfwScore = Math.Min(personalizedNsfwScore, aiCalibration.DecisionThreshold - guardMargin);
            }
            else if (finalLabel == ImageLabel.Nsfw && explicitNsfwScore < 0.5d && sampleNsfwProbability < 0.68d && !nearbyNsfwBoost && !hasStrongPersonalNsfw && !hasUserPreferenceNsfwOverride)
            {
                if (!hasPersonalLoraNsfwOverride)
                {
                    finalLabel = ImageLabel.Sfw;
                    personalizedNsfwScore = Math.Min(personalizedNsfwScore, aiCalibration.DecisionThreshold - 0.03d);
                }
            }

            var nearDecisionBoundary = Math.Abs(personalizedNsfwScore - aiCalibration.DecisionThreshold) <= aiCalibration.ReviewMargin;
            var personalLoraLabel = hasPersonalLoraPrediction && personalAiPrediction is not null
                ? (personalAiPrediction.NsfwProbability >= personalAiSettings.DecisionThreshold ? ImageLabel.Nsfw : ImageLabel.Sfw)
                : ImageLabel.Unknown;
            var hasModelConflict = personalLoraLabel is ImageLabel.Sfw or ImageLabel.Nsfw && personalLoraLabel != finalLabel;
            var displayLabel = nearDecisionBoundary || hasModelConflict ? ImageLabel.Uncertain : finalLabel;
            var aiConfidence = finalLabel == ImageLabel.Nsfw
                ? Math.Clamp((personalizedNsfwScore - aiCalibration.DecisionThreshold) / Math.Max(0.01d, 1d - aiCalibration.DecisionThreshold), 0.46d, 0.98d)
                : Math.Clamp((aiCalibration.DecisionThreshold - personalizedNsfwScore) / Math.Max(0.01d, aiCalibration.DecisionThreshold), 0.42d, 0.98d);
            var calibrationText = aiCalibration.IsPersonalized
                ? $"已结合 {aiCalibration.TrainingSampleCount} 张个人样本小幅校准阈值，其中人工纠错 {aiCalibration.ManualCorrectionCount} 张。"
                : $"个人阈值仍使用默认保守值，已积累 AI 样本 {aiCalibration.TrainingSampleCount} 张（SFW {aiCalibration.SfwCount}，NSFW {aiCalibration.NsfwCount}）。";
            var safetyText = hasModelConflict
                ? "通用模型与个人模型发生冲突，已自动放入不确定队列，建议优先复查。"
                : nearDecisionBoundary
                    ? "该图接近个人阈值，已自动放入不确定队列，建议优先复查。"
                : lowExplicitSafetyGuard
                    ? "显性 NSFW 与性感提示都很低，已启用 SFW 保护，避免个人学习层过度误判。"
                : suggestiveOnlySfwGuard
                    ? "显性 NSFW 很低，仅性感提示和个人近邻不足以判为 NSFW，已启用 SFW 保护。"
                    : "AI 视觉模型已参与图片内容判断。";
            var personalAiText = personalAiSignal is null
                ? $"AI 纠错校准已积累 {aiCorrectionProfile.TrainingSampleCount} 条经验（人工纠正 {aiCorrectionProfile.ManualCorrectionCount}，确认移动 {aiCorrectionProfile.ConfirmedCount}，样本库 {aiCorrectionProfile.SampleLibraryCount}），本图未达到校准触发条件。"
                : personalCorrectionWasLimited
                    ? $"已把你的纠错经验融入 AI 综合分，个人模型倾向 NSFW，但缺少显性 NSFW 证据，本次只按 {personalCorrectionBlendWeight:P0} 权重轻微校准。"
                    : personalAiSignal.IsGlobalModel
                        ? $"已把你的纠错经验融入 AI 综合分，全局个人模型倾向 {(personalAiSignal.Label == ImageLabel.Nsfw ? "NSFW" : "SFW")}，NSFW 概率 {personalAiSignal.NsfwProbability:P0}，校准权重 {personalCorrectionBlendWeight:P0}，训练样本 {personalAiSignal.NeighborCount} 张。"
                        : $"已把你的纠错经验融入 AI 综合分，近邻经验倾向 {(personalAiSignal.Label == ImageLabel.Nsfw ? "NSFW" : "SFW")}，NSFW 概率 {personalAiSignal.NsfwProbability:P0}，校准权重 {personalCorrectionBlendWeight:P0}，人工纠正近邻 {personalAiSignal.ManualNeighborCount} 个，同向近邻 {personalAiSignal.SameLabelCloseNeighborCount} 个。";
            var personalLoraText = hasPersonalLoraPrediction
                ? $"个人 LoRA 大模型已参与主判，NSFW 概率 {personalAiPrediction!.NsfwProbability:P0}，置信度 {personalAiPrediction.Confidence:P0}，版本 {personalAiPrediction.ModelVersion}。"
                : "个人 LoRA 大模型尚未参与本图主判。";

            return new PredictionResult
            {
                SuggestedLabel = finalLabel,
                FinalLabel = displayLabel,
                LabelOrigin = hasPersonalLoraPrediction ||
                              (personalAiSignal is not null &&
                               personalCorrectionBlendWeight > 0d &&
                               ((personalAiSignal.Confidence >= 0.85d &&
                                 (personalAiSignal.IsGlobalModel || personalAiSignal.ClosestDistance <= 0.12d) &&
                                 personalAiSignal.SameLabelCloseNeighborCount >= PersonalAiMinimumSameLabelCloseNeighbors) ||
                                (personalAiSignal.ManualNeighborCount > 0 &&
                                 personalAiSignal.SameLabelCloseNeighborCount > personalAiSignal.OppositeLabelCloseNeighborCount &&
                                 personalAiSignal.Confidence >= UserPreferenceNsfwOverrideConfidence)))
                    ? LabelOrigin.PersonalAi
                    : LabelOrigin.AiModel,
                Confidence = aiConfidence,
                Explanation = $"综合 NSFW 分数 {personalizedNsfwScore:P0}，显性 NSFW {explicitNsfwScore:P0}，性感提示 {suggestiveScore:P0}，个人阈值 {aiCalibration.DecisionThreshold:P0}。{personalLoraText}{calibrationText}{personalAiText}{safetyText}",
                UsedAiModel = true,
                UsedPersonalLearning = personalAiSignal is not null && personalCorrectionBlendWeight > 0d,
                UsedPersonalLora = hasPersonalLoraPrediction,
                GenericNsfwScore = aiClassification.NsfwScore,
                PersonalNsfwScore = personalizedNsfwScore,
            };
        }

        if (personalAiPrediction is { IsAvailable: true } &&
            personalAiPrediction.Confidence >= personalAiSettings.MinimumConfidence)
        {
            var finalLabel = personalAiPrediction.NsfwProbability >= personalAiSettings.DecisionThreshold ? ImageLabel.Nsfw : ImageLabel.Sfw;
            return new PredictionResult
            {
                SuggestedLabel = finalLabel,
                FinalLabel = finalLabel,
                LabelOrigin = LabelOrigin.PersonalAi,
                Confidence = Math.Clamp(personalAiPrediction.Confidence, 0.42d, 0.98d),
                Explanation = $"个人 LoRA 大模型已参与主判，NSFW 概率 {personalAiPrediction.NsfwProbability:P0}，置信度 {personalAiPrediction.Confidence:P0}，版本 {personalAiPrediction.ModelVersion}。本图未拿到通用 AI 特征，因此优先采用个人大模型结果。",
                UsedPersonalLora = true,
                PersonalNsfwScore = personalAiPrediction.NsfwProbability,
            };
        }

        if (learnedIndex.All.Count == 0)
        {
            return new PredictionResult
            {
                SuggestedLabel = ImageLabel.Nsfw,
                FinalLabel = ImageLabel.Uncertain,
                LabelOrigin = LabelOrigin.ModelPrediction,
                Confidence = 0d,
                Explanation = "当前没有可用样本，已直接放入不确定队列等待人工复查。",
                RequiresAiReview = true,
            };
        }

        var sampleVoteSignal = CalculateSampleVoteSignal(learnedIndex, averageHashBits, 20);

        var totalScore = sampleVoteSignal.SfwScore + sampleVoteSignal.NsfwScore;
        var dominantLabel = sampleVoteSignal.NsfwScore >= sampleVoteSignal.SfwScore ? ImageLabel.Nsfw : ImageLabel.Sfw;
        var voteGap = totalScore <= 0d ? 0d : Math.Abs(sampleVoteSignal.SfwScore - sampleVoteSignal.NsfwScore) / totalScore;
        var closestDistance = sampleVoteSignal.ClosestDistance;
        var closeness = Math.Max(0d, 1d - (closestDistance / 24d));
        var confidence = Math.Clamp((voteGap * 0.65d) + (closeness * 0.35d), 0d, 1d);
        var forceNsfwByNearbySample = sampleVoteSignal.ClosestNsfwDistance <= NearbyNsfwDistanceThreshold &&
                                      sampleVoteSignal.ClosestNsfwDistance + NearbyLabelWinningDistance <= sampleVoteSignal.ClosestSfwDistance;
        var forceNsfwByUncertainty = dominantLabel == ImageLabel.Sfw &&
                                     (voteGap <= ConservativeVoteGapThreshold || confidence < ConservativeConfidenceThreshold);
        var forceNsfwByUnknownPattern = dominantLabel == ImageLabel.Sfw &&
                                        closestDistance > UnknownPatternDistanceThreshold;

        if (forceNsfwByNearbySample)
        {
            dominantLabel = ImageLabel.Nsfw;
            confidence = Math.Max(confidence, 0.72d);
        }

        var nearbyNsfwNeedsReview = dominantLabel == ImageLabel.Sfw &&
                                    sampleVoteSignal.ClosestNsfwDistance <= 12 &&
                                    sampleVoteSignal.ClosestNsfwDistance + 2 <= sampleVoteSignal.ClosestSfwDistance;
        var requiresAiReview = forceNsfwByUncertainty || forceNsfwByUnknownPattern || nearbyNsfwNeedsReview ||
                               (dominantLabel == ImageLabel.Nsfw && confidence < 0.55d);
        var conservativeReason = forceNsfwByNearbySample
            ? "最近的 NSFW 样本明显更近，才启用防漏判归入 NSFW。"
            : requiresAiReview
                ? "按样本相似度综合判断，因置信度偏低会交给 AI 复核。"
                : "按样本相似度综合判断。";

        return new PredictionResult
        {
            SuggestedLabel = dominantLabel,
            FinalLabel = requiresAiReview ? ImageLabel.Uncertain : dominantLabel,
            LabelOrigin = LabelOrigin.ModelPrediction,
            Confidence = confidence,
            Explanation = $"参考了 {sampleVoteSignal.NeighborCount} 张已学习图片；最近 SFW 距离 {sampleVoteSignal.ClosestSfwDistance}，最近 NSFW 距离 {sampleVoteSignal.ClosestNsfwDistance}，综合置信度 {confidence:P0}。{conservativeReason}",
            RequiresAiReview = requiresAiReview,
        };
    }

    private PredictionResult PredictBySampleTask(
        string contentId,
        ulong averageHashBits,
        int width,
        int height,
        LearnedSampleIndex learnedIndex,
        ImageLabel primaryLabel,
        ImageLabel secondaryLabel,
        string primaryText,
        string secondaryText,
        PersonalAiPredictionResult? personalAiPrediction = null,
        PersonalAiModelSettings? personalAiSettings = null)
    {
        if (learnedIndex.ByContentId.TryGetValue(contentId, out var exactMatch) &&
            (exactMatch.CurrentLabel == primaryLabel || exactMatch.CurrentLabel == secondaryLabel))
        {
            return new PredictionResult
            {
                SuggestedLabel = exactMatch.CurrentLabel,
                FinalLabel = exactMatch.CurrentLabel,
                LabelOrigin = exactMatch.LabelOrigin,
                Confidence = 1d,
                Explanation = "命中当前任务已学习图片缓存，直接沿用历史判断。",
                IsManualCorrection = exactMatch.LabelOrigin == LabelOrigin.ManualCorrection,
                IsHumanConfirmed = exactMatch.IsHumanConfirmed || exactMatch.LabelOrigin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove,
                UsedHistoryCache = true,
            };
        }

        var visualKey = BuildVisualKey(averageHashBits, width, height);
        if (learnedIndex.ByVisualKey.TryGetValue(visualKey, out var visualMatch) &&
            (visualMatch.CurrentLabel == primaryLabel || visualMatch.CurrentLabel == secondaryLabel))
        {
            return new PredictionResult
            {
                SuggestedLabel = visualMatch.CurrentLabel,
                FinalLabel = visualMatch.CurrentLabel,
                LabelOrigin = visualMatch.LabelOrigin,
                Confidence = 0.98d,
                Explanation = "命中当前任务同尺寸同指纹图片，直接沿用已学习结果。",
                IsManualCorrection = visualMatch.LabelOrigin == LabelOrigin.ManualCorrection,
                IsHumanConfirmed = visualMatch.IsHumanConfirmed || visualMatch.LabelOrigin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove,
                UsedHistoryCache = true,
            };
        }

        var nearestManualCorrection = FindNearestManualCorrection(learnedIndex, averageHashBits, ManualDuplicateDistanceThreshold);
        if (nearestManualCorrection.HasValue &&
            (nearestManualCorrection.Value.Sample.CurrentLabel == primaryLabel || nearestManualCorrection.Value.Sample.CurrentLabel == secondaryLabel))
        {
            var (sample, distance) = nearestManualCorrection.Value;
            var manualConfidence = Math.Clamp(1d - (distance / 16d), 0.9d, 0.99d);
            return new PredictionResult
            {
                SuggestedLabel = sample.CurrentLabel,
                FinalLabel = sample.CurrentLabel,
                LabelOrigin = LabelOrigin.ManualCorrection,
                Confidence = manualConfidence,
                Explanation = $"命中近重复人工纠正图，相似距离 {distance}，沿用你的手动判断。",
                IsHumanConfirmed = true,
                UsedHistoryCache = true,
            };
        }

        var decisionThreshold = personalAiSettings?.DecisionThreshold > 0d
            ? Math.Clamp(personalAiSettings.DecisionThreshold, 0.01d, 0.99d)
            : 0.5d;
        var minimumConfidence = personalAiSettings?.MinimumConfidence > 0d
            ? Math.Clamp(personalAiSettings.MinimumConfidence, 0.01d, 0.99d)
            : 0.56d;
        var hasPersonalPrediction = personalAiPrediction is { IsAvailable: true } &&
                                    personalAiPrediction.Confidence >= minimumConfidence;
        var personalLabel = hasPersonalPrediction && personalAiPrediction is not null
            ? (personalAiPrediction.NsfwProbability >= decisionThreshold ? secondaryLabel : primaryLabel)
            : ImageLabel.Unknown;
        var personalConfidence = personalAiPrediction?.Confidence ?? 0d;

        if (learnedIndex.All.Count == 0)
        {
            if (hasPersonalPrediction)
            {
                return new PredictionResult
                {
                    SuggestedLabel = personalLabel,
                    FinalLabel = personalLabel,
                    LabelOrigin = LabelOrigin.PersonalAi,
                    Confidence = Math.Clamp(personalConfidence, 0.42d, 0.98d),
                    Explanation = $"当前任务样本库暂无可用近邻，已使用人物个人模型判断为 {GetLabelDisplayText(personalLabel)}，置信度 {personalConfidence:P0}。",
                    UsedPersonalLora = true,
                    UsedPersonalLearning = true,
                };
            }

            return new PredictionResult
            {
                SuggestedLabel = ImageLabel.Unknown,
                FinalLabel = ImageLabel.Unknown,
                LabelOrigin = LabelOrigin.ModelPrediction,
                Confidence = 0d,
                Explanation = "当前任务还没有可用样本，建议先加入人物/非人物样本库。",
            };
        }

        var sampleVoteSignal = CalculateSampleVoteSignal(learnedIndex, averageHashBits, 24, primaryLabel, secondaryLabel, secondaryBias: 1.0d);
        var totalScore = sampleVoteSignal.SfwScore + sampleVoteSignal.NsfwScore;
        var primaryScore = sampleVoteSignal.SfwScore;
        var secondaryScore = sampleVoteSignal.NsfwScore;
        var dominantLabel = primaryScore >= secondaryScore ? primaryLabel : secondaryLabel;
        var dominantText = dominantLabel == primaryLabel ? primaryText : secondaryText;
        var voteGap = totalScore <= 0d ? 0d : Math.Abs(primaryScore - secondaryScore) / totalScore;
        var closestDistance = sampleVoteSignal.ClosestDistance;
        var closeness = Math.Max(0d, 1d - (closestDistance / 26d));
        var confidence = Math.Clamp((voteGap * 0.62d) + (closeness * 0.38d), 0d, 1d);
        var finalLabel = dominantLabel;
        var labelOrigin = LabelOrigin.ModelPrediction;
        var usedPersonalLora = false;
        var explanationSuffix = string.Empty;

        if (hasPersonalPrediction)
        {
            usedPersonalLora = true;
            if (personalLabel == dominantLabel)
            {
                labelOrigin = LabelOrigin.PersonalAi;
                confidence = Math.Clamp(Math.Max(confidence, personalConfidence), 0d, 0.98d);
                explanationSuffix = $" 人物个人模型与样本库一致，模型置信度 {personalConfidence:P0}。";
            }
            else if (confidence >= 0.62d && personalConfidence >= 0.62d)
            {
                finalLabel = ImageLabel.Uncertain;
                labelOrigin = LabelOrigin.PersonalAi;
                confidence = Math.Clamp(Math.Min(confidence, personalConfidence), 0.42d, 0.72d);
                explanationSuffix = $" 人物个人模型倾向 {GetLabelDisplayText(personalLabel)}，样本库倾向 {dominantText}，已放入不确定队列。";
            }
            else if (personalConfidence > confidence + 0.1d)
            {
                finalLabel = personalLabel;
                labelOrigin = LabelOrigin.PersonalAi;
                confidence = Math.Clamp(personalConfidence, 0d, 0.96d);
                explanationSuffix = $" 人物个人模型置信度 {personalConfidence:P0} 高于样本近邻，优先采用模型判断。";
            }
        }

        return new PredictionResult
        {
            SuggestedLabel = finalLabel == ImageLabel.Uncertain ? dominantLabel : finalLabel,
            FinalLabel = finalLabel,
            LabelOrigin = labelOrigin,
            Confidence = confidence,
            Explanation = $"人物标准：真人、真人手/腿等真人元素、真人海报算人物；二次元、风景、玩偶/雕像、二次元海报不算。参考 {sampleVoteSignal.NeighborCount} 张当前任务样本，最近 {primaryText} 距离 {sampleVoteSignal.ClosestSfwDistance}，最近 {secondaryText} 距离 {sampleVoteSignal.ClosestNsfwDistance}，综合判断为 {GetLabelDisplayText(finalLabel)}，置信度 {confidence:P0}。{explanationSuffix}",
            UsedPersonalLora = usedPersonalLora,
            UsedPersonalLearning = usedPersonalLora,
        };
    }

    private async Task<AiClassificationResult?> ResolveAiClassificationAsync(PersistedAppState state, FileCacheRecord cacheRecord, string normalizedPath, WorkspaceSettings settings, object stateLock, CancellationToken cancellationToken)
    {
        if (settings.AiModel is null || !settings.AiModel.IsEnabled || string.IsNullOrWhiteSpace(cacheRecord.ContentId))
        {
            return null;
        }

        var cachedResult = TryGetCachedAiClassification(state, cacheRecord, stateLock);
        if (cachedResult is not null)
        {
            return cachedResult;
        }

        var result = await _aiNsfwClassifierService.ClassifyAsync(normalizedPath, settings.AiModel, cancellationToken).ConfigureAwait(false);
        if (!result.IsAvailable)
        {
            return result;
        }

        lock (stateLock)
        {
            state.AiClassificationsById[cacheRecord.ContentId] = new AiClassificationCacheRecord
            {
                ContentId = cacheRecord.ContentId,
                ModelId = result.ModelId,
                NsfwScore = result.NsfwScore,
                SfwScore = result.SfwScore,
                LabelScores = result.LabelScores.ToDictionary(item => item.Key, item => item.Value, StringComparer.OrdinalIgnoreCase),
                UpdatedAtUtc = DateTime.UtcNow,
            };
        }

        return result;
    }

    private async Task EnsureAiClassificationForCorrectionAsync(PersistedAppState state, FileCacheRecord cacheRecord, string filePath, ClassificationTaskMode taskMode, CancellationToken cancellationToken)
    {
        if (taskMode != ClassificationTaskMode.ContentSafety ||
            string.IsNullOrWhiteSpace(cacheRecord.ContentId) ||
            !File.Exists(filePath))
        {
            return;
        }

        var aiSettings = state.Settings.AiModel ?? new AiModelSettings();
        if (!aiSettings.IsEnabled || !GetAiModelStatus(state.Settings).IsInstalled)
        {
            return;
        }

        if (state.AiClassificationsById.TryGetValue(cacheRecord.ContentId, out var cachedRecord) &&
            TryUpgradeAiCacheRecord(cachedRecord))
        {
            return;
        }

        var result = await _aiNsfwClassifierService.ClassifyAsync(filePath, aiSettings, cancellationToken).ConfigureAwait(false);
        if (!result.IsAvailable)
        {
            return;
        }

        state.AiClassificationsById[cacheRecord.ContentId] = new AiClassificationCacheRecord
        {
            ContentId = cacheRecord.ContentId,
            ModelId = result.ModelId,
            NsfwScore = result.NsfwScore,
            SfwScore = result.SfwScore,
            LabelScores = result.LabelScores.ToDictionary(item => item.Key, item => item.Value, StringComparer.OrdinalIgnoreCase),
            UpdatedAtUtc = DateTime.UtcNow,
        };
    }

    private async Task<AiFeatureExtractionSummary> ExtractAiFeaturesInBatchesAsync(
        PersistedAppState state,
        IReadOnlyList<AiFeatureRequest> requests,
        WorkspaceSettings settings,
        object stateLock,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress,
        WorkspaceProgressStage progressStage,
        string progressTitle,
        string progressDetail)
    {
        if (requests.Count == 0 || settings.AiModel is null || !settings.AiModel.IsEnabled)
        {
            return new AiFeatureExtractionSummary(0, requests.Count, 0);
        }

        var stopwatch = Stopwatch.StartNew();
        var inferenceCount = 0;
        var skippedCount = 0;
        var completedCount = 0;
        var lastLoggedCompletedCount = 0;
        var lastLoggedElapsedMilliseconds = 0L;
        ReportProgress(progress, progressStage, progressTitle, progressDetail, string.Empty, 0, requests.Count);

        var batchIndex = 0;
        var totalBatches = (int)Math.Ceiling(requests.Count / (double)AiFeatureExtractionBatchSize);
        foreach (var batch in requests.Chunk(AiFeatureExtractionBatchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();

            batchIndex++;
            var batchRequests = batch.ToArray();
            var batchStart = completedCount + 1;
            var batchEnd = completedCount + batchRequests.Length;
            var batchCurrentItem = $"{Path.GetFileName(batchRequests[0].FilePath)} - {Path.GetFileName(batchRequests[^1].FilePath)}";
            ReportProgress(
                progress,
                progressStage,
                progressTitle,
                $"{progressDetail} 正在处理第 {batchIndex}/{totalBatches} 批，本批 {batchRequests.Length} 张，整体位置 {batchStart}-{batchEnd}/{requests.Count}。",
                batchCurrentItem,
                completedCount,
                requests.Count);

            IReadOnlyList<AiClassificationResult> batchResults;
            try
            {
                batchResults = await _aiNsfwClassifierService
                    .ClassifyBatchAsync(batchRequests.Select(item => item.FilePath).ToList(), settings.AiModel, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                batchResults = batchRequests.Select(_ => new AiClassificationResult { IsAvailable = false }).ToList();
            }

            lock (stateLock)
            {
                for (var index = 0; index < batchRequests.Length; index++)
                {
                    var result = index < batchResults.Count ? batchResults[index] : null;
                    if (result is not { IsAvailable: true })
                    {
                        skippedCount++;
                        continue;
                    }

                    state.AiClassificationsById[batchRequests[index].ContentId] = new AiClassificationCacheRecord
                    {
                        ContentId = batchRequests[index].ContentId,
                        ModelId = result.ModelId,
                        NsfwScore = result.NsfwScore,
                        SfwScore = result.SfwScore,
                        LabelScores = result.LabelScores.ToDictionary(item => item.Key, item => item.Value, StringComparer.OrdinalIgnoreCase),
                        UpdatedAtUtc = DateTime.UtcNow,
                    };
                    inferenceCount++;
                }
            }

            completedCount += batchRequests.Length;
            if (completedCount % AiFeatureExtractionPersistInterval == 0)
            {
                await PersistAsync().ConfigureAwait(false);
                var elapsedMilliseconds = stopwatch.ElapsedMilliseconds;
                var recentCount = completedCount - lastLoggedCompletedCount;
                var recentElapsedMilliseconds = Math.Max(1L, elapsedMilliseconds - lastLoggedElapsedMilliseconds);
                await _performanceLogService.LogAsync(
                    "AiFeatureBatch",
                    $"阶段 {progressTitle}，已处理 {completedCount}/{requests.Count}，新增缓存 {inferenceCount}，跳过 {skippedCount}，最近 {recentCount} 张耗时 {recentElapsedMilliseconds / 1000d:F2}s，均速 {recentCount * 1000d / recentElapsedMilliseconds:F2} 张/秒。").ConfigureAwait(false);
                lastLoggedCompletedCount = completedCount;
                lastLoggedElapsedMilliseconds = elapsedMilliseconds;
            }

            if (ShouldReportProgress(completedCount, requests.Count))
            {
                ReportProgress(
                    progress,
                    progressStage,
                    progressTitle,
                    $"AI 特征已提取 {completedCount} / {requests.Count} 张，新增缓存 {inferenceCount} 张，跳过 {skippedCount} 张；已完成第 {batchIndex}/{totalBatches} 批。",
                    Path.GetFileName(batchRequests[^1].FilePath),
                    completedCount,
                    requests.Count);
            }
        }

        stopwatch.Stop();
        return new AiFeatureExtractionSummary(inferenceCount, skippedCount, stopwatch.ElapsedMilliseconds);
    }

    private static bool ShouldRunAiInference(PredictionResult prediction)
    {
        return prediction.LabelOrigin == LabelOrigin.ModelPrediction;
    }

    private static double CalculatePersonalCorrectionBlendWeight(AiCorrectionSignal? signal, bool canPromoteNsfw)
    {
        if (signal is null)
        {
            return 0d;
        }

        var manualBoost = Math.Clamp(signal.ManualNeighborCount / 4d, 0d, 0.18d);
        var localMaxWeight = Math.Clamp(0.22d + (signal.Confidence * 0.26d) + manualBoost, PersonalAiMaximumBlendWeight, 0.72d);
        var maxWeight = signal.IsGlobalModel ? PersonalAiModelMaximumBlendWeight : localMaxWeight;
        var hasManualPreferenceSignal = signal.ManualNeighborCount > 0;
        var confidenceFloor = hasManualPreferenceSignal ? UserPreferenceAiMinimumConfidence : PersonalAiMinimumConfidence;
        var minimumWeight = signal.IsGlobalModel
            ? 0.06d
            : hasManualPreferenceSignal
                ? 0.1d
                : 0.04d;
        var confidenceProgress = Math.Clamp(
            (signal.Confidence - confidenceFloor) / Math.Max(0.01d, 1d - confidenceFloor),
            0d,
            1d);
        var weight = Math.Clamp(
            minimumWeight + (confidenceProgress * (maxWeight - minimumWeight)),
            minimumWeight,
            maxWeight);

        return signal.Label == ImageLabel.Nsfw && !canPromoteNsfw
            ? Math.Min(weight, PersonalAiLimitedNsfwBlendWeight)
            : weight;
    }

    internal static bool CanPersonalAiPromoteNsfwForScores(
        double explicitNsfwScore,
        double suggestiveScore,
        bool hardNsfwSampleEvidence,
        bool strongLocalPersonalNsfw)
    {
        if (hardNsfwSampleEvidence)
        {
            return true;
        }

        if (explicitNsfwScore >= PersonalAiNsfwPromotionExplicitThreshold)
        {
            return true;
        }

        if (suggestiveScore >= PersonalAiNsfwPromotionSuggestiveThreshold &&
            explicitNsfwScore >= PersonalAiSuggestiveRequiresExplicitThreshold)
        {
            return true;
        }

        return strongLocalPersonalNsfw &&
               explicitNsfwScore >= PersonalAiSuggestiveRequiresExplicitThreshold;
    }

    internal static bool ShouldApplySuggestiveOnlySfwGuard(
        bool finalLabelIsNsfw,
        double explicitNsfwScore,
        double suggestiveScore,
        bool hardNsfwSampleEvidence,
        bool hasPersonalLoraNsfwOverride)
    {
        return finalLabelIsNsfw &&
               explicitNsfwScore < PersonalAiSuggestiveRequiresExplicitThreshold &&
               suggestiveScore >= PersonalAiNsfwPromotionSuggestiveThreshold &&
               !hardNsfwSampleEvidence &&
               !hasPersonalLoraNsfwOverride;
    }

    internal static bool IsPersonalModelPrimaryEligible(
        PersonalAiEvaluationSummary? summary,
        PersonalAiModelSettings settings)
    {
        if (summary is null ||
            summary.EvaluatedSamples <= 0 ||
            summary.SfwSamples <= 0 ||
            summary.NsfwSamples <= 0)
        {
            return false;
        }

        var minimumAccuracy = settings.PrimaryModelMinimumAccuracy <= 0d
            ? 0.92d
            : Math.Clamp(settings.PrimaryModelMinimumAccuracy, 0.5d, 0.995d);
        var maxNsfwFalseNegativeRate = settings.PrimaryModelMaximumNsfwFalseNegativeRate <= 0d
            ? 0.04d
            : Math.Clamp(settings.PrimaryModelMaximumNsfwFalseNegativeRate, 0d, 0.5d);
        var maxSfwFalsePositiveRate = settings.PrimaryModelMaximumSfwFalsePositiveRate <= 0d
            ? 0.08d
            : Math.Clamp(settings.PrimaryModelMaximumSfwFalsePositiveRate, 0d, 0.5d);

        return summary.Accuracy >= minimumAccuracy &&
               summary.NsfwFalseNegativeRate <= maxNsfwFalseNegativeRate &&
               summary.SfwFalsePositiveRate <= maxSfwFalsePositiveRate &&
               summary.AllowsPersonalModelPrimary;
    }

    private static bool HasHardNsfwSampleEvidence(SampleVoteSignal sampleSignal)
    {
        return sampleSignal.ClosestNsfwDistance <= HardNsfwSampleDistanceThreshold &&
               sampleSignal.ClosestNsfwDistance + HardNsfwSampleWinningDistance <= sampleSignal.ClosestSfwDistance;
    }

    private static AiClassificationResult? TryGetCachedAiClassification(PersistedAppState state, FileCacheRecord cacheRecord, object stateLock)
    {
        if (string.IsNullOrWhiteSpace(cacheRecord.ContentId))
        {
            return null;
        }

        lock (stateLock)
        {
            return state.AiClassificationsById.TryGetValue(cacheRecord.ContentId, out var cachedRecord) &&
                   TryUpgradeAiCacheRecord(cachedRecord)
                ? ToAiClassificationResult(cachedRecord)
                : null;
        }
    }

    private static bool TryUpgradeAiCacheRecord(AiClassificationCacheRecord cachedRecord)
    {
        if (string.Equals(cachedRecord.ModelId, AiNsfwClassifierService.ScoringModelId, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (cachedRecord.LabelScores.Count == 0 ||
            !cachedRecord.LabelScores.ContainsKey("nude.strong_explicit"))
        {
            return false;
        }

        cachedRecord.ModelId = AiNsfwClassifierService.ScoringModelId;
        cachedRecord.NsfwScore = AiNsfwClassifierService.CalculateWeightedNsfwScore(cachedRecord.LabelScores);
        cachedRecord.SfwScore = AiNsfwClassifierService.CalculateSfwScore(cachedRecord.LabelScores);
        cachedRecord.UpdatedAtUtc = DateTime.UtcNow;
        return true;
    }

    private static AiClassificationResult ToAiClassificationResult(AiClassificationCacheRecord cachedRecord)
    {
        return new AiClassificationResult
        {
            IsAvailable = true,
            ModelId = cachedRecord.ModelId,
            NsfwScore = cachedRecord.NsfwScore,
            SfwScore = cachedRecord.SfwScore,
            LabelScores = cachedRecord.LabelScores,
            Message = "已复用 AI 视觉缓存。",
        };
    }

    private static AiPersonalCalibration BuildPersonalAiCalibration(
        PersistedAppState state,
        AiModelSettings settings,
        PersonalAiModelSettings personalAiSettings)
    {
        var sfwWeight = 0d;
        var sfwScoreSum = 0d;
        var nsfwWeight = 0d;
        var nsfwScoreSum = 0d;
        var trainingSampleCount = 0;
        var manualCorrectionCount = 0;
        var sfwCount = 0;
        var nsfwCount = 0;

        foreach (var learnedRecord in state.LearnedImagesById.Values)
        {
            if (learnedRecord.TaskMode != ClassificationTaskMode.ContentSafety ||
                learnedRecord.CurrentLabel is not (ImageLabel.Sfw or ImageLabel.Nsfw) ||
                learnedRecord.LabelOrigin is not (LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove) ||
                !ShouldUseLearnedRecordForPersonalLearning(learnedRecord, ClassificationTaskMode.ContentSafety, personalAiSettings) ||
                !state.AiClassificationsById.TryGetValue(learnedRecord.ContentId, out var aiRecord) ||
                !TryUpgradeAiCacheRecord(aiRecord))
            {
                continue;
            }

            var weight = learnedRecord.LabelOrigin switch
            {
                LabelOrigin.ManualCorrection => 4d,
                LabelOrigin.ConfirmedMove => 2d,
                _ => 1d,
            };
            trainingSampleCount++;
            if (learnedRecord.LabelOrigin == LabelOrigin.ManualCorrection)
            {
                manualCorrectionCount++;
            }

            if (learnedRecord.CurrentLabel == ImageLabel.Nsfw)
            {
                nsfwCount++;
                nsfwWeight += weight;
                nsfwScoreSum += aiRecord.NsfwScore * weight;
            }
            else
            {
                sfwCount++;
                sfwWeight += weight;
                sfwScoreSum += aiRecord.NsfwScore * weight;
            }
        }

        var defaultThreshold = ResolveDefaultNsfwThreshold(settings);
        if (trainingSampleCount < PersonalCalibrationMinimumTrainingSamples ||
            sfwCount < PersonalCalibrationMinimumSamplesPerLabel ||
            nsfwCount < PersonalCalibrationMinimumSamplesPerLabel ||
            sfwWeight <= 0d ||
            nsfwWeight <= 0d)
        {
            return new AiPersonalCalibration(defaultThreshold, 0.08d, trainingSampleCount, manualCorrectionCount, sfwCount, nsfwCount, false);
        }

        var sfwAverage = sfwScoreSum / sfwWeight;
        var nsfwAverage = nsfwScoreSum / nsfwWeight;
        if (nsfwAverage - sfwAverage < PersonalCalibrationMinimumScoreGap)
        {
            return new AiPersonalCalibration(defaultThreshold, 0.08d, trainingSampleCount, manualCorrectionCount, sfwCount, nsfwCount, false);
        }

        var learnedThreshold = Math.Clamp((sfwAverage + nsfwAverage) / 2d, 0.5d, 0.85d);
        var maxThresholdShift = manualCorrectionCount >= 20 ? 0.08d : 0.04d;
        var blendedThreshold = (defaultThreshold * 0.85d) + (learnedThreshold * 0.15d);
        var threshold = Math.Clamp(
            Math.Clamp(blendedThreshold, defaultThreshold - maxThresholdShift, defaultThreshold + maxThresholdShift),
            0.5d,
            0.85d);
        var reviewMargin = manualCorrectionCount > 0 ? 0.1d : 0.08d;
        return new AiPersonalCalibration(threshold, reviewMargin, trainingSampleCount, manualCorrectionCount, sfwCount, nsfwCount, true);
    }

    private static AiCorrectionProfile BuildPersonalAiCorrectionProfile(PersistedAppState state, PersonalAiModelSettings personalAiSettings)
    {
        var rawExamples = new List<AiCorrectionExample>();

        foreach (var learnedRecord in state.LearnedImagesById.Values)
        {
            if (learnedRecord.TaskMode != ClassificationTaskMode.ContentSafety ||
                learnedRecord.CurrentLabel is not (ImageLabel.Sfw or ImageLabel.Nsfw) ||
                learnedRecord.LabelOrigin is not (LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove or LabelOrigin.SampleLibrary) ||
                !ShouldUseLearnedRecordForPersonalLearning(learnedRecord, ClassificationTaskMode.ContentSafety, personalAiSettings) ||
                !state.AiClassificationsById.TryGetValue(learnedRecord.ContentId, out var aiRecord) ||
                !TryUpgradeAiCacheRecord(aiRecord))
            {
                continue;
            }

            var weight = learnedRecord.LabelOrigin switch
            {
                LabelOrigin.ManualCorrection => 4d + Math.Min(2d, learnedRecord.CorrectionCount * 0.5d),
                LabelOrigin.ConfirmedMove => 1.6d,
                LabelOrigin.SampleLibrary => 0.55d,
                _ => 0.25d,
            };
            rawExamples.Add(new AiCorrectionExample(
                learnedRecord.CurrentLabel,
                learnedRecord.LabelOrigin,
                BuildAiFeatureVector(aiRecord.LabelScores),
                weight));
        }

        var examples = BalancePersonalAiExamples(rawExamples);
        var userPreferenceExamples = examples
            .Where(example => example.Origin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove)
            .ToList();
        var sameLabelFeatureDistances = EstimateSameLabelAiFeatureDistances(examples);
        var examplesByScoreBucket = BuildAiCorrectionFeatureBuckets(examples);
        var userPreferenceExamplesByScoreBucket = BuildAiCorrectionFeatureBuckets(userPreferenceExamples);
        var manualCorrectionCount = examples.Count(example => example.Origin == LabelOrigin.ManualCorrection);
        var confirmedCount = examples.Count(example => example.Origin == LabelOrigin.ConfirmedMove);
        var sampleLibraryCount = examples.Count(example => example.Origin == LabelOrigin.SampleLibrary);
        var sfwCount = examples.Count(example => example.Label == ImageLabel.Sfw);
        var nsfwCount = examples.Count(example => example.Label == ImageLabel.Nsfw);
        return new AiCorrectionProfile(
            examples,
            userPreferenceExamples,
            sameLabelFeatureDistances,
            examplesByScoreBucket,
            userPreferenceExamplesByScoreBucket,
            manualCorrectionCount,
            confirmedCount,
            sampleLibraryCount,
            sfwCount,
            nsfwCount);
    }

    private static List<AiCorrectionExample> BalancePersonalAiExamples(List<AiCorrectionExample> rawExamples)
    {
        if (rawExamples.Count == 0)
        {
            return [];
        }

        static IEnumerable<AiCorrectionExample> SelectLabelExamples(IEnumerable<AiCorrectionExample> examples)
        {
            return examples
                .OrderByDescending(example => example.Origin == LabelOrigin.ManualCorrection)
                .ThenByDescending(example => example.Origin == LabelOrigin.ConfirmedMove)
                .ThenByDescending(example => example.Weight)
                .Take(PersonalAiModelMaxSamplesPerLabel);
        }

        var sfwExamples = SelectLabelExamples(rawExamples.Where(example => example.Label == ImageLabel.Sfw));
        var nsfwExamples = SelectLabelExamples(rawExamples.Where(example => example.Label == ImageLabel.Nsfw));
        return sfwExamples.Concat(nsfwExamples).ToList();
    }

    private static PersonalAiModel BuildPersonalAiModel(AiCorrectionProfile profile)
    {
        if (profile.TrainingSampleCount < PersonalAiMinimumTrainingSamples ||
            profile.SfwCount < PersonalAiMinimumSamplesPerLabel ||
            profile.NsfwCount < PersonalAiMinimumSamplesPerLabel ||
            profile.Examples.Count == 0)
        {
            return PersonalAiModel.Unavailable(profile.TrainingSampleCount, profile.SfwCount, profile.NsfwCount, profile.ManualCorrectionCount);
        }

        var featureLength = profile.Examples.Max(example => example.Features.Length);
        if (featureLength == 0)
        {
            return PersonalAiModel.Unavailable(profile.TrainingSampleCount, profile.SfwCount, profile.NsfwCount, profile.ManualCorrectionCount);
        }

        var weights = CreateInitialPersonalAiWeights(featureLength);
        var bias = 0d;
        var totalLabelCount = Math.Max(1d, profile.SfwCount + profile.NsfwCount);
        var sfwBalanceWeight = totalLabelCount / (2d * Math.Max(1d, profile.SfwCount));
        var nsfwBalanceWeight = totalLabelCount / (2d * Math.Max(1d, profile.NsfwCount));

        for (var epoch = 0; epoch < PersonalAiModelEpochs; epoch++)
        {
            var learningRate = PersonalAiModelLearningRate / (1d + (epoch / 55d));
            foreach (var example in profile.Examples)
            {
                var target = example.Label == ImageLabel.Nsfw ? 1d : 0d;
                var classBalanceWeight = example.Label == ImageLabel.Nsfw ? nsfwBalanceWeight : sfwBalanceWeight;
                var exampleWeight = example.Weight * classBalanceWeight;
                var probability = Sigmoid(Dot(weights, example.Features) + bias);
                var error = probability - target;

                for (var index = 0; index < weights.Length; index++)
                {
                    var feature = index < example.Features.Length ? example.Features[index] : 0d;
                    weights[index] -= learningRate * ((error * feature * exampleWeight) + (PersonalAiModelL2 * weights[index]));
                }

                bias -= learningRate * error * exampleWeight;
            }
        }

        var trainingPredictions = profile.Examples
            .Select(example => (
                Probability: Sigmoid(Dot(weights, example.Features) + bias),
                Label: example.Label,
                Weight: example.Weight * (example.Label == ImageLabel.Nsfw ? nsfwBalanceWeight : sfwBalanceWeight)))
            .ToList();
        var decisionThreshold = SelectPersonalAiDecisionThreshold(trainingPredictions);

        return new PersonalAiModel(
            weights,
            bias,
            decisionThreshold,
            profile.TrainingSampleCount,
            profile.SfwCount,
            profile.NsfwCount,
            profile.ManualCorrectionCount,
            true);
    }

    private static double[] CreateInitialPersonalAiWeights(int featureLength)
    {
        var weights = new double[featureLength];
        var seed = new[]
        {
            0.9d, 1.0d, 0.45d, 1.2d, 1.35d, 1.55d, 1.35d, 0.65d, 0.35d,
            0.30d, 0.28d, 0.45d, 0.45d, -0.38d, -0.25d, -0.42d, -0.05d, 0.02d,
        };

        for (var index = 0; index < Math.Min(featureLength, seed.Length); index++)
        {
            weights[index] = seed[index] * 0.18d;
        }

        return weights;
    }

    private static double SelectPersonalAiDecisionThreshold(IReadOnlyList<(double Probability, ImageLabel Label, double Weight)> predictions)
    {
        if (predictions.Count == 0)
        {
            return 0.55d;
        }

        var candidates = predictions
            .Select(item => Math.Clamp(item.Probability, 0.08d, 0.92d))
            .Append(0.55d)
            .Distinct()
            .OrderBy(value => value)
            .ToList();

        var bestThreshold = 0.55d;
        var bestLoss = double.PositiveInfinity;
        foreach (var threshold in candidates)
        {
            var loss = 0d;
            foreach (var prediction in predictions)
            {
                var predictedNsfw = prediction.Probability >= threshold;
                if (predictedNsfw && prediction.Label == ImageLabel.Sfw)
                {
                    loss += prediction.Weight * 1.20d;
                }
                else if (!predictedNsfw && prediction.Label == ImageLabel.Nsfw)
                {
                    loss += prediction.Weight * 1.08d;
                }
            }

            var stabilityPenalty = Math.Abs(threshold - 0.55d) * 0.05d;
            loss += stabilityPenalty;
            if (loss < bestLoss)
            {
                bestLoss = loss;
                bestThreshold = threshold;
            }
        }

        return Math.Clamp((bestThreshold * 0.82d) + (0.55d * 0.18d), 0.42d, 0.76d);
    }

    private static AiCorrectionSignal? PredictPersonalAiModel(PersonalAiModel model, AiClassificationResult aiClassification)
    {
        if (!model.IsAvailable)
        {
            return null;
        }

        var features = BuildAiFeatureVector(aiClassification.LabelScores);
        var nsfwProbability = Sigmoid(Dot(model.Weights, features) + model.Bias);
        var label = nsfwProbability >= model.DecisionThreshold ? ImageLabel.Nsfw : ImageLabel.Sfw;
        var margin = Math.Abs(nsfwProbability - model.DecisionThreshold);
        var confidence = Math.Clamp(margin / 0.32d, 0d, 1d);
        if (confidence < PersonalAiMinimumConfidence)
        {
            return null;
        }

        var sameLabelCount = label == ImageLabel.Nsfw ? model.NsfwCount : model.SfwCount;
        var oppositeLabelCount = label == ImageLabel.Nsfw ? model.SfwCount : model.NsfwCount;
        if (sameLabelCount < PersonalAiMinimumSamplesPerLabel)
        {
            return null;
        }

        return new AiCorrectionSignal(
            label,
            nsfwProbability,
            confidence,
            model.TrainingSampleCount,
            0d,
            model.ManualCorrectionCount,
            Math.Min(sameLabelCount, PersonalAiStrongSameLabelCloseNeighbors),
            Math.Min(oppositeLabelCount, sameLabelCount - 1),
            IsGlobalModel: true);
    }

    private static AiCorrectionSignal? MergePersonalAiSignals(AiCorrectionSignal? neighborSignal, AiCorrectionSignal? modelSignal)
    {
        if (neighborSignal is null)
        {
            return modelSignal;
        }

        if (modelSignal is null)
        {
            return neighborSignal;
        }

        if (neighborSignal.Label != modelSignal.Label)
        {
            var confidenceGap = Math.Abs(neighborSignal.Confidence - modelSignal.Confidence);
            if (confidenceGap < 0.18d)
            {
                return null;
            }

            return neighborSignal.Confidence > modelSignal.Confidence
                ? neighborSignal with { Confidence = Math.Max(PersonalAiMinimumConfidence, neighborSignal.Confidence - 0.08d) }
                : modelSignal with { Confidence = Math.Max(PersonalAiMinimumConfidence, modelSignal.Confidence - 0.08d) };
        }

        var totalConfidence = neighborSignal.Confidence + modelSignal.Confidence;
        if (totalConfidence <= 0d)
        {
            return neighborSignal;
        }

        var nsfwProbability = ((neighborSignal.NsfwProbability * neighborSignal.Confidence) + (modelSignal.NsfwProbability * modelSignal.Confidence)) / totalConfidence;
        var confidence = Math.Clamp(Math.Max(neighborSignal.Confidence, modelSignal.Confidence) + 0.06d, 0d, 1d);
        return new AiCorrectionSignal(
            neighborSignal.Label,
            nsfwProbability,
            confidence,
            neighborSignal.NeighborCount + modelSignal.NeighborCount,
            neighborSignal.ClosestDistance,
            neighborSignal.ManualNeighborCount + modelSignal.ManualNeighborCount,
            Math.Max(neighborSignal.SameLabelCloseNeighborCount, modelSignal.SameLabelCloseNeighborCount),
            Math.Min(neighborSignal.OppositeLabelCloseNeighborCount, modelSignal.OppositeLabelCloseNeighborCount),
            IsGlobalModel: false);
    }

    private static AiCorrectionSignal? MatchPersonalAiCorrection(AiCorrectionProfile profile, AiClassificationResult aiClassification)
    {
        if (profile.Examples.Count < PersonalAiEarlyLearningMinimumSamples)
        {
            return null;
        }

        var targetFeatures = BuildAiFeatureVector(aiClassification.LabelScores);
        var neighbors = new List<(AiCorrectionExample Example, double Distance)>(15);
        foreach (var example in GetPersonalAiCorrectionCandidates(profile, targetFeatures))
        {
            var distance = CalculateAiFeatureDistance(targetFeatures, example.Features);
            var insertIndex = neighbors.FindIndex(item => distance < item.Distance);
            if (insertIndex < 0)
            {
                if (neighbors.Count < 15)
                {
                    neighbors.Add((example, distance));
                }
            }
            else
            {
                neighbors.Insert(insertIndex, (example, distance));
                if (neighbors.Count > 15)
                {
                    neighbors.RemoveAt(neighbors.Count - 1);
                }
            }
        }

        if (neighbors.Count == 0)
        {
            return null;
        }

        var closestDistance = neighbors[0].Distance;
        var adaptiveRadius = ResolveAdaptiveAiCorrectionRadius(profile, neighbors);

        var relevantNeighbors = neighbors
            .Where(neighbor => neighbor.Distance <= adaptiveRadius)
            .ToList();
        if (relevantNeighbors.Count == 0)
        {
            return null;
        }

        var sfwScore = 0d;
        var nsfwScore = 0d;
        var manualNeighborCount = 0;
        var manualSfwScore = 0d;
        var manualNsfwScore = 0d;
        var sfwBalanceWeight = Math.Sqrt(Math.Max(1d, profile.NsfwCount) / Math.Max(1d, profile.SfwCount));
        var nsfwBalanceWeight = Math.Sqrt(Math.Max(1d, profile.SfwCount) / Math.Max(1d, profile.NsfwCount));
        foreach (var neighbor in relevantNeighbors)
        {
            var normalizedDistance = neighbor.Distance / Math.Max(0.001d, adaptiveRadius);
            var distanceAffinity = Math.Exp(-normalizedDistance * normalizedDistance);
            var balanceWeight = neighbor.Example.Label == ImageLabel.Nsfw ? nsfwBalanceWeight : sfwBalanceWeight;
            var score = neighbor.Example.Weight * Math.Clamp(balanceWeight, 0.45d, 2.2d) * distanceAffinity;

            if (neighbor.Example.Origin == LabelOrigin.ManualCorrection)
            {
                manualNeighborCount++;
            }

            if (neighbor.Example.Label == ImageLabel.Nsfw)
            {
                nsfwScore += score;
                if (neighbor.Example.Origin == LabelOrigin.ManualCorrection)
                {
                    manualNsfwScore += score;
                }
            }
            else
            {
                sfwScore += score;
                if (neighbor.Example.Origin == LabelOrigin.ManualCorrection)
                {
                    manualSfwScore += score;
                }
            }
        }

        var totalScore = sfwScore + nsfwScore;
        if (totalScore <= 0d)
        {
            return null;
        }

        var nsfwProbability = nsfwScore / totalScore;
        var label = nsfwProbability >= 0.5d ? ImageLabel.Nsfw : ImageLabel.Sfw;
        var sameLabelCloseNeighborCount = relevantNeighbors.Count(neighbor => neighbor.Example.Label == label);
        var oppositeLabelCloseNeighborCount = relevantNeighbors.Count(neighbor => neighbor.Example.Label != label);
        var userPreferenceVoteWins = manualNeighborCount > 0 && Math.Abs(nsfwProbability - 0.5d) >= 0.04d;
        if ((sameLabelCloseNeighborCount == 0 || sameLabelCloseNeighborCount <= oppositeLabelCloseNeighborCount) &&
            !userPreferenceVoteWins)
        {
            return null;
        }

        var closeness = Math.Exp(-Math.Pow(closestDistance / Math.Max(0.001d, adaptiveRadius), 2d));
        var agreement = Math.Clamp((sameLabelCloseNeighborCount - oppositeLabelCloseNeighborCount) / Math.Max(1d, relevantNeighbors.Count), 0d, 1d);
        var probabilityConfidence = Math.Abs(nsfwProbability - 0.5d) * 2d;
        var manualShare = (label == ImageLabel.Nsfw ? manualNsfwScore : manualSfwScore) / totalScore;
        var confidence = Math.Clamp((probabilityConfidence * 0.48d) + (closeness * 0.26d) + (agreement * 0.18d) + (manualShare * 0.18d), 0d, 1d);
        var minimumConfidence = manualNeighborCount > 0 ? UserPreferenceAiMinimumConfidence : PersonalAiMinimumConfidence;
        if (confidence < minimumConfidence)
        {
            return null;
        }

        return new AiCorrectionSignal(
            label,
            nsfwProbability,
            confidence,
            relevantNeighbors.Count,
            closestDistance,
            manualNeighborCount,
            sameLabelCloseNeighborCount,
            oppositeLabelCloseNeighborCount);
    }

    private static IReadOnlyList<AiCorrectionExample> GetPersonalAiCorrectionCandidates(AiCorrectionProfile profile, IReadOnlyList<double> targetFeatures)
    {
        if (profile.Examples.Count <= AiCorrectionCandidateFullScanThreshold || profile.ExamplesByScoreBucket.Count == 0)
        {
            return profile.Examples;
        }

        var targetBucket = GetAiCorrectionScoreBucket(targetFeatures);
        var candidates = new List<AiCorrectionExample>(AiCorrectionMinimumCandidateCount);
        var visitedBuckets = new bool[AiCorrectionFeatureBucketCount];
        foreach (var example in GetUserPreferenceAiCorrectionCandidates(profile, targetFeatures))
        {
            AddAiCorrectionCandidate(candidates, example);
        }

        for (var radius = 0; radius < AiCorrectionFeatureBucketCount && candidates.Count < AiCorrectionMinimumCandidateCount; radius++)
        {
            AddAiCorrectionBucketCandidates(profile, targetBucket - radius, visitedBuckets, candidates);
            if (radius > 0)
            {
                AddAiCorrectionBucketCandidates(profile, targetBucket + radius, visitedBuckets, candidates);
            }
        }

        return candidates.Count == 0 ? profile.Examples : candidates;
    }

    private static IReadOnlyList<AiCorrectionExample> GetUserPreferenceAiCorrectionCandidates(AiCorrectionProfile profile, IReadOnlyList<double> targetFeatures)
    {
        if (profile.UserPreferenceExamples.Count <= AiCorrectionCandidateFullScanThreshold ||
            profile.UserPreferenceExamplesByScoreBucket.Count == 0)
        {
            return profile.UserPreferenceExamples;
        }

        var targetBucket = GetAiCorrectionScoreBucket(targetFeatures);
        var candidates = new List<AiCorrectionExample>(AiCorrectionMinimumCandidateCount);
        var visitedBuckets = new bool[AiCorrectionFeatureBucketCount];
        for (var radius = 0; radius < AiCorrectionFeatureBucketCount && candidates.Count < AiCorrectionMinimumCandidateCount; radius++)
        {
            AddAiCorrectionBucketCandidates(
                profile.UserPreferenceExamplesByScoreBucket,
                targetBucket - radius,
                visitedBuckets,
                candidates);
            if (radius > 0)
            {
                AddAiCorrectionBucketCandidates(
                    profile.UserPreferenceExamplesByScoreBucket,
                    targetBucket + radius,
                    visitedBuckets,
                    candidates);
            }
        }

        return candidates.Count == 0 ? profile.UserPreferenceExamples : candidates;
    }

    private static Dictionary<int, List<AiCorrectionExample>> BuildAiCorrectionFeatureBuckets(IEnumerable<AiCorrectionExample> examples)
    {
        var buckets = new Dictionary<int, List<AiCorrectionExample>>();
        foreach (var example in examples)
        {
            var bucketKey = GetAiCorrectionScoreBucket(example.Features);
            AddToBucket(buckets, bucketKey, example);
        }

        return buckets;
    }

    private static void AddAiCorrectionBucketCandidates(
        AiCorrectionProfile profile,
        int bucketKey,
        bool[] visitedBuckets,
        List<AiCorrectionExample> candidates)
    {
        AddAiCorrectionBucketCandidates(profile.ExamplesByScoreBucket, bucketKey, visitedBuckets, candidates);
    }

    private static void AddAiCorrectionBucketCandidates(
        IReadOnlyDictionary<int, List<AiCorrectionExample>> buckets,
        int bucketKey,
        bool[] visitedBuckets,
        List<AiCorrectionExample> candidates)
    {
        if (bucketKey < 0 ||
            bucketKey >= AiCorrectionFeatureBucketCount ||
            visitedBuckets[bucketKey] ||
            !buckets.TryGetValue(bucketKey, out var bucket))
        {
            return;
        }

        visitedBuckets[bucketKey] = true;
        foreach (var example in bucket)
        {
            AddAiCorrectionCandidate(candidates, example);
        }
    }

    private static void AddAiCorrectionCandidate(List<AiCorrectionExample> candidates, AiCorrectionExample example)
    {
        if (!candidates.Contains(example))
        {
            candidates.Add(example);
        }
    }

    private static int GetAiCorrectionScoreBucket(IReadOnlyList<double> features)
    {
        var score = features.Count == 0 ? 0d : Math.Clamp(features[0], 0d, 1d);
        return Math.Clamp((int)Math.Floor(score * AiCorrectionFeatureBucketCount), 0, AiCorrectionFeatureBucketCount - 1);
    }

    private static double ResolveAdaptiveAiCorrectionRadius(
        AiCorrectionProfile profile,
        IReadOnlyList<(AiCorrectionExample Example, double Distance)> neighbors)
    {
        var closestDistance = neighbors.Count == 0 ? PersonalAiMaximumClosestDistance : neighbors[0].Distance;
        var localDistances = neighbors
            .Take(Math.Min(neighbors.Count, 7))
            .Select(neighbor => neighbor.Distance)
            .OrderBy(distance => distance)
            .ToList();
        var localMedian = localDistances.Count == 0
            ? closestDistance
            : localDistances[localDistances.Count / 2];

        var sameLabelDistances = profile.SameLabelFeatureDistances;
        if (sameLabelDistances.Count == 0)
        {
            return Math.Clamp(Math.Max(PersonalAiRelevantNeighborDistance, Math.Max(closestDistance * 1.8d, localMedian * 1.35d)), 0.08d, 0.34d);
        }

        var adaptiveIndex = Math.Min(sameLabelDistances.Count - 1, (int)Math.Round((sameLabelDistances.Count - 1) * 0.75d));
        var learnedRadius = sameLabelDistances[adaptiveIndex] * 1.35d;
        var localRadius = Math.Max(closestDistance * 1.8d, localMedian * 1.25d);
        return Math.Clamp(Math.Max(learnedRadius, localRadius), 0.06d, 0.36d);
    }

    private static List<double> EstimateSameLabelAiFeatureDistances(IReadOnlyList<AiCorrectionExample> examples)
    {
        var distances = new List<double>(examples.Count);
        foreach (var group in examples
                     .GroupBy(example => example.Label)
                     .Select(group => group
                         .OrderBy(example => example.Features.Length == 0 ? 0d : example.Features[0])
                         .ToList()))
        {
            for (var leftIndex = 0; leftIndex < group.Count; leftIndex++)
            {
                var nearest = double.PositiveInfinity;
                var minIndex = Math.Max(0, leftIndex - AiCorrectionDistanceEstimateWindow);
                var maxIndex = Math.Min(group.Count - 1, leftIndex + AiCorrectionDistanceEstimateWindow);
                for (var rightIndex = minIndex; rightIndex <= maxIndex; rightIndex++)
                {
                    if (leftIndex == rightIndex)
                    {
                        continue;
                    }

                    nearest = Math.Min(nearest, CalculateAiFeatureDistance(group[leftIndex].Features, group[rightIndex].Features));
                }

                if (!double.IsInfinity(nearest))
                {
                    distances.Add(nearest);
                }
            }
        }

        distances.Sort();
        return distances;
    }

    private static double ResolveDefaultNsfwThreshold(AiModelSettings settings)
    {
        var threshold = settings.DefaultNsfwThreshold <= 0.4d ? 0.62d : settings.DefaultNsfwThreshold;
        return Math.Clamp(threshold, 0.5d, 0.85d);
    }

    private static int NormalizeNudeDetectorInputSize(int inputSize)
    {
        return inputSize switch
        {
            320 or 640 => inputSize,
            <= 0 => 320,
            _ => Math.Clamp(inputSize, 256, 768),
        };
    }

    private static double GetAiExplicitNsfwScore(AiClassificationResult aiClassification)
    {
        return AiNsfwClassifierService.CalculateExplicitNsfwScore(aiClassification.LabelScores);
    }

    private static double GetAiScore(AiClassificationResult aiClassification, string label)
    {
        return AiNsfwClassifierService.GetScore(aiClassification.LabelScores, label);
    }

    private static double[] BuildAiFeatureVector(IReadOnlyDictionary<string, double> labelScores)
    {
        var explicitNsfwScore = AiNsfwClassifierService.CalculateExplicitNsfwScore(labelScores);
        var genericExplicitNsfwScore = AiNsfwClassifierService.CalculateGenericExplicitNsfwScore(labelScores);
        var nudeDetectionScore = AiNsfwClassifierService.CalculateNudeDetectionScore(labelScores);
        var weightedNsfwScore = AiNsfwClassifierService.CalculateWeightedNsfwScore(labelScores);
        var sfwScore = AiNsfwClassifierService.CalculateSfwScore(labelScores);
        return
        [
            weightedNsfwScore,
            explicitNsfwScore,
            genericExplicitNsfwScore,
            nudeDetectionScore,
            AiNsfwClassifierService.GetScore(labelScores, "nude.female_breast_exposed"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.genitalia_exposed"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.anus_exposed"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.buttocks_exposed"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.explicit_count"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.explicit_area"),
            AiNsfwClassifierService.GetScore(labelScores, "sexy"),
            AiNsfwClassifierService.GetScore(labelScores, "porn"),
            AiNsfwClassifierService.GetScore(labelScores, "hentai"),
            AiNsfwClassifierService.GetScore(labelScores, "neutral"),
            AiNsfwClassifierService.GetScore(labelScores, "drawings"),
            sfwScore,
            AiNsfwClassifierService.GetScore(labelScores, "nude.female_breast_covered"),
            AiNsfwClassifierService.GetScore(labelScores, "nude.detection_count"),
        ];
    }

    private static double Dot(IReadOnlyList<double> weights, IReadOnlyList<double> features)
    {
        var length = Math.Min(weights.Count, features.Count);
        var sum = 0d;
        for (var index = 0; index < length; index++)
        {
            sum += weights[index] * features[index];
        }

        return sum;
    }

    private static double Sigmoid(double value)
    {
        if (value >= 35d)
        {
            return 1d;
        }

        if (value <= -35d)
        {
            return 0d;
        }

        return 1d / (1d + Math.Exp(-value));
    }

    private static double CalculateAiFeatureDistance(IReadOnlyList<double> left, IReadOnlyList<double> right)
    {
        var weightedDistance = 0d;
        var totalWeight = 0d;
        var length = Math.Min(Math.Min(left.Count, right.Count), AiFeatureWeights.Length);
        for (var index = 0; index < length; index++)
        {
            var weight = AiFeatureWeights[index];
            var delta = left[index] - right[index];
            weightedDistance += delta * delta * weight;
            totalWeight += weight;
        }

        return totalWeight <= 0d ? 1d : Math.Sqrt(weightedDistance / totalWeight);
    }

    private List<(LearnedImageRecord Sample, int Distance)> FindNearestNeighbors(LearnedSampleIndex learnedIndex, ulong averageHashBits, int neighborCount)
    {
        var candidates = learnedIndex.ByPrefixByte.TryGetValue(GetHashPrefixByte(averageHashBits), out var byteBucket) ? byteBucket : learnedIndex.All;

        if (candidates.Count < SampleNeighborMinimumCandidateCount && learnedIndex.ByPrefixNibble.TryGetValue(GetHashPrefixNibble(averageHashBits), out var nibbleBucket))
        {
            candidates = nibbleBucket;
        }

        if (candidates.Count < SampleNeighborMinimumCandidateCount)
        {
            candidates = learnedIndex.All;
        }

        var neighbors = new List<(LearnedImageRecord Sample, int Distance)>(neighborCount);
        foreach (var sample in candidates)
        {
            var sampleBits = sample.AverageHashBits != 0UL ? sample.AverageHashBits : _imageFingerprintService.ParseAverageHashBits(sample.AverageHash);
            var distance = _imageFingerprintService.ComputeHammingDistance(averageHashBits, sampleBits);
            var insertIndex = neighbors.FindIndex(item => distance < item.Distance);

            if (insertIndex < 0)
            {
                if (neighbors.Count < neighborCount)
                {
                    neighbors.Add((sample, distance));
                }
            }
            else
            {
                neighbors.Insert(insertIndex, (sample, distance));
                if (neighbors.Count > neighborCount)
                {
                    neighbors.RemoveAt(neighbors.Count - 1);
                }
            }
        }

        return neighbors;
    }

    private SampleVoteSignal CalculateSampleVoteSignal(
        LearnedSampleIndex learnedIndex,
        ulong averageHashBits,
        int neighborCount,
        ImageLabel primaryLabel = ImageLabel.Sfw,
        ImageLabel secondaryLabel = ImageLabel.Nsfw,
        double secondaryBias = NsfwScoreBias)
    {
        var neighbors = FindNearestNeighbors(learnedIndex, averageHashBits, neighborCount);
        var sfwScore = 0d;
        var nsfwScore = 0d;
        var closestSfwDistance = 64;
        var closestNsfwDistance = 64;

        foreach (var neighbor in neighbors)
        {
            var sourceWeight = neighbor.Sample.LabelOrigin == LabelOrigin.ManualCorrection ? ManualCorrectionWeight : 1.0d;
            var labelWeight = neighbor.Sample.CurrentLabel == secondaryLabel ? secondaryBias : 1.0d;
            var score = (sourceWeight * labelWeight) / (1d + neighbor.Distance);
            if (neighbor.Sample.CurrentLabel == primaryLabel)
            {
                sfwScore += score;
                closestSfwDistance = Math.Min(closestSfwDistance, neighbor.Distance);
            }
            else if (neighbor.Sample.CurrentLabel == secondaryLabel)
            {
                nsfwScore += score;
                closestNsfwDistance = Math.Min(closestNsfwDistance, neighbor.Distance);
            }
        }

        return new SampleVoteSignal(sfwScore, nsfwScore, closestSfwDistance, closestNsfwDistance, neighbors.Count == 0 ? 64 : neighbors[0].Distance, neighbors.Count);
    }

    private static WorkspaceSettings CloneSettings(WorkspaceSettings source)
    {
        var sourceFolders = NormalizeWorkspaceFolderConfigs(source.SourceFolders, source.SourceFolder);
        var sfwTargetFolders = NormalizeWorkspaceFolderConfigs(source.SfwTargetFolders, source.SfwTargetFolder);
        var nsfwTargetFolders = NormalizeWorkspaceFolderConfigs(source.NsfwTargetFolders, source.NsfwTargetFolder);
        var personTaskSource = source.PersonTask ?? new PersonTaskSettings();
        var personSourceFolders = NormalizeWorkspaceFolderConfigs(personTaskSource.SourceFolders, personTaskSource.SourceFolder);
        var personTargetFolders = NormalizeWorkspaceFolderConfigs(personTaskSource.PersonTargetFolders, personTaskSource.PersonTargetFolder);
        var nonPersonTargetFolders = NormalizeWorkspaceFolderConfigs(personTaskSource.NonPersonTargetFolders, personTaskSource.NonPersonTargetFolder);

        return new WorkspaceSettings
        {
            TaskMode = source.TaskMode,
            SourceFolder = ResolveActiveWorkspaceFolder(sourceFolders),
            SfwTargetFolder = ResolveActiveWorkspaceFolder(sfwTargetFolders),
            NsfwTargetFolder = ResolveActiveWorkspaceFolder(nsfwTargetFolders),
            AiModel = new AiModelSettings
            {
                IsEnabled = source.AiModel?.IsEnabled ?? true,
                ModelPath = source.AiModel?.ModelPath ?? string.Empty,
                IsNudeDetectorEnabled = source.AiModel?.IsNudeDetectorEnabled ?? true,
                IsGpuAccelerationEnabled = source.AiModel?.IsGpuAccelerationEnabled ?? true,
                NudeDetectorModelPath = source.AiModel?.NudeDetectorModelPath ?? string.Empty,
                NudeDetectorInputSize = NormalizeNudeDetectorInputSize(source.AiModel?.NudeDetectorInputSize ?? 320),
                DefaultNsfwThreshold = ResolveDefaultNsfwThreshold(source.AiModel ?? new AiModelSettings()),
            },
            PersonalAi = ClonePersonalAiSettings(source.PersonalAi ?? new PersonalAiModelSettings(), ClassificationTaskMode.ContentSafety),
            SourceFolders = sourceFolders,
            SfwTargetFolders = sfwTargetFolders,
            NsfwTargetFolders = nsfwTargetFolders,
            SampleFolders = (source.SampleFolders ?? Enumerable.Empty<SampleFolderConfig>())
                .Select(folder => new SampleFolderConfig
                {
                    Id = folder.Id,
                    FolderPath = folder.FolderPath,
                    Label = folder.Label,
                    IsEnabled = folder.IsEnabled,
                })
                .ToList(),
            PersonTask = new PersonTaskSettings
            {
                SourceFolder = ResolveActiveWorkspaceFolder(personSourceFolders),
                PersonTargetFolder = ResolveActiveWorkspaceFolder(personTargetFolders),
                NonPersonTargetFolder = ResolveActiveWorkspaceFolder(nonPersonTargetFolders),
                SourceFolders = personSourceFolders,
                PersonTargetFolders = personTargetFolders,
                NonPersonTargetFolders = nonPersonTargetFolders,
                SampleFolders = (personTaskSource.SampleFolders ?? Enumerable.Empty<SampleFolderConfig>())
                    .Select(folder => new SampleFolderConfig
                    {
                        Id = folder.Id,
                        FolderPath = folder.FolderPath,
                        Label = folder.Label,
                        IsEnabled = folder.IsEnabled,
                    })
                    .ToList(),
                PersonalAi = ClonePersonalAiSettings(personTaskSource.PersonalAi ?? new PersonalAiModelSettings(), ClassificationTaskMode.PersonPresence),
            },
        };
    }

    private static PersonalAiModelSettings ClonePersonalAiSettings(PersonalAiModelSettings source, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        return new PersonalAiModelSettings
        {
            IsEnabled = source.IsEnabled,
            AutoTrainEnabled = source.AutoTrainEnabled,
            PythonPath = source.PythonPath,
            RootFolder = source.RootFolder,
            BaseModelId = string.IsNullOrWhiteSpace(source.BaseModelId) ||
                          string.Equals(source.BaseModelId, "google/vit-base-patch16-224-in21k", StringComparison.OrdinalIgnoreCase)
                ? "google/vit-base-patch16-224"
                : source.BaseModelId,
            MaxModelVersions = source.MaxModelVersions <= 0 ? 2 : Math.Clamp(source.MaxModelVersions, 1, 2),
            TrainEpochs = source.TrainEpochs <= 0 ? 3 : source.TrainEpochs,
            TrainBatchSize = PersonalAiTrainingService.NormalizePersonalAiBatchSize(source.TrainBatchSize),
            TriggerCorrectionCount = source.TriggerCorrectionCount <= 0 ? 50 : source.TriggerCorrectionCount,
            LearningRate = source.LearningRate <= 0d ? 0.0002d : source.LearningRate,
            DecisionThreshold = source.DecisionThreshold <= 0d ? 0.5d : source.DecisionThreshold,
            MinimumConfidence = source.MinimumConfidence <= 0d ? 0.56d : source.MinimumConfidence,
            TrainingRoots = (source.TrainingRoots ?? [])
                .Where(root => !string.IsNullOrWhiteSpace(root.FolderPath) &&
                               IsTaskLabel(root.Label, taskMode))
                .Select(root => new PersonalAiTrainingRoot
                {
                    FolderPath = NormalizePath(root.FolderPath),
                    Label = root.Label,
                    IsEnabled = root.IsEnabled,
                })
                .ToList(),
            EvaluationSamplesPerLabel = source.EvaluationSamplesPerLabel <= 0 ? 500 : Math.Clamp(source.EvaluationSamplesPerLabel, 50, 5000),
            PrimaryModelMinimumAccuracy = source.PrimaryModelMinimumAccuracy <= 0d ? 0.92d : Math.Clamp(source.PrimaryModelMinimumAccuracy, 0.5d, 0.995d),
            PrimaryModelMaximumNsfwFalseNegativeRate = source.PrimaryModelMaximumNsfwFalseNegativeRate <= 0d ? 0.04d : Math.Clamp(source.PrimaryModelMaximumNsfwFalseNegativeRate, 0d, 0.5d),
            PrimaryModelMaximumSfwFalsePositiveRate = source.PrimaryModelMaximumSfwFalsePositiveRate <= 0d ? 0.08d : Math.Clamp(source.PrimaryModelMaximumSfwFalsePositiveRate, 0d, 0.5d),
        };
    }

    internal static PersonalAiModelSettings MergeTrainingRootsFromSampleFolders(
        PersonalAiModelSettings source,
        IEnumerable<SampleFolderConfig> sampleFolders,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var merged = ClonePersonalAiSettings(source, taskMode);
        var rootsByKey = new Dictionary<string, PersonalAiTrainingRoot>(StringComparer.OrdinalIgnoreCase);
        var durableSampleFolders = SelectDurableTrainingSampleFolders(sampleFolders, taskMode);
        var hasDesktopWallpaperRoots = durableSampleFolders.Any(folder => IsLikelyDesktopWallpaperFolder(folder.FolderPath));
        var hasDedicatedTrainingRoots = HasDedicatedTrainingRootsForAllLabels(merged.TrainingRoots, taskMode);

        foreach (var root in merged.TrainingRoots.Where(root => IsTaskLabel(root.Label, taskMode)))
        {
            var normalizedPath = NormalizePath(root.FolderPath);
            if (!string.IsNullOrWhiteSpace(normalizedPath) &&
                !(hasDesktopWallpaperRoots && IsLikelyTemporaryPhoneWallpaperFolder(normalizedPath)))
            {
                rootsByKey[$"{root.Label}:{normalizedPath}"] = new PersonalAiTrainingRoot
                {
                    FolderPath = normalizedPath,
                    Label = root.Label,
                    IsEnabled = root.IsEnabled,
                };
            }
        }

        if (!hasDedicatedTrainingRoots)
        {
            foreach (var folder in durableSampleFolders)
            {
                var normalizedPath = NormalizePath(folder.FolderPath);
                rootsByKey[$"{folder.Label}:{normalizedPath}"] = new PersonalAiTrainingRoot
                {
                    FolderPath = normalizedPath,
                    Label = folder.Label,
                    IsEnabled = true,
                };
            }
        }

        merged.TrainingRoots = rootsByKey.Values
            .OrderBy(root => root.Label)
            .ThenBy(root => root.FolderPath, StringComparer.OrdinalIgnoreCase)
            .ToList();
        return merged;
    }

    internal static IReadOnlyList<SampleFolderConfig> SelectDurableTrainingSampleFolders(IEnumerable<SampleFolderConfig> sampleFolders, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var activeFolders = sampleFolders
            .Where(folder => folder.IsEnabled &&
                             IsTaskLabel(folder.Label, taskMode) &&
                             !string.IsNullOrWhiteSpace(folder.FolderPath))
            .ToList();
        var desktopWallpaperFolders = activeFolders
            .Where(folder => IsLikelyDesktopWallpaperFolder(folder.FolderPath))
            .ToList();

        return desktopWallpaperFolders.Count > 0 ? desktopWallpaperFolders : activeFolders;
    }

    private static bool HasDedicatedTrainingRootsForAllLabels(IEnumerable<PersonalAiTrainingRoot> roots, ClassificationTaskMode taskMode)
    {
        var primaryLabel = GetPrimaryLabel(taskMode);
        var secondaryLabel = GetSecondaryLabel(taskMode);
        var labels = roots
            .Where(root => root.IsEnabled &&
                           IsTaskLabel(root.Label, taskMode) &&
                           IsDedicatedTrainingFolder(root.FolderPath))
            .Select(root => root.Label)
            .Distinct()
            .ToHashSet();

        return labels.Contains(primaryLabel) && labels.Contains(secondaryLabel);
    }

    private static bool IsDedicatedTrainingFolder(string folderPath)
    {
        return !string.IsNullOrWhiteSpace(folderPath) &&
               NormalizePath(folderPath).Contains("for-training", StringComparison.OrdinalIgnoreCase);
    }

    private static void RefreshPersonalAiEvaluationSet(
        PersistedAppState state,
        WorkspaceSettings settings,
        IReadOnlyCollection<SampleFolderConfig> sampleFolders)
    {
        var taskMode = settings.TaskMode;
        var primaryLabel = GetPrimaryLabel(taskMode);
        var secondaryLabel = GetSecondaryLabel(taskMode);
        var activePersonalAiSettings = GetActivePersonalAiSettings(settings);
        var targetPerLabel = Math.Clamp(activePersonalAiSettings.EvaluationSamplesPerLabel, 50, 5000);
        var durableSampleFolders = SelectDurableTrainingSampleFolders(sampleFolders, taskMode);
        var evaluationSamplesById = GetActivePersonalAiEvaluationSamples(state, taskMode);
        var existing = evaluationSamplesById.Values
            .Where(sample => IsTaskLabel(sample.Label, taskMode) &&
                             !string.IsNullOrWhiteSpace(sample.ContentId) &&
                             IsUnderAnySampleFolder(sample.FilePath, sample.Label, durableSampleFolders))
            .GroupBy(sample => sample.Label)
            .ToDictionary(
                group => group.Key,
                group => group
                    .OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase)
                    .Take(targetPerLabel)
                    .ToList());

        var learnedSamples = state.LearnedImagesById.Values
            .Where(sample => sample.TaskMode == taskMode &&
                             IsTaskLabel(sample.CurrentLabel, taskMode) &&
                             sample.LabelOrigin == LabelOrigin.SampleLibrary &&
                             !string.IsNullOrWhiteSpace(sample.ContentId) &&
                             IsUnderAnySampleFolder(sample.LastKnownPath, sample.CurrentLabel, durableSampleFolders))
            .GroupBy(sample => sample.CurrentLabel)
            .ToDictionary(
                group => group.Key,
                group => group
                    .OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase)
                    .ToList());

        var previousKeys = evaluationSamplesById.Keys
            .OrderBy(key => key, StringComparer.OrdinalIgnoreCase)
            .ToList();
        var nextSamples = new Dictionary<string, PersonalAiEvaluationSample>(StringComparer.OrdinalIgnoreCase);
        foreach (var label in new[] { primaryLabel, secondaryLabel })
        {
            existing.TryGetValue(label, out var existingForLabel);
            learnedSamples.TryGetValue(label, out var candidatesForLabel);
            foreach (var sample in SelectEvaluationSamples(existingForLabel ?? [], candidatesForLabel ?? [], targetPerLabel))
            {
                nextSamples[BuildLearnedRecordKey(taskMode, sample.ContentId)] = sample;
            }
        }

        var nextKeys = nextSamples.Keys
            .OrderBy(key => key, StringComparer.OrdinalIgnoreCase)
            .ToList();
        var evaluationSetChanged = previousKeys.Count != nextKeys.Count ||
                                   previousKeys.Where((key, index) => !string.Equals(key, nextKeys[index], StringComparison.OrdinalIgnoreCase)).Any();

        SetActivePersonalAiEvaluationSamples(state, taskMode, nextSamples);
        if (evaluationSetChanged)
        {
            var nextSummary = new PersonalAiEvaluationSummary
            {
                TotalSamples = nextSamples.Count,
                SfwSamples = nextSamples.Values.Count(sample => sample.Label == primaryLabel),
                NsfwSamples = nextSamples.Values.Count(sample => sample.Label == secondaryLabel),
                AllowsPersonalModelPrimary = false,
                UpdatedAtUtc = DateTime.UtcNow,
            };
            SetActivePersonalAiEvaluationReport(state, taskMode, new PersonalAiEvaluationReport
            {
                Summary = nextSummary,
            });
        }
        else
        {
            var summary = GetActivePersonalAiEvaluationSummary(state, taskMode) ?? new PersonalAiEvaluationSummary();
            summary.TotalSamples = nextSamples.Count;
            summary.SfwSamples = nextSamples.Values.Count(sample => sample.Label == primaryLabel);
            summary.NsfwSamples = nextSamples.Values.Count(sample => sample.Label == secondaryLabel);
            summary.UpdatedAtUtc = DateTime.UtcNow;
            SetActivePersonalAiEvaluationSummary(state, taskMode, summary);
        }
    }

    internal static IReadOnlyList<PersonalAiEvaluationSample> SelectEvaluationSamples(
        IReadOnlyList<PersonalAiEvaluationSample> existingSamples,
        IReadOnlyList<LearnedImageRecord> candidates,
        int targetPerLabel)
    {
        var selected = new Dictionary<string, PersonalAiEvaluationSample>(StringComparer.OrdinalIgnoreCase);
        foreach (var sample in existingSamples
                     .Where(sample => !string.IsNullOrWhiteSpace(sample.ContentId))
                     .OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase)
                     .Take(targetPerLabel))
        {
            selected[sample.ContentId] = sample;
        }

        foreach (var candidate in candidates.OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase))
        {
            if (selected.Count >= targetPerLabel)
            {
                break;
            }

            if (string.IsNullOrWhiteSpace(candidate.ContentId) || selected.ContainsKey(candidate.ContentId))
            {
                continue;
            }

            selected[candidate.ContentId] = new PersonalAiEvaluationSample
            {
                ContentId = candidate.ContentId,
                FilePath = candidate.LastKnownPath,
                AverageHash = candidate.AverageHash,
                AverageHashBits = candidate.AverageHashBits,
                Width = candidate.Width,
                Height = candidate.Height,
                Label = candidate.CurrentLabel,
                Source = "sample_library",
                CreatedAtUtc = DateTime.UtcNow,
                UpdatedAtUtc = DateTime.UtcNow,
            };
        }

        return selected.Values
            .OrderBy(sample => sample.ContentId, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static bool IsUnderAnySampleFolder(
        string filePath,
        ImageLabel label,
        IEnumerable<SampleFolderConfig> sampleFolders)
    {
        return sampleFolders.Any(folder => folder.IsEnabled &&
                                           folder.Label == label &&
                                           PersonalAiTrainingService.IsPathUnderRoot(filePath, folder.FolderPath));
    }

    private static bool IsLikelyDesktopWallpaperFolder(string folderPath)
    {
        if (string.IsNullOrWhiteSpace(folderPath))
        {
            return false;
        }

        var normalizedPath = NormalizePath(folderPath);
        return normalizedPath.Contains("电脑壁纸", StringComparison.OrdinalIgnoreCase) ||
               normalizedPath.Contains("desktop wallpaper", StringComparison.OrdinalIgnoreCase) ||
               normalizedPath.Contains("desktop-wallpaper", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsLikelyTemporaryPhoneWallpaperFolder(string folderPath)
    {
        if (string.IsNullOrWhiteSpace(folderPath))
        {
            return false;
        }

        var normalizedPath = NormalizePath(folderPath);
        return normalizedPath.Contains("手机壁纸", StringComparison.OrdinalIgnoreCase) ||
               normalizedPath.Contains("phone wallpaper", StringComparison.OrdinalIgnoreCase) ||
               normalizedPath.Contains("phone-wallpaper", StringComparison.OrdinalIgnoreCase);
    }

    private static string GetActiveSourceFolder(WorkspaceSettings settings)
    {
        return settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? settings.PersonTask?.SourceFolder ?? string.Empty
            : settings.SourceFolder;
    }

    private static string GetActivePrimaryTargetFolder(WorkspaceSettings settings)
    {
        return settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? settings.PersonTask?.PersonTargetFolder ?? string.Empty
            : settings.SfwTargetFolder;
    }

    private static string GetActiveSecondaryTargetFolder(WorkspaceSettings settings)
    {
        return settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? settings.PersonTask?.NonPersonTargetFolder ?? string.Empty
            : settings.NsfwTargetFolder;
    }

    private static IEnumerable<SampleFolderConfig> GetActiveSampleFolders(WorkspaceSettings settings)
    {
        return settings.TaskMode == ClassificationTaskMode.PersonPresence
            ? settings.PersonTask?.SampleFolders ?? []
            : settings.SampleFolders;
    }

    private static PersonalAiModelSettings GetActivePersonalAiSettings(WorkspaceSettings settings)
    {
        return GetPersonalAiSettings(settings, settings.TaskMode);
    }

    private static PersonalAiModelSettings GetPersonalAiSettings(WorkspaceSettings settings, ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence
            ? settings.PersonTask?.PersonalAi ?? new PersonalAiModelSettings()
            : settings.PersonalAi ?? new PersonalAiModelSettings();
    }

    private static void SetActivePersonalAiSettings(WorkspaceSettings settings, PersonalAiModelSettings personalAiSettings)
    {
        if (settings.TaskMode == ClassificationTaskMode.PersonPresence)
        {
            settings.PersonTask ??= new PersonTaskSettings();
            settings.PersonTask.PersonalAi = personalAiSettings;
            return;
        }

        settings.PersonalAi = personalAiSettings;
    }

    private static Dictionary<string, PersonalAiEvaluationSample> GetActivePersonalAiEvaluationSamples(PersistedAppState state, ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence
            ? state.PersonAiEvaluationSamplesById
            : state.PersonalAiEvaluationSamplesById;
    }

    private static void SetActivePersonalAiEvaluationSamples(PersistedAppState state, ClassificationTaskMode taskMode, Dictionary<string, PersonalAiEvaluationSample> samples)
    {
        if (taskMode == ClassificationTaskMode.PersonPresence)
        {
            state.PersonAiEvaluationSamplesById = samples;
            return;
        }

        state.PersonalAiEvaluationSamplesById = samples;
    }

    private static PersonalAiEvaluationSummary? GetActivePersonalAiEvaluationSummary(PersistedAppState? state, ClassificationTaskMode taskMode)
    {
        if (state is null)
        {
            return null;
        }

        return taskMode == ClassificationTaskMode.PersonPresence
            ? state.PersonAiEvaluationSummary
            : state.PersonalAiEvaluationSummary;
    }

    private static void SetActivePersonalAiEvaluationSummary(PersistedAppState state, ClassificationTaskMode taskMode, PersonalAiEvaluationSummary summary)
    {
        if (taskMode == ClassificationTaskMode.PersonPresence)
        {
            state.PersonAiEvaluationSummary = summary;
            return;
        }

        state.PersonalAiEvaluationSummary = summary;
    }

    private static void SetActivePersonalAiEvaluationReport(PersistedAppState state, ClassificationTaskMode taskMode, PersonalAiEvaluationReport report)
    {
        if (taskMode == ClassificationTaskMode.PersonPresence)
        {
            state.PersonAiEvaluationReport = report;
            state.PersonAiEvaluationSummary = report.Summary;
            return;
        }

        state.PersonalAiEvaluationReport = report;
        state.PersonalAiEvaluationSummary = report.Summary;
    }

    private static ImageLabel GetPrimaryLabel(ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence ? ImageLabel.Person : ImageLabel.Sfw;
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

    private static ImageLabel GetSecondaryLabel(ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence ? ImageLabel.NonPerson : ImageLabel.Nsfw;
    }

    private static bool IsTaskLabel(ImageLabel label, ClassificationTaskMode taskMode)
    {
        return label == GetPrimaryLabel(taskMode) || label == GetSecondaryLabel(taskMode);
    }

    private static string BuildLearnedRecordKey(ClassificationTaskMode taskMode, string contentId)
    {
        return $"{taskMode}:{contentId}";
    }

    private static List<WorkspaceFolderConfig> NormalizeWorkspaceFolderConfigs(IEnumerable<WorkspaceFolderConfig>? folders, string legacyActiveFolder)
    {
        var normalizedFolders = new List<WorkspaceFolderConfig>();

        foreach (var folder in folders ?? Enumerable.Empty<WorkspaceFolderConfig>())
        {
            if (string.IsNullOrWhiteSpace(folder.FolderPath) ||
                normalizedFolders.Any(item => string.Equals(item.FolderPath, folder.FolderPath, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }

            normalizedFolders.Add(new WorkspaceFolderConfig
            {
                Id = string.IsNullOrWhiteSpace(folder.Id) ? Guid.NewGuid().ToString("N") : folder.Id,
                FolderPath = folder.FolderPath,
                IsActive = folder.IsActive,
            });
        }

        if (!string.IsNullOrWhiteSpace(legacyActiveFolder) &&
            normalizedFolders.All(item => !string.Equals(item.FolderPath, legacyActiveFolder, StringComparison.OrdinalIgnoreCase)))
        {
            normalizedFolders.Insert(0, new WorkspaceFolderConfig
            {
                FolderPath = legacyActiveFolder,
                IsActive = true,
            });
        }

        var activeFolder = !string.IsNullOrWhiteSpace(legacyActiveFolder)
            ? normalizedFolders.FirstOrDefault(item => string.Equals(item.FolderPath, legacyActiveFolder, StringComparison.OrdinalIgnoreCase))
            : normalizedFolders.FirstOrDefault(item => item.IsActive) ?? normalizedFolders.FirstOrDefault();

        foreach (var folder in normalizedFolders)
        {
            folder.IsActive = ReferenceEquals(folder, activeFolder);
        }

        return normalizedFolders;
    }

    private static string ResolveActiveWorkspaceFolder(IEnumerable<WorkspaceFolderConfig> folders)
    {
        return folders.FirstOrDefault(folder => folder.IsActive)?.FolderPath ?? string.Empty;
    }

    private static string NormalizePath(string path)
    {
        return Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar);
    }

    private IEnumerable<string> EnumerateSupportedImages(string rootFolder)
    {
        var pendingFolders = new Stack<string>();
        pendingFolders.Push(rootFolder);

        while (pendingFolders.Count > 0)
        {
            var currentFolder = pendingFolders.Pop();

            string[] subFolders;
            try
            {
                subFolders = Directory.GetDirectories(currentFolder);
            }
            catch
            {
                continue;
            }

            foreach (var subFolder in subFolders)
            {
                pendingFolders.Push(subFolder);
            }

            string[] files;
            try
            {
                files = Directory.GetFiles(currentFolder);
            }
            catch
            {
                continue;
            }

            foreach (var file in files)
            {
                if (_imageFingerprintService.IsSupportedImage(file))
                {
                    yield return file;
                }
            }
        }
    }

    private bool TryReuseFileCache(PersistedAppState state, string normalizedPath, out FileCacheRecord cacheRecord)
    {
        if (state.FileCacheByPath.TryGetValue(normalizedPath, out cacheRecord!))
        {
            var fileInfo = new FileInfo(normalizedPath);
            if (fileInfo.Exists && fileInfo.Length == cacheRecord.FileSize && fileInfo.LastWriteTimeUtc == cacheRecord.LastWriteUtc)
            {
                if (cacheRecord.AverageHashBits == 0UL && !string.IsNullOrWhiteSpace(cacheRecord.AverageHash))
                {
                    cacheRecord.AverageHashBits = _imageFingerprintService.ParseAverageHashBits(cacheRecord.AverageHash);
                }

                cacheRecord.LastSeenUtc = DateTime.UtcNow;
                return true;
            }
        }

        cacheRecord = null!;
        return false;
    }

    private static FileCacheRecord ResolveFileCache(PersistedAppState state, string filePath, string contentId, string averageHash, ulong averageHashBits, int width, int height)
    {
        if (state.FileCacheByPath.TryGetValue(filePath, out var cacheRecord))
        {
            return cacheRecord;
        }

        cacheRecord = new FileCacheRecord
        {
            FilePath = filePath,
            ContentId = contentId,
            AverageHash = averageHash,
            AverageHashBits = averageHashBits,
            Width = width,
            Height = height,
            LastSeenUtc = DateTime.UtcNow,
        };

        state.FileCacheByPath[filePath] = cacheRecord;
        return cacheRecord;
    }

    private static FileCacheRecord UpsertFileCacheRecord(PersistedAppState state, string normalizedPath, ImageFingerprint fingerprint)
    {
        var fileInfo = new FileInfo(normalizedPath);
        var cacheRecord = new FileCacheRecord
        {
            FilePath = normalizedPath,
            FileSize = fileInfo.Length,
            LastWriteUtc = fileInfo.LastWriteTimeUtc,
            ContentId = fingerprint.ContentId,
            AverageHash = fingerprint.AverageHash,
            AverageHashBits = fingerprint.AverageHashBits,
            Width = fingerprint.Width,
            Height = fingerprint.Height,
            LastSeenUtc = DateTime.UtcNow,
        };

        state.FileCacheByPath[normalizedPath] = cacheRecord;
        return cacheRecord;
    }

    private static void UpsertLearnedRecord(
        PersistedAppState state,
        FileCacheRecord cacheRecord,
        ImageLabel imageLabel,
        LabelOrigin labelOrigin,
        string reason,
        ClassificationTaskMode taskMode,
        bool incrementCorrectionCount = false)
    {
        var learnedKey = BuildLearnedRecordKey(taskMode, cacheRecord.ContentId);
        if (!state.LearnedImagesById.TryGetValue(learnedKey, out var learnedRecord) &&
            !(taskMode == ClassificationTaskMode.ContentSafety && state.LearnedImagesById.TryGetValue(cacheRecord.ContentId, out learnedRecord)))
        {
            learnedRecord = new LearnedImageRecord
            {
                ContentId = cacheRecord.ContentId,
                AverageHash = cacheRecord.AverageHash,
                AverageHashBits = cacheRecord.AverageHashBits,
                TaskMode = taskMode,
                Width = cacheRecord.Width,
                Height = cacheRecord.Height,
            };
            state.LearnedImagesById[learnedKey] = learnedRecord;
        }

        if (GetLabelOriginPriority(learnedRecord.LabelOrigin) > GetLabelOriginPriority(labelOrigin))
        {
            return;
        }

        learnedRecord.AverageHash = cacheRecord.AverageHash;
        learnedRecord.AverageHashBits = cacheRecord.AverageHashBits;
        learnedRecord.TaskMode = taskMode;
        learnedRecord.CurrentLabel = imageLabel;
        learnedRecord.LabelOrigin = labelOrigin;
        learnedRecord.LastKnownPath = cacheRecord.FilePath;
        learnedRecord.Width = cacheRecord.Width;
        learnedRecord.Height = cacheRecord.Height;
        learnedRecord.LastReason = reason;
        learnedRecord.IsHumanConfirmed = learnedRecord.IsHumanConfirmed || labelOrigin is LabelOrigin.ManualCorrection or LabelOrigin.ConfirmedMove;
        learnedRecord.UpdatedAtUtc = DateTime.UtcNow;

        if (incrementCorrectionCount)
        {
            learnedRecord.CorrectionCount++;
        }
    }

    internal static bool ShouldBackfillManualCorrectionIntoMistakeBook(LearnedImageRecord learnedRecord, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        return learnedRecord.TaskMode == taskMode &&
               learnedRecord.LabelOrigin == LabelOrigin.ManualCorrection &&
               IsTaskLabel(learnedRecord.CurrentLabel, taskMode) &&
               !string.IsNullOrWhiteSpace(learnedRecord.ContentId) &&
               !string.IsNullOrWhiteSpace(learnedRecord.LastKnownPath) &&
               File.Exists(learnedRecord.LastKnownPath);
    }

    private async Task<PersistedAppState> GetStateAsync()
    {
        _state ??= await _appStateStore.LoadAsync();
        return _state;
    }

    private Task PersistAsync()
    {
        return _state is null ? Task.CompletedTask : _appStateStore.SaveAsync(_state);
    }

    private LearnedSampleIndex BuildLearnedSampleIndex(
        PersistedAppState state,
        ClassificationTaskMode taskMode,
        PersonalAiModelSettings personalAiSettings,
        IReadOnlyCollection<SampleFolderConfig>? activeSampleFolders = null)
    {
        var all = new List<LearnedImageRecord>();
        var byContentId = new Dictionary<string, LearnedImageRecord>(StringComparer.OrdinalIgnoreCase);
        var byVisualKey = new Dictionary<string, LearnedImageRecord>(StringComparer.OrdinalIgnoreCase);
        var byPrefixByte = new Dictionary<byte, List<LearnedImageRecord>>();
        var byPrefixNibble = new Dictionary<byte, List<LearnedImageRecord>>();
        var manualCorrections = new List<LearnedImageRecord>();
        var manualCorrectionsByPrefixByte = new Dictionary<byte, List<LearnedImageRecord>>();

        foreach (var sample in state.LearnedImagesById.Values)
        {
            if (sample.TaskMode != taskMode ||
                !IsTaskLabel(sample.CurrentLabel, taskMode) ||
                !ShouldUseLearnedRecordForAnalysisIndex(sample, taskMode, personalAiSettings, activeSampleFolders))
            {
                continue;
            }

            if (sample.AverageHashBits == 0UL && !string.IsNullOrWhiteSpace(sample.AverageHash))
            {
                sample.AverageHashBits = _imageFingerprintService.ParseAverageHashBits(sample.AverageHash);
            }

            all.Add(sample);
            if (sample.LabelOrigin == LabelOrigin.ManualCorrection)
            {
                manualCorrections.Add(sample);
                AddToBucket(manualCorrectionsByPrefixByte, GetHashPrefixByte(sample.AverageHashBits), sample);
            }

            UpsertIndexRecord(byContentId, sample.ContentId, sample);
            UpsertIndexRecord(byVisualKey, BuildVisualKey(sample.AverageHashBits, sample.Width, sample.Height), sample);
            AddToBucket(byPrefixByte, GetHashPrefixByte(sample.AverageHashBits), sample);
            AddToBucket(byPrefixNibble, GetHashPrefixNibble(sample.AverageHashBits), sample);
        }

        return new LearnedSampleIndex(all, manualCorrections, byContentId, byVisualKey, byPrefixByte, byPrefixNibble, manualCorrectionsByPrefixByte);
    }

    private static bool ShouldUseLearnedRecordForAnalysisIndex(
        LearnedImageRecord learnedRecord,
        ClassificationTaskMode taskMode,
        PersonalAiModelSettings personalAiSettings,
        IReadOnlyCollection<SampleFolderConfig>? activeSampleFolders)
    {
        if (learnedRecord.LabelOrigin == LabelOrigin.ManualCorrection ||
            learnedRecord.LabelOrigin == LabelOrigin.ConfirmedMove)
        {
            return true;
        }

        if (learnedRecord.LabelOrigin == LabelOrigin.SampleLibrary &&
            activeSampleFolders is { Count: > 0 })
        {
            return IsUnderAnySampleFolder(
                learnedRecord.LastKnownPath,
                learnedRecord.CurrentLabel,
                activeSampleFolders);
        }

        return ShouldUseLearnedRecordForPersonalLearning(learnedRecord, taskMode, personalAiSettings);
    }

    internal static bool ShouldUseLearnedRecordForPersonalLearning(
        LearnedImageRecord learnedRecord,
        ClassificationTaskMode taskMode,
        PersonalAiModelSettings personalAiSettings)
    {
        if (taskMode != ClassificationTaskMode.ContentSafety ||
            learnedRecord.LabelOrigin == LabelOrigin.ManualCorrection)
        {
            return true;
        }

        if (learnedRecord.LabelOrigin is not (LabelOrigin.SampleLibrary or LabelOrigin.ConfirmedMove))
        {
            return true;
        }

        return PersonalAiTrainingService.ShouldRecordTrainingSample(
            personalAiSettings,
            learnedRecord.LastKnownPath,
            learnedRecord.CurrentLabel,
            taskMode);
    }

    private static void AddToBucket<TKey, TValue>(Dictionary<TKey, List<TValue>> index, TKey key, TValue item)
        where TKey : notnull
    {
        if (!index.TryGetValue(key, out var bucket))
        {
            bucket = [];
            index[key] = bucket;
        }

        bucket.Add(item);
    }

    private static void UpsertIndexRecord<TKey>(Dictionary<TKey, LearnedImageRecord> index, TKey key, LearnedImageRecord candidate)
        where TKey : notnull
    {
        if (!index.TryGetValue(key, out var existing) || ShouldReplaceLearnedIndexRecord(existing, candidate))
        {
            index[key] = candidate;
        }
    }

    private static bool ShouldReplaceLearnedIndexRecord(LearnedImageRecord existing, LearnedImageRecord candidate)
    {
        var existingPriority = GetLabelOriginPriority(existing.LabelOrigin);
        var candidatePriority = GetLabelOriginPriority(candidate.LabelOrigin);
        if (existingPriority != candidatePriority)
        {
            return candidatePriority > existingPriority;
        }

        return candidate.UpdatedAtUtc >= existing.UpdatedAtUtc;
    }

    private static int GetLabelOriginPriority(LabelOrigin labelOrigin)
    {
        return labelOrigin switch
        {
            LabelOrigin.ManualCorrection => 4,
            LabelOrigin.ConfirmedMove => 3,
            LabelOrigin.SampleLibrary => 2,
            LabelOrigin.PersonalAi => 1,
            LabelOrigin.AiModel => 1,
            LabelOrigin.ModelPrediction => 1,
            _ => 0,
        };
    }

    private (LearnedImageRecord Sample, int Distance)? FindNearestManualCorrection(LearnedSampleIndex learnedIndex, ulong averageHashBits, int maxDistance)
    {
        (LearnedImageRecord Sample, int Distance)? bestMatch = null;
        var candidates = GetManualCorrectionCandidates(learnedIndex, averageHashBits, maxDistance);

        foreach (var sample in candidates)
        {
            var sampleBits = sample.AverageHashBits != 0UL ? sample.AverageHashBits : _imageFingerprintService.ParseAverageHashBits(sample.AverageHash);
            var distance = _imageFingerprintService.ComputeHammingDistance(averageHashBits, sampleBits);
            if (distance > maxDistance)
            {
                continue;
            }

            if (!bestMatch.HasValue || distance < bestMatch.Value.Distance)
            {
                bestMatch = (sample, distance);
            }
        }

        return bestMatch;
    }

    private IReadOnlyList<LearnedImageRecord> GetManualCorrectionCandidates(LearnedSampleIndex learnedIndex, ulong averageHashBits, int maxDistance)
    {
        if (learnedIndex.ManualCorrections.Count <= SmallLearnedIndexFullScanThreshold)
        {
            return learnedIndex.ManualCorrections;
        }

        var prefixByte = GetHashPrefixByte(averageHashBits);
        var candidates = new List<LearnedImageRecord>();
        foreach (var bucket in learnedIndex.ManualCorrectionsByPrefixByte)
        {
            if (_imageFingerprintService.ComputeHammingDistance(prefixByte, bucket.Key) <= maxDistance)
            {
                candidates.AddRange(bucket.Value);
            }
        }

        return candidates;
    }

    private static string BuildVisualKey(ulong averageHashBits, int width, int height)
    {
        return $"{averageHashBits:X16}:{width}:{height}";
    }

    private static byte GetHashPrefixByte(ulong averageHashBits)
    {
        return (byte)(averageHashBits >> 56);
    }

    private static byte GetHashPrefixNibble(ulong averageHashBits)
    {
        return (byte)(averageHashBits >> 60);
    }

    private static int GetScanParallelism()
    {
        return Math.Clamp(Environment.ProcessorCount - 1, 2, 8);
    }

    private static bool ShouldReportProgress(int current, int total)
    {
        if (total <= 0)
        {
            return true;
        }

        if (current == total)
        {
            return true;
        }

        if (current == 1)
        {
            return true;
        }

        var interval = total switch
        {
            <= 100 => 5,
            <= 1_000 => 10,
            <= 10_000 => 50,
            _ => 100,
        };
        return current % interval == 0;
    }

    private static void ReportProgress(IProgress<WorkspaceProgressInfo>? progress, WorkspaceProgressStage stage, string title, string detail, string currentItem, int current, int total)
    {
        progress?.Report(new WorkspaceProgressInfo
        {
            Stage = stage,
            Title = title,
            Detail = detail,
            CurrentItem = currentItem,
            Current = current,
            Total = total,
        });
    }

    private sealed record LearnedSampleIndex(
        List<LearnedImageRecord> All,
        List<LearnedImageRecord> ManualCorrections,
        Dictionary<string, LearnedImageRecord> ByContentId,
        Dictionary<string, LearnedImageRecord> ByVisualKey,
        Dictionary<byte, List<LearnedImageRecord>> ByPrefixByte,
        Dictionary<byte, List<LearnedImageRecord>> ByPrefixNibble,
        Dictionary<byte, List<LearnedImageRecord>> ManualCorrectionsByPrefixByte);

    private sealed record AiPersonalCalibration(
        double DecisionThreshold,
        double ReviewMargin,
        int TrainingSampleCount,
        int ManualCorrectionCount,
        int SfwCount,
        int NsfwCount,
        bool IsPersonalized);

    private sealed record AiCorrectionProfile(
        List<AiCorrectionExample> Examples,
        List<AiCorrectionExample> UserPreferenceExamples,
        IReadOnlyList<double> SameLabelFeatureDistances,
        IReadOnlyDictionary<int, List<AiCorrectionExample>> ExamplesByScoreBucket,
        IReadOnlyDictionary<int, List<AiCorrectionExample>> UserPreferenceExamplesByScoreBucket,
        int ManualCorrectionCount,
        int ConfirmedCount,
        int SampleLibraryCount,
        int SfwCount,
        int NsfwCount)
    {
        public int TrainingSampleCount => ManualCorrectionCount + ConfirmedCount + SampleLibraryCount;
    }

    private sealed record AiCorrectionExample(
        ImageLabel Label,
        LabelOrigin Origin,
        double[] Features,
        double Weight);

    private sealed record SourceAnalysisInput(string FilePath, FileCacheRecord CacheRecord);

    private sealed record AiFeatureRequest(string ContentId, string FilePath);

    private sealed record AiFeatureExtractionSummary(int InferenceCount, int SkippedCount, long ElapsedMilliseconds);

    private sealed record PersonalAiModel(
        double[] Weights,
        double Bias,
        double DecisionThreshold,
        int TrainingSampleCount,
        int SfwCount,
        int NsfwCount,
        int ManualCorrectionCount,
        bool IsAvailable)
    {
        public static PersonalAiModel Unavailable(int trainingSampleCount, int sfwCount, int nsfwCount, int manualCorrectionCount)
        {
            return new PersonalAiModel([], 0d, 0.55d, trainingSampleCount, sfwCount, nsfwCount, manualCorrectionCount, false);
        }
    }

    private sealed record AiCorrectionSignal(
        ImageLabel Label,
        double NsfwProbability,
        double Confidence,
        int NeighborCount,
        double ClosestDistance,
        int ManualNeighborCount,
        int SameLabelCloseNeighborCount,
        int OppositeLabelCloseNeighborCount,
        bool IsGlobalModel = false);

    private sealed record SampleVoteSignal(
        double SfwScore,
        double NsfwScore,
        int ClosestSfwDistance,
        int ClosestNsfwDistance,
        int ClosestDistance,
        int NeighborCount)
    {
        public static SampleVoteSignal Empty { get; } = new(0d, 0d, 64, 64, 64, 0);
    }
}
