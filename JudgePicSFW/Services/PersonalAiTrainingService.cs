using System.Diagnostics;
using System.Globalization;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Net.Http;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class PersonalAiTrainingService
{
    private const string DefaultBaseModelId = "google/vit-base-patch16-224";
    private const int DefaultMaxModelVersions = 2;
    private const long DefaultTensorCacheMaxBytes = 1L * 1024L * 1024L * 1024L;
    private const int IncrementalReplaySampleCount = 1000;
    private const int MistakeAssetMaxPixelSize = 224;
    private const string MistakeBookManualSource = "mistake_book_manual";
    private const string HuggingFaceEndpoint = "https://huggingface.co";
    private const string DefaultPythonMirror = "https://pypi.tuna.tsinghua.edu.cn/simple";
    private const string SharedManagedEnvironmentFolderName = "PersonalAi";
    private static readonly string[] LegacyBaseModelIds = ["google/vit-base-patch16-224-in21k"];
    private static readonly string[] RequiredPythonModules = ["torch", "torchvision", "transformers", "peft", "PIL", "safetensors"];
    private static readonly (string FileName, long MinBytes)[] DefaultBaseModelFiles =
    [
        ("config.json", 512),
        ("preprocessor_config.json", 64),
        ("model.safetensors", 100_000_000),
    ];
    private static readonly string[] EmbeddedToolFiles =
    [
        "install_personal_ai_env.ps1",
        "predict_personal_lora.py",
        "requirements.txt",
        "train_personal_lora.py",
    ];
    private static readonly HttpClient SharedHttpClient = new()
    {
        Timeout = TimeSpan.FromMinutes(30),
    };

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
    private static readonly Encoding Utf8NoBom = new UTF8Encoding(false);

    private readonly PerformanceLogService _performanceLogService;
    private readonly ImageDecodeCacheService? _imageDecodeCacheService;
    private readonly SemaphoreSlim _processLock = new(1, 1);

    public PersonalAiTrainingService(
        PerformanceLogService performanceLogService,
        ImageDecodeCacheService? imageDecodeCacheService = null)
    {
        _performanceLogService = performanceLogService;
        _imageDecodeCacheService = imageDecodeCacheService;
    }

    private static PersonalAiTaskProfile GetTaskProfile(ClassificationTaskMode taskMode)
    {
        return taskMode == ClassificationTaskMode.PersonPresence
            ? new PersonalAiTaskProfile(
                ClassificationTaskMode.PersonPresence,
                ImageLabel.Person,
                ImageLabel.NonPerson,
                "人物个人模型",
                "人物 LoRA",
                "人物模型",
                "PersonAi",
                "人物",
                "非人物",
                "person",
                "non_person")
            : new PersonalAiTaskProfile(
                ClassificationTaskMode.ContentSafety,
                ImageLabel.Sfw,
                ImageLabel.Nsfw,
                "个人大模型",
                "个人 LoRA",
                "内容安全模型",
                "PersonalAi",
                "SFW",
                "NSFW",
                "sfw",
                "nsfw");
    }

    public PersonalAiTrainingStatus GetStatus(PersonalAiModelSettings? settings, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        settings ??= new PersonalAiModelSettings();
        var rootFolder = ResolveRootFolder(settings, taskMode);
        var datasetPath = ResolveDatasetPath(rootFolder);
        var activeModel = ReadActiveModelManifest(rootFolder);
        var candidateModel = ReadCandidateModelManifest(rootFolder);
        var modelVersions = ReadModelVersionInfos(rootFolder, activeModel, candidateModel);
        var checkpointState = ReadLatestCheckpointState(rootFolder);
        var samples = SelectTrainingSamplesForRequest(
                settings,
                ReadTrainingSamples(datasetPath),
                requireExistingFiles: false,
                taskMode: taskMode)
            .ToList();
        var pythonPath = ResolvePythonPath(settings);
        var environmentRootFolder = ResolveManagedEnvironmentRootFolder();
        var environmentMarker = ReadEnvironmentMarker();
        var isEnvironmentReady = !string.IsNullOrWhiteSpace(settings.PythonPath)
            ? File.Exists(pythonPath)
            : File.Exists(ResolveManagedPythonPath(environmentRootFolder)) && File.Exists(ResolveEnvironmentReadyMarkerPath(environmentRootFolder));
        var runtimeDevice = environmentMarker.TryGetValue("runtimeDevice", out var deviceText) ? deviceText : string.Empty;
        var trainingDeviceDisplay = ResolveTrainingDeviceDisplay(environmentMarker, runtimeDevice);
        var preferCudaInstall = environmentMarker.TryGetValue("preferredCudaInstall", out var preferredCudaText) &&
                                bool.TryParse(preferredCudaText, out var preferredCudaInstall) &&
                                preferredCudaInstall;
        var torchCudaVersion = environmentMarker.TryGetValue("torchCudaVersion", out var cudaVersionText) ? cudaVersionText : string.Empty;
        var cachedTensorCount = CountCachedTensorFiles(rootFolder);

        string message;
        if (!settings.IsEnabled)
        {
            message = $"{profile.DisplayName}已关闭。";
        }
        else if (activeModel is not null && candidateModel is not null)
        {
            message = $"{profile.DisplayName}当前启用 {activeModel.Version}，并且已有候选模型 {candidateModel.Version} 等待你确认启用。训练样本 {samples.Count} 张。";
        }
        else if (activeModel is not null)
        {
            message = $"{profile.DisplayName}已就绪，当前版本 {activeModel.Version}，训练样本 {samples.Count} 张。";
        }
        else if (candidateModel is not null)
        {
            message = $"{profile.DisplayName}暂未启用，但已有候选模型 {candidateModel.Version} 等待你确认启用。当前训练样本 {samples.Count} 张。";
        }
        else
        {
            message = $"{profile.DisplayName}等待训练，当前已收集 {samples.Count} 张训练样本。";
        }

        if (!isEnvironmentReady)
        {
            message += " 首次训练会自动准备本地 Python 环境。";
        }

        if (!string.IsNullOrWhiteSpace(trainingDeviceDisplay))
        {
            message += $" 当前训练环境设备：{trainingDeviceDisplay}。";
        }

        if (checkpointState is not null)
        {
            message += $" 检测到可续训进度：第 {checkpointState.Epoch} 轮，第 {checkpointState.Batch} 批。";
        }

        if (cachedTensorCount > 0)
        {
            message += $" 检测到 {cachedTensorCount} 份预处理张量缓存，当前版本会把缓存控制在约 1GB 内，超出部分按最久未使用自动淘汰。";
        }

        return new PersonalAiTrainingStatus
        {
            IsEnabled = settings.IsEnabled,
            IsEnvironmentReady = isEnvironmentReady,
            HasActiveModel = activeModel is not null,
            HasCandidateModel = candidateModel is not null,
            TrainingSampleCount = samples.Count,
            SfwSampleCount = samples.Count(item => item.Label == profile.PrimaryLabel),
            NsfwSampleCount = samples.Count(item => item.Label == profile.SecondaryLabel),
            ActiveModelVersion = activeModel?.Version ?? string.Empty,
            ActiveModelEvaluationAccuracy = activeModel?.EvaluationAccuracy,
            ActiveModelEvaluatedSamples = activeModel?.EvaluatedSamples ?? 0,
            CandidateModelVersion = candidateModel?.Version ?? string.Empty,
            CandidateModelEvaluationAccuracy = candidateModel?.EvaluationAccuracy,
            CandidateModelEvaluatedSamples = candidateModel?.EvaluatedSamples ?? 0,
            ModelVersions = modelVersions,
            RootFolder = rootFolder,
            PythonPath = pythonPath,
            TrainingDevice = runtimeDevice,
            TrainingDeviceDisplay = trainingDeviceDisplay,
            PreferCudaInstall = preferCudaInstall,
            TorchCudaVersion = torchCudaVersion,
            HasResumeCheckpoint = checkpointState is not null,
            ResumeEpoch = checkpointState?.Epoch ?? 0,
            ResumeBatch = checkpointState?.Batch ?? 0,
            CachedTensorCount = cachedTensorCount,
            Message = message,
        };
    }

    public async Task<PersonalAiTrainingStatus> ActivateCandidateModelAsync(PersonalAiModelSettings? settings, CancellationToken cancellationToken, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        var rootFolder = ResolveRootFolder(settings, taskMode);
        var candidateModel = ReadCandidateModelManifest(rootFolder)
            ?? throw new InvalidOperationException("当前没有可启用的候选模型。");

        return await ActivateModelAsync(settings, candidateModel.Version, cancellationToken, taskMode).ConfigureAwait(false);
    }

    public async Task<PersonalAiTrainingStatus> ActivateModelAsync(
        PersonalAiModelSettings? settings,
        string modelVersion,
        CancellationToken cancellationToken,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        if (string.IsNullOrWhiteSpace(modelVersion))
        {
            throw new InvalidOperationException("未指定要启用的模型版本。");
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);

        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var model = ReadModelManifestByVersion(rootFolder, modelVersion)
                ?? throw new InvalidOperationException($"找不到模型版本：{modelVersion}。");
            var modelsFolder = ResolveModelsFolder(rootFolder);
            if (!IsPathInsideDirectory(modelsFolder, model.ModelPath))
            {
                throw new InvalidOperationException("模型路径不在当前模型目录内，已拒绝切换。");
            }

            await File.WriteAllTextAsync(
                ResolveActiveModelPath(rootFolder),
                JsonSerializer.Serialize(model, JsonOptions),
                Utf8NoBom,
                cancellationToken).ConfigureAwait(false);

            if (IsPathInsideDirectory(modelsFolder, model.ModelPath))
            {
                await WriteModelManifestAsync(ResolveModelManifestPath(model.ModelPath), model, cancellationToken).ConfigureAwait(false);
            }

            var candidateManifestPath = ResolveCandidateModelPath(rootFolder);
            var candidateModel = ReadCandidateModelManifest(rootFolder);
            if (candidateModel is not null &&
                string.Equals(candidateModel.Version, model.Version, StringComparison.OrdinalIgnoreCase) &&
                File.Exists(candidateManifestPath))
            {
                File.Delete(candidateManifestPath);
            }

            PruneModelVersions(rootFolder, ResolveMaxModelVersions(settings), cancellationToken);
            return GetStatus(settings, taskMode);
        }
        finally
        {
            _processLock.Release();
        }
    }

    public async Task<bool> UpdateModelEvaluationSummaryAsync(
        PersonalAiModelSettings? settings,
        PersonalAiEvaluationSummary summary,
        CancellationToken cancellationToken,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        if (string.IsNullOrWhiteSpace(summary.ModelVersion))
        {
            return false;
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var updated = false;
            foreach (var manifestPath in EnumerateKnownModelManifestPaths(rootFolder))
            {
                cancellationToken.ThrowIfCancellationRequested();
                var manifest = ReadModelManifest(manifestPath);
                if (manifest is null ||
                    !string.Equals(manifest.Version, summary.ModelVersion, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                manifest.EvaluationAccuracy = summary.Accuracy;
                manifest.EvaluatedSamples = summary.EvaluatedSamples;
                manifest.AllowsPersonalModelPrimary = summary.AllowsPersonalModelPrimary;
                manifest.EvaluationUpdatedAtUtc = summary.UpdatedAtUtc.ToString("O", CultureInfo.InvariantCulture);
                await WriteModelManifestAsync(manifestPath, manifest, cancellationToken).ConfigureAwait(false);

                var modelsFolder = ResolveModelsFolder(rootFolder);
                if (IsPathInsideDirectory(modelsFolder, manifest.ModelPath))
                {
                    var modelManifestPath = ResolveModelManifestPath(manifest.ModelPath);
                    await WriteModelManifestAsync(modelManifestPath, manifest, cancellationToken).ConfigureAwait(false);
                }
                updated = true;
            }

            return updated;
        }
        finally
        {
            _processLock.Release();
        }
    }

    public async Task<int> PruneModelVersionsAsync(PersonalAiModelSettings? settings, CancellationToken cancellationToken, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        var rootFolder = ResolveRootFolder(settings, taskMode);
        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return PruneModelVersions(rootFolder, ResolveMaxModelVersions(settings), cancellationToken);
        }
        finally
        {
            _processLock.Release();
        }
    }

    public async Task RecordTrainingSampleAsync(
        PersonalAiModelSettings? settings,
        string filePath,
        string contentId,
        ImageLabel label,
        string source,
        CancellationToken cancellationToken,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        await RecordTrainingSamplesAsync(
            settings,
            [(filePath, contentId, label, source)],
            cancellationToken,
            taskMode).ConfigureAwait(false);
    }

    public async Task<bool> RecordManualCorrectionSampleAsync(
        PersonalAiModelSettings? settings,
        string filePath,
        string contentId,
        ImageLabel correctedLabel,
        ImageLabel previousLabel,
        LabelOrigin previousOrigin,
        CancellationToken cancellationToken,
        int correctionCount = 1,
        bool incrementExistingMistake = true,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        settings ??= new PersonalAiModelSettings();
        if (!settings.IsEnabled ||
            !IsTaskLabel(correctedLabel, profile) ||
            string.IsNullOrWhiteSpace(filePath))
        {
            return false;
        }

        string normalizedOriginalPath;
        try
        {
            normalizedOriginalPath = Path.GetFullPath(filePath);
        }
        catch
        {
            return false;
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        Directory.CreateDirectory(rootFolder);
        var processingPath = _imageDecodeCacheService is null
            ? normalizedOriginalPath
            : await _imageDecodeCacheService
                .ResolveForProcessingAsync(normalizedOriginalPath, ImageDecodeCacheService.DefaultProcessingWidth, cancellationToken)
                .ConfigureAwait(false) ?? normalizedOriginalPath;
        var assetPath = await Task.Run(
                () => TryCreateMistakeTrainingAsset(processingPath, rootFolder, contentId, correctedLabel, cancellationToken),
                cancellationToken)
            .ConfigureAwait(false);
        var trainingPath = string.IsNullOrWhiteSpace(assetPath) ? normalizedOriginalPath : assetPath;
        var now = DateTime.UtcNow;

        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var mistakeBookPath = ResolveMistakeBookPath(rootFolder);
            var mistakesByKey = ReadMistakeSamples(mistakeBookPath)
                .Where(sample => !string.IsNullOrWhiteSpace(sample.OriginalFilePath) ||
                                 !string.IsNullOrWhiteSpace(sample.ContentId))
                .GroupBy(sample => BuildTrainingSampleKey(
                    string.IsNullOrWhiteSpace(sample.OriginalFilePath) ? sample.TrainingAssetPath : sample.OriginalFilePath,
                    sample.ContentId),
                    StringComparer.OrdinalIgnoreCase)
                .ToDictionary(group => group.Key, group => group.Last(), StringComparer.OrdinalIgnoreCase);
            var mistakeKey = BuildTrainingSampleKey(normalizedOriginalPath, contentId);
            mistakesByKey.TryGetValue(mistakeKey, out var existingMistake);
            mistakesByKey[mistakeKey] = new PersonalAiMistakeSample
            {
                OriginalFilePath = normalizedOriginalPath,
                TrainingAssetPath = assetPath,
                ContentId = contentId,
                CorrectedLabel = correctedLabel,
                PreviousLabel = previousLabel,
                PreviousOrigin = previousOrigin,
                Source = MistakeBookManualSource,
                CorrectionCount = incrementExistingMistake
                    ? Math.Max(0, existingMistake?.CorrectionCount ?? 0) + 1
                    : Math.Max(1, existingMistake?.CorrectionCount ?? correctionCount),
                CreatedAtUtc = existingMistake?.CreatedAtUtc ?? now,
                UpdatedAtUtc = now,
            };

            await WriteMistakeSamplesAsync(
                    mistakeBookPath,
                    mistakesByKey.Values.OrderBy(sample => sample.UpdatedAtUtc).ToList(),
                    cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            _processLock.Release();
        }

        await RecordTrainingSamplesAsync(
                settings,
                [(trainingPath, contentId, correctedLabel, MistakeBookManualSource)],
                cancellationToken,
                taskMode)
            .ConfigureAwait(false);
        return true;
    }

    internal bool HasMistakeBookSample(
        PersonalAiModelSettings? settings,
        string filePath,
        string contentId,
        ImageLabel correctedLabel,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        if (!IsTaskLabel(correctedLabel, profile) ||
            string.IsNullOrWhiteSpace(filePath))
        {
            return false;
        }

        string normalizedOriginalPath;
        try
        {
            normalizedOriginalPath = Path.GetFullPath(filePath);
        }
        catch
        {
            return false;
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        var mistakeBookPath = ResolveMistakeBookPath(rootFolder);
        if (!File.Exists(mistakeBookPath))
        {
            return false;
        }

        var sampleKey = BuildTrainingSampleKey(normalizedOriginalPath, contentId);
        return ReadMistakeSamples(mistakeBookPath)
            .Any(sample =>
            {
                var samplePath = string.IsNullOrWhiteSpace(sample.OriginalFilePath)
                    ? sample.TrainingAssetPath
                    : sample.OriginalFilePath;
                return sample.CorrectedLabel == correctedLabel &&
                       string.Equals(BuildTrainingSampleKey(samplePath, sample.ContentId), sampleKey, StringComparison.OrdinalIgnoreCase);
            });
    }

    public async Task RecordTrainingSamplesAsync(
        PersonalAiModelSettings? settings,
        IEnumerable<(string FilePath, string ContentId, ImageLabel Label, string Source)> samples,
        CancellationToken cancellationToken,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        settings ??= new PersonalAiModelSettings();
        if (!settings.IsEnabled)
        {
            return;
        }

        var newSamples = samples
            .Where(sample => IsTaskLabel(sample.Label, profile) &&
                             !string.IsNullOrWhiteSpace(sample.FilePath) &&
                             ShouldRecordTrainingSample(settings, sample.FilePath, sample.Label, sample.Source, taskMode))
            .Select(sample => new PersonalAiTrainingSample
            {
                FilePath = Path.GetFullPath(sample.FilePath),
                ContentId = sample.ContentId,
                Label = sample.Label,
                Source = sample.Source,
                CreatedAtUtc = DateTime.UtcNow,
            })
            .ToList();
        if (newSamples.Count == 0)
        {
            return;
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        Directory.CreateDirectory(rootFolder);
        var datasetPath = ResolveDatasetPath(rootFolder);

        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var mergedSamples = new Dictionary<string, PersonalAiTrainingSample>(StringComparer.OrdinalIgnoreCase);
            foreach (var sample in ReadTrainingSamples(datasetPath)
                         .Where(sample => IsTaskLabel(sample.Label, profile) &&
                                          !string.IsNullOrWhiteSpace(sample.FilePath) &&
                                          ShouldRecordTrainingSample(settings, sample.FilePath, sample.Label, sample.Source, taskMode)))
            {
                var samplePath = Path.GetFullPath(sample.FilePath);
                var sampleKey = BuildTrainingSampleKey(samplePath, sample.ContentId);
                if (!mergedSamples.TryGetValue(sampleKey, out var existing) ||
                    GetSourcePriority(existing.Source) <= GetSourcePriority(sample.Source))
                {
                    sample.FilePath = samplePath;
                    mergedSamples[sampleKey] = sample;
                }
            }

            foreach (var sample in newSamples)
            {
                var sampleKey = BuildTrainingSampleKey(sample.FilePath, sample.ContentId);
                if (mergedSamples.TryGetValue(sampleKey, out var existingForTimestamp) &&
                    existingForTimestamp.Label == sample.Label &&
                    string.Equals(existingForTimestamp.Source, sample.Source, StringComparison.OrdinalIgnoreCase))
                {
                    sample.CreatedAtUtc = existingForTimestamp.CreatedAtUtc;
                }

                if (mergedSamples.TryGetValue(sampleKey, out var existing) &&
                    GetSourcePriority(existing.Source) > GetSourcePriority(sample.Source))
                {
                    continue;
                }

                mergedSamples[sampleKey] = sample;
            }

            var orderedSamples = mergedSamples.Values
                .OrderBy(sample => sample.FilePath, StringComparer.OrdinalIgnoreCase)
                .ToList();
            await WriteTrainingSamplesAsync(datasetPath, orderedSamples, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _processLock.Release();
        }
    }

    private async Task<IReadOnlyList<PersonalAiTrainingSample>> ResolveTrainingSamplesForProcessingAsync(
        IReadOnlyList<PersonalAiTrainingSample> samples,
        CancellationToken cancellationToken)
    {
        if (_imageDecodeCacheService is null || samples.Count == 0)
        {
            return samples;
        }

        var resolvedPaths = await _imageDecodeCacheService
            .ResolveForProcessingAsync(
                samples.Select(sample => sample.FilePath).ToList(),
                ImageDecodeCacheService.DefaultProcessingWidth,
                cancellationToken)
            .ConfigureAwait(false);
        return samples
            .Select((sample, index) => new PersonalAiTrainingSample
            {
                FilePath = resolvedPaths[index],
                ContentId = sample.ContentId,
                Label = sample.Label,
                Source = sample.Source,
                CreatedAtUtc = sample.CreatedAtUtc,
            })
            .ToList();
    }

    public async Task<PersonalAiTrainingStatus> TrainAsync(
        PersonalAiModelSettings? settings,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress = null,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        settings ??= new PersonalAiModelSettings();
        var rootFolder = ResolveRootFolder(settings, taskMode);
        var datasetPath = ResolveDatasetPath(rootFolder);
        var samples = SelectTrainingSamplesForRequest(
                settings,
                ReadTrainingSamples(datasetPath),
                requireExistingFiles: true,
                taskMode: taskMode)
            .ToList();

        if (!settings.IsEnabled)
        {
            throw new InvalidOperationException($"{profile.DisplayName}已关闭。");
        }

        if (samples.Count < 8 ||
            samples.Count(item => item.Label == profile.PrimaryLabel) < 2 ||
            samples.Count(item => item.Label == profile.SecondaryLabel) < 2)
        {
            throw new InvalidOperationException($"{profile.DisplayName}训练样本不足，至少需要 {profile.PrimaryText} 和 {profile.SecondaryText} 各 2 张，总数不少于 8 张。");
        }

        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(rootFolder);
            Directory.CreateDirectory(ResolveJobsFolder(rootFolder));
            Directory.CreateDirectory(ResolveModelsFolder(rootFolder));
            await EnsureEnvironmentAsync(settings, rootFolder, progress, cancellationToken).ConfigureAwait(false);
            var continuationModel = ResolveTrainingContinuationModel(rootFolder);
            var baseModelId = await ResolveTrainingBaseModelAsync(
                continuationModel?.BaseModelId ?? settings.BaseModelId,
                rootFolder,
                progress,
                cancellationToken).ConfigureAwait(false);

            var jobTimestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
            var filteredDatasetPath = Path.Combine(ResolveJobsFolder(rootFolder), $"train-dataset-{jobTimestamp}.jsonl");
            var processingSamples = await ResolveTrainingSamplesForProcessingAsync(samples, cancellationToken).ConfigureAwait(false);
            await WriteTrainingSamplesAsync(filteredDatasetPath, processingSamples, cancellationToken).ConfigureAwait(false);

            var requestPath = Path.Combine(ResolveJobsFolder(rootFolder), $"train-request-{jobTimestamp}.json");
            var request = new TrainingRequest
            {
                DatasetPath = filteredDatasetPath,
                RootFolder = rootFolder,
                BaseModelId = baseModelId,
                MaxModelVersions = ResolveMaxModelVersions(settings),
                Epochs = Math.Clamp(settings.TrainEpochs, 1, 20),
                BatchSize = NormalizePersonalAiBatchSize(settings.TrainBatchSize),
                LearningRate = settings.LearningRate <= 0d ? 0.0002d : settings.LearningRate,
                ContinueFromModelPath = continuationModel?.ModelPath ?? string.Empty,
                PreviousModelCreatedAtUtc = continuationModel?.CreatedAt ?? string.Empty,
                ReplaySampleCount = continuationModel is null ? 0 : IncrementalReplaySampleCount,
                EnableTensorCache = true,
                TensorCacheMaxBytes = DefaultTensorCacheMaxBytes,
                FixedReplayPathMarkers = ["for-training-"],
                PrimaryLabel = profile.PrimaryToken,
                SecondaryLabel = profile.SecondaryToken,
                PrimaryLabelValue = (int)profile.PrimaryLabel,
                SecondaryLabelValue = (int)profile.SecondaryLabel,
                VersionPrefix = profile.VersionPrefix,
            };
            await File.WriteAllTextAsync(requestPath, JsonSerializer.Serialize(request, JsonOptions), Utf8NoBom, cancellationToken).ConfigureAwait(false);

            Report(progress, $"正在训练{profile.DisplayName}", $"正在用你的样本微调 {profile.TrainingName}。首次训练会准备基础视觉模型。", $"{profile.TrainingName} 训练", 0, 1);
            var stopwatch = Stopwatch.StartNew();
            await RunPythonAsync(
                settings,
                "train_personal_lora.py",
                requestPath,
                rootFolder,
                cancellationToken,
                line => HandleTrainingOutputLine(line, progress, profile)).ConfigureAwait(false);
            stopwatch.Stop();

            var status = GetStatus(settings, taskMode);
            EnsureTrainingProducedCandidateModel(status, profile.DisplayName);
            Report(progress, $"{profile.DisplayName}训练完成", status.Message, string.IsNullOrWhiteSpace(status.CandidateModelVersion) ? status.ActiveModelVersion : status.CandidateModelVersion, 1, 1);
            await _performanceLogService.LogAsync("PersonalAiTrain", $"训练完成，耗时 {stopwatch.Elapsed.TotalSeconds:F2}s，样本 {samples.Count}，当前启用 {status.ActiveModelVersion}，候选模型 {status.CandidateModelVersion}，目录 {rootFolder}。").ConfigureAwait(false);
            return status;
        }
        finally
        {
            _processLock.Release();
        }
    }

    public async Task<IReadOnlyDictionary<string, PersonalAiPredictionResult>> PredictBatchAsync(
        PersonalAiModelSettings? settings,
        IReadOnlyList<(string ContentId, string FilePath)> requests,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress = null,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        if (!settings.IsEnabled || requests.Count == 0)
        {
            return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        var activeModel = ReadActiveModelManifest(rootFolder);
        if (activeModel is null)
        {
            return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }

        return await PredictBatchWithModelAsync(settings, rootFolder, activeModel, requests, cancellationToken, progress, taskMode).ConfigureAwait(false);
    }

    public async Task<IReadOnlyDictionary<string, PersonalAiPredictionResult>> PredictCandidateBatchAsync(
        PersonalAiModelSettings? settings,
        IReadOnlyList<(string ContentId, string FilePath)> requests,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress = null,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        if (!settings.IsEnabled || requests.Count == 0)
        {
            return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }

        var rootFolder = ResolveRootFolder(settings, taskMode);
        var candidateModel = ReadCandidateModelManifest(rootFolder) ?? ReadActiveModelManifest(rootFolder);
        if (candidateModel is null)
        {
            return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }

        return await PredictBatchWithModelAsync(settings, rootFolder, candidateModel, requests, cancellationToken, progress, taskMode).ConfigureAwait(false);
    }

    private async Task<IReadOnlyDictionary<string, PersonalAiPredictionResult>> PredictBatchWithModelAsync(
        PersonalAiModelSettings settings,
        string rootFolder,
        ActiveModelManifest modelManifest,
        IReadOnlyList<(string ContentId, string FilePath)> requests,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress,
        ClassificationTaskMode taskMode)
    {
        var profile = GetTaskProfile(taskMode);
        await _processLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            Directory.CreateDirectory(ResolveJobsFolder(rootFolder));
            var requestPath = Path.Combine(ResolveJobsFolder(rootFolder), $"predict-request-{DateTime.Now:yyyyMMdd-HHmmss-fff}.json");
            var outputPath = Path.Combine(ResolveJobsFolder(rootFolder), $"predict-output-{DateTime.Now:yyyyMMdd-HHmmss-fff}.json");
            var processingPaths = _imageDecodeCacheService is null
                ? requests.Select(item => item.FilePath).ToList()
                : await _imageDecodeCacheService
                    .ResolveForProcessingAsync(
                        requests.Select(item => item.FilePath).ToList(),
                        ImageDecodeCacheService.DefaultProcessingWidth,
                        cancellationToken)
                    .ConfigureAwait(false);
            var processingRequests = requests
                .Select((item, index) => (item.ContentId, FilePath: processingPaths[index]))
                .ToList();
            var payload = new PredictionRequest
            {
                RootFolder = rootFolder,
                ActiveModelPath = modelManifest.ModelPath,
                BaseModelId = modelManifest.BaseModelId,
                OutputPath = outputPath,
                BatchSize = NormalizePersonalAiPredictionBatchSize(settings.TrainBatchSize),
                PrimaryLabel = profile.PrimaryToken,
                SecondaryLabel = profile.SecondaryToken,
                Images = processingRequests
                    .Where(item => !string.IsNullOrWhiteSpace(item.ContentId) && File.Exists(item.FilePath))
                    .GroupBy(item => item.ContentId, StringComparer.OrdinalIgnoreCase)
                    .Select(group => group.First())
                    .Select(item => new PredictionImageRequest { ContentId = item.ContentId, FilePath = item.FilePath })
                    .ToList(),
            };

            if (payload.Images.Count == 0)
            {
                return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
            }

            await File.WriteAllTextAsync(requestPath, JsonSerializer.Serialize(payload, JsonOptions), Utf8NoBom, cancellationToken).ConfigureAwait(false);
            await RunPythonAsync(
                    settings,
                    "predict_personal_lora.py",
                    requestPath,
                    rootFolder,
                    cancellationToken,
                    line => HandlePredictionOutputLine(line, progress, profile))
                .ConfigureAwait(false);
            if (!File.Exists(outputPath))
            {
                return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
            }

            await using var outputStream = File.OpenRead(outputPath);
            var response = await JsonSerializer.DeserializeAsync<PredictionResponse>(outputStream, JsonOptions, cancellationToken).ConfigureAwait(false);
            return (response?.Predictions ?? [])
                .Where(item => !string.IsNullOrWhiteSpace(item.ContentId))
                .GroupBy(item => item.ContentId, StringComparer.OrdinalIgnoreCase)
                .Select(group => group.Last())
                .ToDictionary(
                    item => item.ContentId,
                    item => new PersonalAiPredictionResult
                    {
                        IsAvailable = item.IsAvailable,
                        ContentId = item.ContentId,
                        FilePath = item.FilePath,
                        NsfwProbability = Math.Clamp(item.NsfwProbability, 0d, 1d),
                        Confidence = Math.Clamp(item.Confidence, 0d, 1d),
                        ModelVersion = modelManifest.Version,
                        Message = item.Message,
                    },
                    StringComparer.OrdinalIgnoreCase);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            await _performanceLogService.LogAsync("PersonalAiPredict", $"预测跳过：{exception.Message}").ConfigureAwait(false);
            return new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
        }
        finally
        {
            _processLock.Release();
        }
    }

    public string ResolveRootFolder(PersonalAiModelSettings? settings, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        return ResolveRootFolder(settings, Environment.ProcessPath, AppContext.BaseDirectory, taskMode);
    }

    internal string ResolveRootFolder(PersonalAiModelSettings? settings, string? processPath, string baseDirectory, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        settings ??= new PersonalAiModelSettings();
        if (!string.IsNullOrWhiteSpace(settings.RootFolder))
        {
            return Path.GetFullPath(settings.RootFolder);
        }

        return Path.Combine(AppRuntimePaths.ResolveApplicationFolder(processPath, baseDirectory), "Data", GetTaskProfile(taskMode).DefaultRootFolderName);
    }

    internal static bool ShouldRecordTrainingSample(PersonalAiModelSettings? settings, string filePath, ImageLabel label)
    {
        return ShouldRecordTrainingSample(settings, filePath, label, source: string.Empty);
    }

    internal static bool ShouldRecordTrainingSample(PersonalAiModelSettings? settings, string filePath, ImageLabel label, ClassificationTaskMode taskMode)
    {
        return ShouldRecordTrainingSample(settings, filePath, label, source: string.Empty, taskMode: taskMode);
    }

    internal static int NormalizePersonalAiBatchSize(int batchSize)
    {
        if (batchSize <= 0 || batchSize >= 8)
        {
            return 2;
        }

        return Math.Clamp(batchSize, 1, 4);
    }

    internal static int NormalizePersonalAiPredictionBatchSize(int batchSize)
    {
        return 16;
    }

    private static bool ShouldRecordTrainingSample(PersonalAiModelSettings? settings, string filePath, ImageLabel label, string source, ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        if (!IsTaskLabel(label, profile) ||
            string.IsNullOrWhiteSpace(filePath))
        {
            return false;
        }

        if (CanBypassTrainingRootFilter(source))
        {
            return true;
        }

        var allRoots = (settings?.TrainingRoots ?? [])
            .Where(root => root.IsEnabled &&
                           IsTaskLabel(root.Label, profile) &&
                           !string.IsNullOrWhiteSpace(root.FolderPath))
            .ToList();
        if (allRoots.Count == 0)
        {
            return true;
        }

        return allRoots
            .Where(root => root.Label == label)
            .Any(root => IsPathUnderRoot(filePath, root.FolderPath));
    }

    private static bool CanBypassTrainingRootFilter(string source)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return false;
        }

        var normalizedSource = source.Trim();
        return normalizedSource.StartsWith("mistake_book", StringComparison.OrdinalIgnoreCase) ||
               normalizedSource.Contains("manual", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsTaskLabel(ImageLabel label, PersonalAiTaskProfile profile)
    {
        return label == profile.PrimaryLabel || label == profile.SecondaryLabel;
    }

    internal static bool IsPathUnderRoot(string filePath, string rootFolder)
    {
        if (string.IsNullOrWhiteSpace(filePath) || string.IsNullOrWhiteSpace(rootFolder))
        {
            return false;
        }

        try
        {
            var comparison = OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;
            var normalizedPath = Path.GetFullPath(filePath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var normalizedRoot = Path.GetFullPath(rootFolder).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return string.Equals(normalizedPath, normalizedRoot, comparison) ||
                   normalizedPath.StartsWith(normalizedRoot + Path.DirectorySeparatorChar, comparison) ||
                   normalizedPath.StartsWith(normalizedRoot + Path.AltDirectorySeparatorChar, comparison);
        }
        catch
        {
            return false;
        }
    }

    internal static string BuildTrainingSampleKey(string filePath, string contentId)
    {
        return !string.IsNullOrWhiteSpace(contentId)
            ? $"content:{contentId.Trim()}"
            : $"path:{Path.GetFullPath(filePath)}";
    }

    internal static IReadOnlyList<PersonalAiTrainingSample> SelectTrainingSamplesForRequest(
        PersonalAiModelSettings? settings,
        IEnumerable<PersonalAiTrainingSample> samples,
        bool requireExistingFiles,
        ClassificationTaskMode taskMode = ClassificationTaskMode.ContentSafety)
    {
        var profile = GetTaskProfile(taskMode);
        settings ??= new PersonalAiModelSettings();
        var mergedSamples = new Dictionary<string, PersonalAiTrainingSample>(StringComparer.OrdinalIgnoreCase);
        foreach (var sample in samples)
        {
            if (!IsTaskLabel(sample.Label, profile) ||
                string.IsNullOrWhiteSpace(sample.FilePath) ||
                !ShouldRecordTrainingSample(settings, sample.FilePath, sample.Label, sample.Source, taskMode))
            {
                continue;
            }

            string samplePath;
            try
            {
                samplePath = Path.GetFullPath(sample.FilePath);
            }
            catch
            {
                continue;
            }

            if (requireExistingFiles && !File.Exists(samplePath))
            {
                continue;
            }

            var normalizedSample = new PersonalAiTrainingSample
            {
                FilePath = samplePath,
                ContentId = sample.ContentId,
                Label = sample.Label,
                Source = sample.Source,
                CreatedAtUtc = sample.CreatedAtUtc,
            };
            var sampleKey = BuildTrainingSampleKey(normalizedSample.FilePath, normalizedSample.ContentId);
            if (!mergedSamples.TryGetValue(sampleKey, out var existing) ||
                GetSourcePriority(existing.Source) <= GetSourcePriority(normalizedSample.Source))
            {
                mergedSamples[sampleKey] = normalizedSample;
            }
        }

        return mergedSamples.Values
            .OrderBy(sample => sample.FilePath, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public static void EnsureTrainingProducedCandidateModel(PersonalAiTrainingStatus status, string modelDisplayName = "个人大模型")
    {
        if (!string.IsNullOrWhiteSpace(status.CandidateModelVersion))
        {
            return;
        }

        if (status.HasResumeCheckpoint)
        {
            throw new InvalidOperationException(
                $"训练还没有完成，当前停在第 {status.ResumeEpoch} 轮、第 {status.ResumeBatch} 批。请点击“继续断点训练{modelDisplayName}”，直到生成候选模型。");
        }

        throw new InvalidOperationException("训练结束后没有生成候选模型，请检查本地训练日志后再重试。");
    }

    private static string ResolveDatasetPath(string rootFolder)
    {
        return Path.Combine(rootFolder, "training-labels.jsonl");
    }

    private static string ResolveMistakeBookPath(string rootFolder)
    {
        return Path.Combine(rootFolder, "mistakes.jsonl");
    }

    private static string ResolveMistakeAssetsFolder(string rootFolder)
    {
        return Path.Combine(rootFolder, "mistake-assets");
    }

    private static string ResolveJobsFolder(string rootFolder)
    {
        return Path.Combine(rootFolder, "jobs");
    }

    private static string ResolveModelsFolder(string rootFolder)
    {
        return Path.Combine(rootFolder, "models");
    }

    private static string ResolveBaseModelsFolder(string rootFolder)
    {
        return Path.Combine(rootFolder, "base-models");
    }

    private static string ResolveActiveModelPath(string rootFolder)
    {
        return Path.Combine(rootFolder, "active-model.json");
    }

    private static string ResolveCandidateModelPath(string rootFolder)
    {
        return Path.Combine(rootFolder, "candidate-model.json");
    }

    private static string ResolveModelManifestPath(string modelFolder)
    {
        return Path.Combine(modelFolder, "model-manifest.json");
    }

    private static string ResolveCheckpointStatePath(string rootFolder)
    {
        return Path.Combine(rootFolder, "checkpoints", "checkpoint-latest", "checkpoint-state.json");
    }

    private static string ResolveTensorCacheFolder(string rootFolder)
    {
        return Path.Combine(rootFolder, "tensor-cache");
    }

    private static string ResolveManagedEnvironmentRootFolder()
    {
        // The task data stays isolated under PersonalAi/PersonAi, but the managed
        // Python runtime is shared because it contains no task labels or models.
        return AppRuntimePaths.ResolveDataPath(SharedManagedEnvironmentFolderName);
    }

    private static string ResolveManagedPythonPath(string environmentRootFolder)
    {
        return Path.Combine(environmentRootFolder, ".venv", "Scripts", "python.exe");
    }

    private static string ResolveEnvironmentReadyMarkerPath(string environmentRootFolder)
    {
        return Path.Combine(environmentRootFolder, ".personal-ai-env-ready");
    }

    private static Dictionary<string, string> ReadEnvironmentMarker()
    {
        var environmentRootFolder = ResolveManagedEnvironmentRootFolder();
        var markerPath = ResolveEnvironmentReadyMarkerPath(environmentRootFolder);
        if (!File.Exists(markerPath))
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in File.ReadLines(markerPath, Encoding.UTF8))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var separatorIndex = line.IndexOf('=');
            if (separatorIndex <= 0 || separatorIndex >= line.Length - 1)
            {
                continue;
            }

            var key = line[..separatorIndex].Trim();
            var value = line[(separatorIndex + 1)..].Trim();
            if (!string.IsNullOrWhiteSpace(key))
            {
                result[key] = value;
            }
        }

        return result;
    }

    private static string ResolvePythonPath(PersonalAiModelSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(settings.PythonPath))
        {
            return Path.GetFullPath(settings.PythonPath);
        }

        var environmentRootFolder = ResolveManagedEnvironmentRootFolder();
        var venvPython = ResolveManagedPythonPath(environmentRootFolder);
        return File.Exists(venvPython) ? venvPython : "python";
    }

    private static string ResolveScriptPath(string scriptName)
    {
        EnsureEmbeddedToolFiles();
        var outputPath = Path.Combine(AppRuntimePaths.ApplicationFolder, "PersonalAiTools", scriptName);
        if (File.Exists(outputPath))
        {
            return outputPath;
        }

        var sourcePath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "PersonalAiTools", scriptName));
        return File.Exists(sourcePath) ? sourcePath : outputPath;
    }

    private static void EnsureEmbeddedToolFiles()
    {
        var assembly = Assembly.GetExecutingAssembly();
        var resourceNames = assembly.GetManifestResourceNames();
        var outputFolder = Path.Combine(AppRuntimePaths.ApplicationFolder, "PersonalAiTools");
        Directory.CreateDirectory(outputFolder);

        foreach (var toolFile in EmbeddedToolFiles)
        {
            var resourceName = resourceNames.FirstOrDefault(name =>
                name.EndsWith($".PersonalAiTools.{toolFile}", StringComparison.OrdinalIgnoreCase));
            if (string.IsNullOrWhiteSpace(resourceName))
            {
                continue;
            }

            using var resourceStream = assembly.GetManifestResourceStream(resourceName);
            if (resourceStream is null)
            {
                continue;
            }

            var outputPath = Path.Combine(outputFolder, toolFile);
            using var memoryStream = new MemoryStream();
            resourceStream.CopyTo(memoryStream);
            var nextBytes = memoryStream.ToArray();
            if (File.Exists(outputPath))
            {
                var existingBytes = File.ReadAllBytes(outputPath);
                if (existingBytes.SequenceEqual(nextBytes))
                {
                    continue;
                }
            }

            File.WriteAllBytes(outputPath, nextBytes);
        }
    }

    private async Task<string> ResolveTrainingBaseModelAsync(
        string? configuredBaseModelId,
        string rootFolder,
        IProgress<WorkspaceProgressInfo>? progress,
        CancellationToken cancellationToken)
    {
        var baseModelId = NormalizeBaseModelId(configuredBaseModelId);
        if (Directory.Exists(baseModelId))
        {
            return Path.GetFullPath(baseModelId);
        }

        if (!string.Equals(baseModelId, DefaultBaseModelId, StringComparison.OrdinalIgnoreCase))
        {
            return baseModelId;
        }

        var localModelFolder = Path.Combine(ResolveBaseModelsFolder(rootFolder), "vit-base-patch16-224");
        if (IsDefaultBaseModelReady(localModelFolder))
        {
            return localModelFolder;
        }

        Directory.CreateDirectory(localModelFolder);
        Report(progress, "正在下载个人大模型基础模型", "首次训练需要缓存 ViT 视觉基础模型，后续训练会复用本地文件，不会重复下载。", "ViT 基础模型", 0, 1);
        try
        {
            foreach (var modelFile in DefaultBaseModelFiles)
            {
                await EnsureDefaultBaseModelFileAsync(localModelFolder, modelFile, cancellationToken).ConfigureAwait(false);
            }

            return localModelFolder;
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            await _performanceLogService.LogAsync("PersonalAiBaseModel", $"基础模型预下载失败，改由 Transformers 本地缓存或在线加载：{exception.Message}").ConfigureAwait(false);
            return baseModelId;
        }
    }

    private static string NormalizeBaseModelId(string? configuredBaseModelId)
    {
        var baseModelId = string.IsNullOrWhiteSpace(configuredBaseModelId)
            ? DefaultBaseModelId
            : configuredBaseModelId.Trim();
        return LegacyBaseModelIds.Contains(baseModelId, StringComparer.OrdinalIgnoreCase)
            ? DefaultBaseModelId
            : baseModelId;
    }

    private static bool IsDefaultBaseModelReady(string localModelFolder)
    {
        if (!Directory.Exists(localModelFolder))
        {
            return false;
        }

        return DefaultBaseModelFiles.All(modelFile => IsFileReady(Path.Combine(localModelFolder, modelFile.FileName), modelFile.MinBytes));
    }

    private static bool IsFileReady(string path, long minBytes)
    {
        return File.Exists(path) && new FileInfo(path).Length >= minBytes;
    }

    private static async Task EnsureDefaultBaseModelFileAsync(
        string localModelFolder,
        (string FileName, long MinBytes) modelFile,
        CancellationToken cancellationToken)
    {
        var targetPath = Path.Combine(localModelFolder, modelFile.FileName);
        if (IsFileReady(targetPath, modelFile.MinBytes))
        {
            return;
        }

        var tempPath = targetPath + ".download";
        var downloadUrl = $"{HuggingFaceEndpoint}/{DefaultBaseModelId}/resolve/main/{modelFile.FileName}";
        using var request = new HttpRequestMessage(HttpMethod.Get, downloadUrl);
        using var response = await SharedHttpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        await using (var sourceStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
        await using (var targetStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, true))
        {
            await sourceStream.CopyToAsync(targetStream, cancellationToken).ConfigureAwait(false);
        }

        if (!IsFileReady(tempPath, modelFile.MinBytes))
        {
            throw new InvalidOperationException($"基础模型文件下载不完整：{modelFile.FileName}");
        }

        File.Move(tempPath, targetPath, true);
    }

    private async Task EnsureEnvironmentAsync(
        PersonalAiModelSettings settings,
        string rootFolder,
        IProgress<WorkspaceProgressInfo>? progress,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(settings.PythonPath))
        {
            var customPythonPath = ResolvePythonPath(settings);
            if (!File.Exists(customPythonPath))
            {
                throw new FileNotFoundException("个人大模型 Python 路径不存在。", customPythonPath);
            }

            var missingModules = await GetMissingRequiredPythonModulesAsync(customPythonPath, cancellationToken).ConfigureAwait(false);
            if (missingModules.Count > 0)
            {
                throw new InvalidOperationException($"个人大模型 Python 环境缺少依赖：{string.Join("、", missingModules)}。请安装 PersonalAiTools\\requirements.txt 后再训练。");
            }

            return;
        }

        var environmentRootFolder = ResolveManagedEnvironmentRootFolder();
        var managedPythonPath = ResolveManagedPythonPath(environmentRootFolder);
        var readyMarkerPath = ResolveEnvironmentReadyMarkerPath(environmentRootFolder);
        if (File.Exists(managedPythonPath) &&
            File.Exists(readyMarkerPath) &&
            (await GetMissingRequiredPythonModulesAsync(managedPythonPath, cancellationToken).ConfigureAwait(false)).Count == 0)
        {
            return;
        }

        var installScriptPath = ResolveScriptPath("install_personal_ai_env.ps1");
        if (!File.Exists(installScriptPath))
        {
            throw new FileNotFoundException("个人大模型环境安装脚本不存在。", installScriptPath);
        }

        Report(progress, "正在准备个人大模型环境", "首次训练会创建本地 Python 环境，并通过清华镜像安装训练依赖。安装输出会实时显示在这里。", "个人 AI 环境", 0, 6);
        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            // The installer script already defaults PreferCudaTorch to $true. Passing the
            // PowerShell token "$true" through ProcessStartInfo turns it into a literal
            // string, which Windows PowerShell cannot bind to a [bool] parameter.
            Arguments = $"-NoProfile -ExecutionPolicy Bypass -File \"{installScriptPath}\" -RootFolder \"{environmentRootFolder}\" -Mirror \"{DefaultPythonMirror}\"",
            WorkingDirectory = environmentRootFolder,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        await RunProcessAsync(
            startInfo,
            "个人大模型环境安装失败",
            cancellationToken,
            line => HandleEnvironmentInstallOutputLine(line, progress),
            line => HandleEnvironmentInstallOutputLine(line, progress)).ConfigureAwait(false);

        if (!File.Exists(managedPythonPath))
        {
            throw new FileNotFoundException("个人大模型环境安装完成后没有找到 Python。", managedPythonPath);
        }

        if (!File.Exists(readyMarkerPath))
        {
            throw new FileNotFoundException("个人大模型环境安装完成后没有找到完成标记。", readyMarkerPath);
        }

        var missingAfterInstall = await GetMissingRequiredPythonModulesAsync(managedPythonPath, cancellationToken).ConfigureAwait(false);
        if (missingAfterInstall.Count > 0)
        {
            throw new InvalidOperationException($"个人大模型环境安装后仍缺少依赖：{string.Join("、", missingAfterInstall)}。");
        }
    }

    private static async Task<IReadOnlyList<string>> GetMissingRequiredPythonModulesAsync(string pythonPath, CancellationToken cancellationToken)
    {
        var moduleList = string.Join(",", RequiredPythonModules.Select(module => $"'{module}'"));
        var startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"-c \"import importlib.util,json; mods=[{moduleList}]; print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        using var process = Process.Start(startInfo) ?? throw new InvalidOperationException($"无法启动进程：{pythonPath}");
        var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        var output = await outputTask.ConfigureAwait(false);
        var error = await errorTask.ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(error) ? output.Trim() : error.Trim());
        }

        try
        {
            return JsonSerializer.Deserialize<List<string>>(output.Trim(), JsonOptions) ?? [];
        }
        catch (JsonException)
        {
            return RequiredPythonModules;
        }
    }

    private async Task RunPythonAsync(
        PersonalAiModelSettings settings,
        string scriptName,
        string requestPath,
        string rootFolder,
        CancellationToken cancellationToken,
        Action<string>? outputLineHandler = null)
    {
        var environmentRootFolder = ResolveManagedEnvironmentRootFolder();
        var managedPythonPath = ResolveManagedPythonPath(environmentRootFolder);
        var pythonPath = !string.IsNullOrWhiteSpace(settings.PythonPath)
            ? ResolvePythonPath(settings)
            : File.Exists(managedPythonPath)
                ? managedPythonPath
                : "python";
        var scriptPath = ResolveScriptPath(scriptName);
        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException("个人大模型脚本不存在。", scriptPath);
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = $"\"{scriptPath}\" \"{requestPath}\"",
            WorkingDirectory = rootFolder,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        startInfo.Environment["PYTHONUTF8"] = "1";
        var modelCacheFolder = Path.Combine(rootFolder, "huggingface");
        Directory.CreateDirectory(modelCacheFolder);
        startInfo.Environment["HF_HOME"] = modelCacheFolder;
        startInfo.Environment["HF_HUB_CACHE"] = Path.Combine(modelCacheFolder, "hub");
        startInfo.Environment["TRANSFORMERS_CACHE"] = Path.Combine(modelCacheFolder, "transformers");
        if (startInfo.Environment.TryGetValue("HF_ENDPOINT", out var endpoint) &&
            !string.IsNullOrWhiteSpace(endpoint) &&
            endpoint.Contains("hf-mirror.com", StringComparison.OrdinalIgnoreCase))
        {
            startInfo.Environment.Remove("HF_ENDPOINT");
        }

        await RunProcessAsync(startInfo, "个人大模型脚本运行失败", cancellationToken, outputLineHandler).ConfigureAwait(false);
    }

    private static async Task RunProcessAsync(
        ProcessStartInfo startInfo,
        string failureTitle,
        CancellationToken cancellationToken,
        Action<string>? outputLineHandler = null,
        Action<string>? errorLineHandler = null)
    {
        using var process = Process.Start(startInfo) ?? throw new InvalidOperationException($"无法启动进程：{startInfo.FileName}");
        var output = new StringBuilder();
        var error = new StringBuilder();
        var outputTask = ReadProcessOutputAsync(process.StandardOutput, output, outputLineHandler, cancellationToken);
        var errorTask = ReadProcessOutputAsync(process.StandardError, error, errorLineHandler, cancellationToken);
        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        await Task.WhenAll(outputTask, errorTask).ConfigureAwait(false);

        if (process.ExitCode != 0)
        {
            var outputText = output.ToString();
            var errorText = error.ToString();
            var detail = string.IsNullOrWhiteSpace(errorText) ? outputText : errorText;
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(detail)
                ? $"{failureTitle}，退出码 {process.ExitCode}。"
                : detail.Trim());
        }
    }

    internal static WorkspaceProgressInfo? CreateEnvironmentInstallProgressFromLine(string line)
    {
        const string progressPrefix = "ENV_PROGRESS ";
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }

        var trimmedLine = line.Trim();
        if (!trimmedLine.StartsWith(progressPrefix, StringComparison.Ordinal))
        {
            return null;
        }

        try
        {
            var message = JsonSerializer.Deserialize<EnvironmentInstallProgressMessage>(
                trimmedLine[progressPrefix.Length..],
                JsonOptions);
            if (message is null)
            {
                return null;
            }

            return new WorkspaceProgressInfo
            {
                Stage = WorkspaceProgressStage.TrainingPersonalModel,
                Title = "正在准备个人大模型环境",
                Detail = string.IsNullOrWhiteSpace(message.Detail) ? message.Step : message.Detail,
                CurrentItem = string.IsNullOrWhiteSpace(message.Step) ? "个人 AI 环境" : message.Step,
                Current = Math.Max(0, message.Current),
                Total = Math.Max(1, message.Total),
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    internal static WorkspaceProgressInfo? CreatePredictionProgressFromLine(string line)
    {
        return CreatePredictionProgressFromLine(line, profile: null);
    }

    private static WorkspaceProgressInfo? CreatePredictionProgressFromLine(string line, PersonalAiTaskProfile? profile)
    {
        profile ??= GetTaskProfile(ClassificationTaskMode.ContentSafety);
        const string startPrefix = "PREDICT_START ";
        const string progressPrefix = "PREDICT_PROGRESS ";
        const string donePrefix = "PREDICT_DONE ";
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }

        var trimmedLine = line.Trim();
        var title = $"正在执行{profile.DisplayName}预测";
        var payloadText = string.Empty;
        var isDone = false;
        if (trimmedLine.StartsWith(startPrefix, StringComparison.Ordinal))
        {
            payloadText = trimmedLine[startPrefix.Length..];
        }
        else if (trimmedLine.StartsWith(progressPrefix, StringComparison.Ordinal))
        {
            payloadText = trimmedLine[progressPrefix.Length..];
        }
        else if (trimmedLine.StartsWith(donePrefix, StringComparison.Ordinal))
        {
            title = $"{profile.DisplayName}预测完成";
            payloadText = trimmedLine[donePrefix.Length..];
            isDone = true;
        }
        else
        {
            return null;
        }

        try
        {
            var message = JsonSerializer.Deserialize<PredictionProgressMessage>(payloadText, JsonOptions);
            if (message is null)
            {
                return null;
            }

            var total = Math.Max(1, message.Total);
            var current = Math.Clamp(message.Current, 0, total);
            var deviceText = NormalizeDeviceDisplayText(message.DeviceDisplay, message.Device);
            var skippedText = message.Skipped > 0 ? $"，跳过 {message.Skipped} 张" : string.Empty;
            var detail = isDone
                ? $"{profile.DisplayName}已完成 {current} / {total} 张预测，设备 {deviceText}，耗时 {FormatElapsedSeconds(message.ElapsedSeconds)}{skippedText}。"
                : $"{profile.DisplayName}正在预测 {current} / {total} 张，设备 {deviceText}，已用时 {FormatElapsedSeconds(message.ElapsedSeconds)}{skippedText}。";
            return new WorkspaceProgressInfo
            {
                Stage = WorkspaceProgressStage.AnalyzingSource,
                Title = title,
                Detail = detail,
                CurrentItem = NormalizePredictionCurrentItem(message.CurrentItem),
                Current = current,
                Total = total,
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static void HandleEnvironmentInstallOutputLine(string line, IProgress<WorkspaceProgressInfo>? progress)
    {
        var parsedProgress = CreateEnvironmentInstallProgressFromLine(line);
        if (parsedProgress is not null)
        {
            progress?.Report(parsedProgress);
            return;
        }

        var trimmedLine = line.Trim();
        if (string.IsNullOrWhiteSpace(trimmedLine))
        {
            return;
        }

        Report(
            progress,
            "正在准备个人大模型环境",
            trimmedLine,
            "依赖安装输出",
            0,
            6);
    }

    private static void HandlePredictionOutputLine(string line, IProgress<WorkspaceProgressInfo>? progress, PersonalAiTaskProfile? profile = null)
    {
        profile ??= GetTaskProfile(ClassificationTaskMode.ContentSafety);
        var parsedProgress = CreatePredictionProgressFromLine(line, profile);
        if (parsedProgress is not null)
        {
            progress?.Report(parsedProgress);
        }
    }

    private static async Task ReadProcessOutputAsync(
        StreamReader reader,
        StringBuilder buffer,
        Action<string>? lineHandler,
        CancellationToken cancellationToken)
    {
        while (await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false) is { } line)
        {
            buffer.AppendLine(line);
            lineHandler?.Invoke(line);
        }
    }

    private static void HandleTrainingOutputLine(string line, IProgress<WorkspaceProgressInfo>? progress, PersonalAiTaskProfile? profile = null)
    {
        profile ??= GetTaskProfile(ClassificationTaskMode.ContentSafety);
        const string progressPrefix = "TRAIN_PROGRESS ";
        const string startPrefix = "TRAIN_START ";
        const string epochPrefix = "TRAIN_EPOCH ";

        try
        {
            if (line.StartsWith(progressPrefix, StringComparison.Ordinal))
            {
                var trainingProgress = JsonSerializer.Deserialize<TrainingProgressMessage>(
                    line[progressPrefix.Length..],
                    JsonOptions);
                if (trainingProgress is null)
                {
                    return;
                }

                var skippedText = trainingProgress.SkippedImages > 0
                    ? $"，已跳过 {trainingProgress.SkippedImages} 张坏图"
                    : string.Empty;
                var cachedText = trainingProgress.CachedSampleCount > 0
                    ? $"，预处理缓存 {trainingProgress.CachedSampleCount} 份"
                    : string.Empty;
                var resumeText = trainingProgress.ResumedFromCheckpoint
                    ? "，本次为断点续训"
                    : string.Empty;
                var deviceText = NormalizeDeviceDisplayText(trainingProgress.DeviceDisplay, trainingProgress.Device);
                Report(
                    progress,
                    $"正在训练{profile.DisplayName}",
                    $"设备 {deviceText}，第 {trainingProgress.Epoch}/{trainingProgress.Epochs} 轮，批次 {trainingProgress.Batch}/{trainingProgress.TotalBatches}，已完成训练步骤 {trainingProgress.CompletedSteps}/{trainingProgress.TotalSteps}，loss {trainingProgress.Loss:F4}，已用时 {FormatElapsedSeconds(trainingProgress.ElapsedSeconds)}{skippedText}{cachedText}{resumeText}。",
                    trainingProgress.ResumedFromCheckpoint
                        ? $"{profile.TrainingName} 续训中 · {deviceText}"
                        : $"{profile.TrainingName} 训练中 · {deviceText}",
                    trainingProgress.CompletedSteps,
                    trainingProgress.TotalSteps);
                return;
            }

            if (line.StartsWith(startPrefix, StringComparison.Ordinal))
            {
                var trainingStart = JsonSerializer.Deserialize<TrainingStartMessage>(
                    line[startPrefix.Length..],
                    JsonOptions);
                if (trainingStart is null)
                {
                    return;
                }

                var resumeText = trainingStart.ResumedFromCheckpoint
                    ? $"本次从第 {trainingStart.ResumeEpoch} 轮第 {trainingStart.ResumeBatch} 批附近继续。"
                    : "本次从头开始训练。";
                var cacheText = trainingStart.CachedSampleCount > 0
                    ? $"当前已有 {trainingStart.CachedSampleCount} 份预处理缓存。"
                    : "当前还没有可复用的预处理缓存。";
                var deviceText = NormalizeDeviceDisplayText(trainingStart.DeviceDisplay, trainingStart.Device);
                Report(
                    progress,
                    $"正在训练{profile.DisplayName}",
                    $"训练已启动：设备 {deviceText}，{trainingStart.Epochs} 轮，共 {trainingStart.TotalBatches} 个训练批次，训练样本 {trainingStart.TrainSamples} 张，验证样本 {trainingStart.ValidationSamples} 张。{resumeText} {cacheText}",
                    $"{profile.TrainingName} 准备完成 · {deviceText}",
                    0,
                    Math.Max(1, trainingStart.TotalSteps));
                return;
            }

            if (line.StartsWith(epochPrefix, StringComparison.Ordinal))
            {
                var epochProgress = JsonSerializer.Deserialize<TrainingEpochMessage>(
                    line[epochPrefix.Length..],
                    JsonOptions);
                if (epochProgress is null)
                {
                    return;
                }

                var cacheText = epochProgress.CachedSampleCount > 0
                    ? $"，预处理缓存 {epochProgress.CachedSampleCount} 份"
                    : string.Empty;
                var totalSteps = Math.Max(1, epochProgress.Epochs);
                var deviceText = NormalizeDeviceDisplayText(epochProgress.DeviceDisplay, epochProgress.Device);
                Report(
                    progress,
                    $"正在训练{profile.DisplayName}",
                    $"设备 {deviceText}，第 {epochProgress.Epoch}/{epochProgress.Epochs} 轮完成，loss {epochProgress.Loss:F4}，已用时 {FormatElapsedSeconds(epochProgress.ElapsedSeconds)}{cacheText}。",
                    $"{profile.TrainingName} 分轮完成 · {deviceText}",
                    epochProgress.Epoch,
                    totalSteps);
                return;
            }

        }
        catch (JsonException)
        {
        }
    }

    private static string FormatElapsedSeconds(double elapsedSeconds)
    {
        var elapsed = TimeSpan.FromSeconds(Math.Max(0d, elapsedSeconds));
        return elapsed.TotalHours >= 1d
            ? $"{(int)elapsed.TotalHours}小时 {elapsed.Minutes}分钟"
            : $"{elapsed.Minutes}分钟 {elapsed.Seconds}秒";
    }

    private static string NormalizeDeviceText(string device)
    {
        if (string.IsNullOrWhiteSpace(device))
        {
            return "未知设备";
        }

        return device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)
            ? "CUDA"
            : device.Equals("cpu", StringComparison.OrdinalIgnoreCase)
                ? "CPU"
                : device;
    }

    private static string NormalizePredictionCurrentItem(string currentItem)
    {
        if (string.IsNullOrWhiteSpace(currentItem))
        {
            return "个人 LoRA 主判";
        }

        return currentItem.Trim() switch
        {
            "loading-model" => "正在加载个人模型",
            "model-loaded" => "个人模型加载完成",
            "personal-lora" => "个人 LoRA 主判",
            _ => currentItem,
        };
    }

    private static string NormalizeDeviceDisplayText(string? deviceDisplay, string device)
    {
        return !string.IsNullOrWhiteSpace(deviceDisplay)
            ? deviceDisplay
            : NormalizeDeviceText(device);
    }

    private static string ResolveTrainingDeviceDisplay(
        IReadOnlyDictionary<string, string> environmentMarker,
        string runtimeDevice)
    {
        if (environmentMarker.TryGetValue("installMode", out var installMode) &&
            string.Equals(installMode, "cuda", StringComparison.OrdinalIgnoreCase))
        {
            var cudaVersion = environmentMarker.TryGetValue("torchCudaVersion", out var cudaVersionText)
                ? cudaVersionText
                : string.Empty;
            return string.IsNullOrWhiteSpace(cudaVersion)
                ? "CUDA"
                : $"CUDA (torch CUDA {cudaVersion})";
        }

        return NormalizeDeviceText(runtimeDevice);
    }

    private static string TryCreateMistakeTrainingAsset(
        string sourcePath,
        string rootFolder,
        string contentId,
        ImageLabel correctedLabel,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!File.Exists(sourcePath))
        {
            return string.Empty;
        }

        try
        {
            var assetFolder = ResolveMistakeAssetsFolder(rootFolder);
            Directory.CreateDirectory(assetFolder);
            var targetPath = Path.Combine(assetFolder, BuildMistakeAssetFileName(sourcePath, contentId, correctedLabel));
            var tempPath = targetPath + ".tmp";

            using var sourceImage = Image.FromFile(sourcePath, useEmbeddedColorManagement: false);
            var largestSide = Math.Max(sourceImage.Width, sourceImage.Height);
            if (largestSide <= 0)
            {
                return string.Empty;
            }

            var scale = Math.Min(1d, MistakeAssetMaxPixelSize / (double)largestSide);
            var targetWidth = Math.Max(1, (int)Math.Round(sourceImage.Width * scale));
            var targetHeight = Math.Max(1, (int)Math.Round(sourceImage.Height * scale));
            using var targetImage = new Bitmap(targetWidth, targetHeight, PixelFormat.Format24bppRgb);
            using (var graphics = Graphics.FromImage(targetImage))
            {
                graphics.Clear(Color.Black);
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                graphics.DrawImage(sourceImage, new Rectangle(0, 0, targetWidth, targetHeight));
            }

            SaveJpeg(targetImage, tempPath, quality: 86L);
            File.Move(tempPath, targetPath, overwrite: true);
            return targetPath;
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string BuildMistakeAssetFileName(string sourcePath, string contentId, ImageLabel correctedLabel)
    {
        var sourceKey = string.IsNullOrWhiteSpace(contentId)
            ? Path.GetFullPath(sourcePath)
            : contentId.Trim();
        var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes($"{sourceKey}|{correctedLabel}")))
            .ToLowerInvariant();
        return $"{digest[..32]}.jpg";
    }

    private static void SaveJpeg(Bitmap bitmap, string targetPath, long quality)
    {
        var jpegEncoder = ImageCodecInfo.GetImageDecoders()
            .FirstOrDefault(codec => codec.FormatID == ImageFormat.Jpeg.Guid);
        if (jpegEncoder is null)
        {
            bitmap.Save(targetPath, ImageFormat.Jpeg);
            return;
        }

        using var encoderParameters = new EncoderParameters(1);
        encoderParameters.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, Math.Clamp(quality, 1L, 100L));
        bitmap.Save(targetPath, jpegEncoder, encoderParameters);
    }

    private static List<PersonalAiTrainingSample> ReadTrainingSamples(string datasetPath)
    {
        var samples = new List<PersonalAiTrainingSample>();
        if (!File.Exists(datasetPath))
        {
            return samples;
        }

        foreach (var line in File.ReadLines(datasetPath, Encoding.UTF8))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                var sample = JsonSerializer.Deserialize<PersonalAiTrainingSample>(line, JsonOptions);
                if (sample is not null)
                {
                    samples.Add(sample);
                }
            }
            catch (JsonException)
            {
            }
        }

        return samples;
    }

    private static List<PersonalAiMistakeSample> ReadMistakeSamples(string mistakeBookPath)
    {
        var samples = new List<PersonalAiMistakeSample>();
        if (!File.Exists(mistakeBookPath))
        {
            return samples;
        }

        foreach (var line in File.ReadLines(mistakeBookPath, Encoding.UTF8))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                var sample = JsonSerializer.Deserialize<PersonalAiMistakeSample>(line, JsonOptions);
                if (sample is not null)
                {
                    samples.Add(sample);
                }
            }
            catch (JsonException)
            {
            }
        }

        return samples;
    }

    private static Task WriteTrainingSamplesAsync(
        string datasetPath,
        IReadOnlyCollection<PersonalAiTrainingSample> samples,
        CancellationToken cancellationToken)
    {
        var jsonLines = samples.Select(sample => JsonSerializer.Serialize(sample, JsonOptions));
        var content = samples.Count == 0
            ? string.Empty
            : string.Join(Environment.NewLine, jsonLines) + Environment.NewLine;
        return File.WriteAllTextAsync(datasetPath, content, Utf8NoBom, cancellationToken);
    }

    private static Task WriteMistakeSamplesAsync(
        string mistakeBookPath,
        IReadOnlyCollection<PersonalAiMistakeSample> samples,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(mistakeBookPath) ?? ".");
        var jsonLines = samples.Select(sample => JsonSerializer.Serialize(sample, JsonOptions));
        var content = samples.Count == 0
            ? string.Empty
            : string.Join(Environment.NewLine, jsonLines) + Environment.NewLine;
        return File.WriteAllTextAsync(mistakeBookPath, content, Utf8NoBom, cancellationToken);
    }

    private static int GetSourcePriority(string source)
    {
        if (string.IsNullOrWhiteSpace(source))
        {
            return 1;
        }

        if (source.StartsWith("mistake_book", StringComparison.OrdinalIgnoreCase))
        {
            return 6;
        }

        if (source.Contains("manual", StringComparison.OrdinalIgnoreCase))
        {
            return 5;
        }

        return source switch
        {
            "confirmed_batch" => 3,
            "confirmed_move" => 3,
            "sample_library" => 2,
            _ => 1,
        };
    }

    private static ActiveModelManifest? ReadActiveModelManifest(string rootFolder)
    {
        return ReadModelManifest(ResolveActiveModelPath(rootFolder));
    }

    private static ActiveModelManifest? ReadCandidateModelManifest(string rootFolder)
    {
        return ReadModelManifest(ResolveCandidateModelPath(rootFolder));
    }

    private static ActiveModelManifest? ReadModelManifestByVersion(string rootFolder, string modelVersion)
    {
        return EnumerateKnownModelManifestPaths(rootFolder)
            .Select(ReadModelManifest)
            .FirstOrDefault(manifest => manifest is not null &&
                                        string.Equals(manifest.Version, modelVersion, StringComparison.OrdinalIgnoreCase));
    }

    private static List<PersonalAiModelVersionInfo> ReadModelVersionInfos(
        string rootFolder,
        ActiveModelManifest? activeModel,
        ActiveModelManifest? candidateModel)
    {
        var activeVersion = activeModel?.Version ?? string.Empty;
        var candidateVersion = candidateModel?.Version ?? string.Empty;
        var manifests = EnumerateKnownModelManifestPaths(rootFolder)
            .Select(ReadModelManifest)
            .Where(manifest => manifest is not null && !string.IsNullOrWhiteSpace(manifest.Version))
            .Cast<ActiveModelManifest>()
            .GroupBy(manifest => manifest.Version, StringComparer.OrdinalIgnoreCase)
            .Select(group => group.First())
            .Select(manifest => new
            {
                Manifest = manifest,
                CreatedAt = ResolveManifestCreatedAtUtc(manifest, manifest.ModelPath),
            })
            .OrderByDescending(item => string.Equals(item.Manifest.Version, activeVersion, StringComparison.OrdinalIgnoreCase))
            .ThenByDescending(item => item.CreatedAt)
            .Select(item => new PersonalAiModelVersionInfo
            {
                Version = item.Manifest.Version,
                TrainingSamples = item.Manifest.TrainingSamples,
                EvaluationAccuracy = item.Manifest.EvaluationAccuracy ?? item.Manifest.ValidationAccuracy,
                EvaluatedSamples = item.Manifest.EvaluatedSamples,
                CreatedAt = item.Manifest.CreatedAt,
                IsActive = string.Equals(item.Manifest.Version, activeVersion, StringComparison.OrdinalIgnoreCase),
                IsCandidate = string.Equals(item.Manifest.Version, candidateVersion, StringComparison.OrdinalIgnoreCase),
            })
            .ToList();

        return manifests;
    }

    private static ActiveModelManifest? ResolveTrainingContinuationModel(string rootFolder)
    {
        var activeModel = ReadActiveModelManifest(rootFolder);
        return activeModel ?? ReadCandidateModelManifest(rootFolder);
    }

    private static CheckpointState? ReadLatestCheckpointState(string rootFolder)
    {
        var checkpointStatePath = ResolveCheckpointStatePath(rootFolder);
        if (!File.Exists(checkpointStatePath))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(checkpointStatePath, Encoding.UTF8);
            var checkpointState = JsonSerializer.Deserialize<CheckpointState>(json, JsonOptions);
            return checkpointState is not null && !checkpointState.IsComplete ? checkpointState : null;
        }
        catch
        {
            return null;
        }
    }

    private static int CountCachedTensorFiles(string rootFolder)
    {
        var tensorCacheFolder = ResolveTensorCacheFolder(rootFolder);
        return Directory.Exists(tensorCacheFolder)
            ? Directory.EnumerateFiles(tensorCacheFolder, "*.pt", SearchOption.TopDirectoryOnly).Count()
            : 0;
    }

    private static int ResolveMaxModelVersions(PersonalAiModelSettings settings)
    {
        return settings.MaxModelVersions <= 0
            ? DefaultMaxModelVersions
            : Math.Clamp(settings.MaxModelVersions, 1, DefaultMaxModelVersions);
    }

    private static IEnumerable<string> EnumerateKnownModelManifestPaths(string rootFolder)
    {
        yield return ResolveActiveModelPath(rootFolder);
        yield return ResolveCandidateModelPath(rootFolder);

        var modelsFolder = ResolveModelsFolder(rootFolder);
        if (!Directory.Exists(modelsFolder))
        {
            yield break;
        }

        foreach (var modelFolder in Directory.EnumerateDirectories(modelsFolder, "*", SearchOption.TopDirectoryOnly))
        {
            yield return ResolveModelManifestPath(modelFolder);
        }
    }

    private static int PruneModelVersions(string rootFolder, int maxModelVersions, CancellationToken cancellationToken)
    {
        var modelsFolder = ResolveModelsFolder(rootFolder);
        if (!Directory.Exists(modelsFolder))
        {
            var emptyRetainedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            CleanupModelPointer(ResolveActiveModelPath(rootFolder), emptyRetainedPaths, modelsFolder);
            CleanupModelPointer(ResolveCandidateModelPath(rootFolder), emptyRetainedPaths, modelsFolder);
            return 0;
        }

        var modelFolders = Directory
            .EnumerateDirectories(modelsFolder, "*", SearchOption.TopDirectoryOnly)
            .Select(Path.GetFullPath)
            .ToList();
        var manifestsByModelPath = LoadRetentionManifests(rootFolder, modelsFolder);
        var candidates = new List<ModelRetentionCandidate>(modelFolders.Count);
        foreach (var modelFolder in modelFolders)
        {
            cancellationToken.ThrowIfCancellationRequested();
            manifestsByModelPath.TryGetValue(modelFolder, out var manifest);
            candidates.Add(new ModelRetentionCandidate(
                modelFolder,
                manifest?.Version ?? Path.GetFileName(modelFolder),
                ResolveRetentionScore(manifest),
                ResolveManifestCreatedAtUtc(manifest, modelFolder)));
        }

        var retainedPaths = candidates
            .OrderByDescending(candidate => candidate.Score)
            .ThenByDescending(candidate => candidate.CreatedAtUtc)
            .Take(Math.Max(1, maxModelVersions))
            .Select(candidate => candidate.ModelPath)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var deletedCount = 0;
        foreach (var candidate in candidates.Where(candidate => !retainedPaths.Contains(candidate.ModelPath)))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!IsPathInsideDirectory(modelsFolder, candidate.ModelPath))
            {
                continue;
            }

            try
            {
                Directory.Delete(candidate.ModelPath, recursive: true);
                deletedCount++;
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        CleanupModelPointer(ResolveActiveModelPath(rootFolder), retainedPaths, modelsFolder);
        CleanupModelPointer(ResolveCandidateModelPath(rootFolder), retainedPaths, modelsFolder);
        return deletedCount;
    }

    private static Dictionary<string, ActiveModelManifest> LoadRetentionManifests(string rootFolder, string modelsFolder)
    {
        var result = new Dictionary<string, ActiveModelManifest>(StringComparer.OrdinalIgnoreCase);
        foreach (var manifestPath in EnumerateKnownModelManifestPaths(rootFolder))
        {
            var manifest = ReadModelManifest(manifestPath);
            if (manifest is null)
            {
                continue;
            }

            var modelPath = Path.GetFullPath(manifest.ModelPath);
            if (!IsPathInsideDirectory(modelsFolder, modelPath))
            {
                continue;
            }

            if (!result.TryGetValue(modelPath, out var existing) ||
                ResolveRetentionScore(manifest) > ResolveRetentionScore(existing))
            {
                result[modelPath] = manifest;
            }
        }

        return result;
    }

    private static void CleanupModelPointer(string manifestPath, HashSet<string> retainedPaths, string modelsFolder)
    {
        if (!File.Exists(manifestPath))
        {
            return;
        }

        var manifest = ReadModelManifestUnchecked(manifestPath);
        if (manifest is null)
        {
            TryDeleteFile(manifestPath);
            return;
        }

        string modelPath;
        try
        {
            modelPath = Path.GetFullPath(manifest.ModelPath);
        }
        catch
        {
            TryDeleteFile(manifestPath);
            return;
        }

        if (!Directory.Exists(modelPath) ||
            !IsPathInsideDirectory(modelsFolder, modelPath) ||
            !retainedPaths.Contains(modelPath))
        {
            TryDeleteFile(manifestPath);
        }
    }

    private static bool IsPathInsideDirectory(string parentFolder, string childPath)
    {
        var normalizedParent = Path.GetFullPath(parentFolder).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        var normalizedChild = Path.GetFullPath(childPath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        return normalizedChild.StartsWith(normalizedParent, StringComparison.OrdinalIgnoreCase);
    }

    private static double ResolveRetentionScore(ActiveModelManifest? manifest)
    {
        return manifest?.EvaluationAccuracy ??
               manifest?.ValidationAccuracy ??
               double.NegativeInfinity;
    }

    private static DateTime ResolveManifestCreatedAtUtc(ActiveModelManifest? manifest, string modelFolder)
    {
        if (manifest is not null &&
            DateTime.TryParse(
                manifest.CreatedAt,
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                out var createdAt))
        {
            return createdAt.ToUniversalTime();
        }

        return Directory.Exists(modelFolder) ? Directory.GetCreationTimeUtc(modelFolder) : DateTime.MinValue;
    }

    private static void TryDeleteFile(string filePath)
    {
        try
        {
            File.Delete(filePath);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private static async Task WriteModelManifestAsync(string manifestPath, ActiveModelManifest manifest, CancellationToken cancellationToken)
    {
        var parentFolder = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrWhiteSpace(parentFolder))
        {
            Directory.CreateDirectory(parentFolder);
        }

        await File.WriteAllTextAsync(
                manifestPath,
                JsonSerializer.Serialize(manifest, JsonOptions),
                Utf8NoBom,
                cancellationToken)
            .ConfigureAwait(false);
    }

    private static ActiveModelManifest? ReadModelManifestUnchecked(string manifestPath)
    {
        if (!File.Exists(manifestPath))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(manifestPath, Encoding.UTF8);
            return JsonSerializer.Deserialize<ActiveModelManifest>(json, JsonOptions);
        }
        catch
        {
            return null;
        }
    }

    private static ActiveModelManifest? ReadModelManifest(string manifestPath)
    {
        var manifest = ReadModelManifestUnchecked(manifestPath);
        return manifest is not null && Directory.Exists(manifest.ModelPath) ? manifest : null;
    }

    private static void Report(
        IProgress<WorkspaceProgressInfo>? progress,
        string title,
        string detail,
        string currentItem,
        int current,
        int total)
    {
        progress?.Report(new WorkspaceProgressInfo
        {
            Stage = WorkspaceProgressStage.TrainingPersonalModel,
            Title = title,
            Detail = detail,
            CurrentItem = currentItem,
            Current = current,
            Total = total,
        });
    }

    private sealed class TrainingRequest
    {
        public string DatasetPath { get; init; } = string.Empty;
        public string RootFolder { get; init; } = string.Empty;
        public string BaseModelId { get; init; } = string.Empty;
        public string ContinueFromModelPath { get; init; } = string.Empty;
        public string PreviousModelCreatedAtUtc { get; init; } = string.Empty;
        public int ReplaySampleCount { get; init; }
        public int MaxModelVersions { get; init; }
        public int Epochs { get; init; }
        public int BatchSize { get; init; }
        public double LearningRate { get; init; }
        public bool EnableTensorCache { get; init; }
        public long TensorCacheMaxBytes { get; init; }
        public List<string> FixedReplayPathMarkers { get; init; } = [];
        public string PrimaryLabel { get; init; } = "sfw";
        public string SecondaryLabel { get; init; } = "nsfw";
        public int PrimaryLabelValue { get; init; } = (int)ImageLabel.Sfw;
        public int SecondaryLabelValue { get; init; } = (int)ImageLabel.Nsfw;
        public string VersionPrefix { get; init; } = "personal-lora";
    }

    private sealed class TrainingStartMessage
    {
        public int Epochs { get; init; }
        public int TotalBatches { get; init; }
        public int TotalSteps { get; init; }
        public int TrainSamples { get; init; }
        public int ValidationSamples { get; init; }
        public string Device { get; init; } = string.Empty;
        public string DeviceDisplay { get; init; } = string.Empty;
        public bool ResumedFromCheckpoint { get; init; }
        public int ResumeEpoch { get; init; }
        public int ResumeBatch { get; init; }
        public int WorkerCount { get; init; }
        public int CachedSampleCount { get; init; }
    }

    private sealed class TrainingProgressMessage
    {
        public int Epoch { get; init; }
        public int Epochs { get; init; }
        public int Batch { get; init; }
        public int TotalBatches { get; init; }
        public int CompletedSteps { get; init; }
        public int TotalSteps { get; init; }
        public double Loss { get; init; }
        public int SkippedImages { get; init; }
        public double ElapsedSeconds { get; init; }
        public string Device { get; init; } = string.Empty;
        public string DeviceDisplay { get; init; } = string.Empty;
        public bool ResumedFromCheckpoint { get; init; }
        public int WorkerCount { get; init; }
        public int CachedSampleCount { get; init; }
    }

    private sealed class TrainingEpochMessage
    {
        public int Epoch { get; init; }
        public int Epochs { get; init; }
        public double Loss { get; init; }
        public int SkippedImages { get; init; }
        public double ElapsedSeconds { get; init; }
        public string Device { get; init; } = string.Empty;
        public string DeviceDisplay { get; init; } = string.Empty;
        public bool ResumedFromCheckpoint { get; init; }
        public int CachedSampleCount { get; init; }
    }

    private sealed class PredictionProgressMessage
    {
        public int Current { get; init; }
        public int Total { get; init; }
        public int Skipped { get; init; }
        public string CurrentItem { get; init; } = string.Empty;
        public string Device { get; init; } = string.Empty;
        public string DeviceDisplay { get; init; } = string.Empty;
        public double ElapsedSeconds { get; init; }
    }

    private sealed class EnvironmentInstallProgressMessage
    {
        public int Current { get; init; }
        public int Total { get; init; }
        public string Step { get; init; } = string.Empty;
        public string Detail { get; init; } = string.Empty;
    }

    private sealed class PredictionRequest
    {
        public string RootFolder { get; init; } = string.Empty;
        public string ActiveModelPath { get; init; } = string.Empty;
        public string BaseModelId { get; init; } = string.Empty;
        public string OutputPath { get; init; } = string.Empty;
        public int BatchSize { get; init; } = 16;
        public string PrimaryLabel { get; init; } = "sfw";
        public string SecondaryLabel { get; init; } = "nsfw";
        public List<PredictionImageRequest> Images { get; init; } = [];
    }

    private sealed class PredictionImageRequest
    {
        public string ContentId { get; init; } = string.Empty;
        public string FilePath { get; init; } = string.Empty;
    }

    private sealed class PredictionResponse
    {
        public List<PredictionItem> Predictions { get; init; } = [];
    }

    private sealed class PredictionItem
    {
        public bool IsAvailable { get; init; }
        public string ContentId { get; init; } = string.Empty;
        public string FilePath { get; init; } = string.Empty;
        public double NsfwProbability { get; init; }
        public double Confidence { get; init; }
        public string Message { get; init; } = string.Empty;
    }

    private sealed class ActiveModelManifest
    {
        public string Version { get; set; } = string.Empty;
        public string ModelPath { get; set; } = string.Empty;
        public string BaseModelId { get; set; } = string.Empty;
        public string CreatedAt { get; set; } = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);
        public int TrainingSamples { get; set; }
        public int SelectedTrainingSamples { get; set; }
        public int SfwSamples { get; set; }
        public int NsfwSamples { get; set; }
        public double? ValidationAccuracy { get; set; }
        public double? EvaluationAccuracy { get; set; }
        public int EvaluatedSamples { get; set; }
        public bool AllowsPersonalModelPrimary { get; set; }
        public string EvaluationUpdatedAtUtc { get; set; } = string.Empty;
    }

    private sealed record ModelRetentionCandidate(string ModelPath, string Version, double Score, DateTime CreatedAtUtc);

    private sealed record PersonalAiTaskProfile(
        ClassificationTaskMode TaskMode,
        ImageLabel PrimaryLabel,
        ImageLabel SecondaryLabel,
        string DisplayName,
        string TrainingName,
        string StatusName,
        string DefaultRootFolderName,
        string PrimaryText,
        string SecondaryText,
        string PrimaryToken,
        string SecondaryToken)
    {
        public string VersionPrefix => TaskMode == ClassificationTaskMode.PersonPresence ? "person-lora" : "personal-lora";
    }

    private sealed class CheckpointState
    {
        public int Epoch { get; init; }
        public int Batch { get; init; }
        public bool IsComplete { get; init; }
    }
}
