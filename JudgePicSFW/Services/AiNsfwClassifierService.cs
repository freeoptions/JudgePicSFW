using System.Buffers;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class AiNsfwClassifierService : IDisposable
{
    public const string DefaultModelId = "onnx-community/nsfw-image-detector-ONNX:model_quantized.onnx";
    public const string NudeDetectorModelId = "notAI-tech/NudeNet:320n.onnx";
    public const string ScoringModelId = DefaultModelId + "+" + NudeDetectorModelId + ":explicit-nsfw-v6";

    private const int DefaultClassifierInputSize = 224;
    private const int DefaultNudeDetectorInputSize = 320;
    private const double NudeDetectionConfidenceThreshold = 0.25d;
    private const double NudeDetectionNmsThreshold = 0.45d;
    private const string NudeDetectorWheelEntryName = "nudenet/320n.onnx";
    private static readonly string[] DefaultLabels = ["drawings", "hentai", "neutral", "porn", "sexy"];
    private static readonly string[] NudeDetectionLabels =
    [
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    ];

    private static readonly string[] DefaultDownloadUrls =
    [
        "https://huggingface.co/onnx-community/nsfw-image-detector-ONNX/resolve/main/onnx/model_quantized.onnx?download=true",
        "https://hf-mirror.com/onnx-community/nsfw-image-detector-ONNX/resolve/main/onnx/model_quantized.onnx?download=true",
    ];

    private static readonly string[] NudeDetectorOnnxDownloadUrls =
    [
        "https://hf-mirror.com/zhangsongbo365/nudenet_onnx/resolve/main/320n.onnx?download=true",
        "https://huggingface.co/zhangsongbo365/nudenet_onnx/resolve/main/320n.onnx?download=true",
    ];

    private static readonly string[] NudeDetectorWheelDownloadUrls =
    [
        "https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/nudenet-3.4.2-py3-none-any.whl",
    ];

    private readonly PerformanceLogService? _performanceLogService;
    private readonly ImageDecodeCacheService? _imageDecodeCacheService;
    private readonly object _classifierSessionLock = new();
    private readonly object _nudeDetectorSessionLock = new();
    private readonly SemaphoreSlim _runSemaphore = new(1, 1);
    private InferenceSession? _classifierSession;
    private InferenceSession? _nudeDetectorSession;
    private string _loadedClassifierModelPath = string.Empty;
    private string _loadedNudeDetectorModelPath = string.Empty;
    private bool _loadedClassifierGpuAccelerationEnabled;
    private bool _loadedNudeDetectorGpuAccelerationEnabled;
    private string _lastClassifierLoadErrorModelPath = string.Empty;
    private string _lastNudeDetectorLoadErrorModelPath = string.Empty;
    private string _lastClassifierLoadError = string.Empty;
    private string _lastNudeDetectorLoadError = string.Empty;
    private bool _disposed;

    public AiNsfwClassifierService(
        PerformanceLogService? performanceLogService = null,
        ImageDecodeCacheService? imageDecodeCacheService = null)
    {
        _performanceLogService = performanceLogService;
        _imageDecodeCacheService = imageDecodeCacheService;
    }

    public AiModelStatus GetStatus(AiModelSettings settings)
    {
        var classifierPath = ResolveModelPath(settings);
        var nudeDetectorPath = ResolveNudeDetectorModelPath(settings);
        var isClassifierInstalled = File.Exists(classifierPath);
        var isNudeDetectorInstalled = File.Exists(nudeDetectorPath);
        var isInstalled = isClassifierInstalled && (!settings.IsNudeDetectorEnabled || isNudeDetectorInstalled);

        return new AiModelStatus
        {
            IsEnabled = settings.IsEnabled,
            IsInstalled = isInstalled,
            ModelPath = isInstalled ? $"{classifierPath} | {nudeDetectorPath}" : $"{classifierPath} | {nudeDetectorPath}",
            ModelId = ScoringModelId,
            Message = BuildStatusMessage(settings, isClassifierInstalled, isNudeDetectorInstalled),
        };
    }

    public async Task DownloadDefaultModelAsync(AiModelSettings settings, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress = null)
    {
        var classifierPath = ResolveModelPath(settings);
        var nudeDetectorPath = ResolveNudeDetectorModelPath(settings);
        var classifierInstalled = File.Exists(classifierPath);
        var nudeDetectorInstalled = File.Exists(nudeDetectorPath) || !settings.IsNudeDetectorEnabled;

        if (classifierInstalled && nudeDetectorInstalled)
        {
            progress?.Report(new WorkspaceProgressInfo
            {
                Stage = WorkspaceProgressStage.DownloadingModel,
                Title = "AI 模型已就绪",
                Detail = "本地通用 NSFW 分类模型和 NudeNet 裸露检测模型都已存在，本次不会重复下载。",
                CurrentItem = "本地 AI 模型包",
                Current = 100,
                Total = 100,
            });
            return;
        }

        if (!classifierInstalled)
        {
            await DownloadClassifierModelAsync(classifierPath, cancellationToken, progress).ConfigureAwait(false);
        }

        if (settings.IsNudeDetectorEnabled && !File.Exists(nudeDetectorPath))
        {
            await DownloadNudeDetectorModelAsync(nudeDetectorPath, cancellationToken, progress).ConfigureAwait(false);
        }

        ResetSessions();
        progress?.Report(new WorkspaceProgressInfo
        {
            Stage = WorkspaceProgressStage.DownloadingModel,
            Title = "AI 模型已就绪",
            Detail = "本地 AI 模型包已经安装完成，下一次扫描会结合通用 NSFW 分类、NudeNet 裸露检测和个人学习结果。",
            CurrentItem = "本地 AI 模型包",
            Current = 100,
            Total = 100,
        });
    }

    public async Task<AiClassificationResult> ClassifyAsync(string filePath, AiModelSettings settings, CancellationToken cancellationToken)
    {
        var results = await ClassifyBatchAsync([filePath], settings, cancellationToken).ConfigureAwait(false);
        return results.Count > 0 ? results[0] : CreateUnavailableResult("AI 视觉模型未返回结果。");
    }

    public async Task<IReadOnlyList<AiClassificationResult>> ClassifyBatchAsync(IReadOnlyList<string> filePaths, AiModelSettings settings, CancellationToken cancellationToken)
    {
        if (!settings.IsEnabled)
        {
            return filePaths.Select(_ => CreateUnavailableResult("AI 视觉模型已关闭。")).ToList();
        }

        var classifierPath = ResolveModelPath(settings);
        if (!File.Exists(classifierPath))
        {
            return filePaths.Select(_ => CreateUnavailableResult("通用 NSFW 分类模型尚未安装。")).ToList();
        }

        var processingFilePaths = _imageDecodeCacheService is null
            ? filePaths
            : await _imageDecodeCacheService
                .ResolveForProcessingAsync(filePaths, ImageDecodeCacheService.DefaultProcessingWidth, cancellationToken)
                .ConfigureAwait(false);

        await _runSemaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            return await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var results = new AiClassificationResult[filePaths.Count];
                var labelScoresByIndex = new Dictionary<int, Dictionary<string, double>>();
                var nudeDetectorPath = ResolveNudeDetectorModelPath(settings);
                try
                {
                    var classifierScores = RunCombinedBatch(
                        processingFilePaths,
                        classifierPath,
                        settings.IsNudeDetectorEnabled && File.Exists(nudeDetectorPath) ? nudeDetectorPath : null,
                        settings.NudeDetectorInputSize,
                        settings);
                    for (var index = 0; index < classifierScores.Count; index++)
                    {
                        labelScoresByIndex[index] = classifierScores[index];
                    }
                }
                catch
                {
                    for (var index = 0; index < filePaths.Count; index++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        try
                        {
                            labelScoresByIndex[index] = RunClassifier(processingFilePaths[index], classifierPath, settings);
                        }
                        catch (Exception exception)
                        {
                            results[index] = CreateUnavailableResult($"通用 NSFW 分类模型运行失败：{exception.Message}");
                        }
                    }
                }

                if (settings.IsNudeDetectorEnabled &&
                    File.Exists(nudeDetectorPath) &&
                    labelScoresByIndex.Count > 0 &&
                    labelScoresByIndex.Values.Any(scores => !scores.ContainsKey("nude.strong_explicit")))
                {
                    var nudeIndexes = labelScoresByIndex.Keys.OrderBy(index => index).ToList();
                    var nudePaths = nudeIndexes.Select(index => processingFilePaths[index]).ToList();
                    IReadOnlyList<Dictionary<string, double>> nudeScores;
                    try
                    {
                        nudeScores = RunNudeDetectorBatch(nudePaths, nudeDetectorPath, settings.NudeDetectorInputSize, settings);
                    }
                    catch
                    {
                        nudeScores = nudePaths.Select(path =>
                        {
                            try
                            {
                                return RunNudeDetector(path, nudeDetectorPath, settings.NudeDetectorInputSize, settings);
                            }
                            catch
                            {
                                return new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
                            }
                        }).ToList();
                    }

                    for (var nudeIndex = 0; nudeIndex < Math.Min(nudeIndexes.Count, nudeScores.Count); nudeIndex++)
                    {
                        var labelScores = labelScoresByIndex[nudeIndexes[nudeIndex]];
                        foreach (var score in nudeScores[nudeIndex])
                        {
                            labelScores[score.Key] = score.Value;
                        }
                    }
                }

                foreach (var item in labelScoresByIndex)
                {
                    var labelScores = item.Value;
                    var nsfwScore = CalculateWeightedNsfwScore(labelScores);
                    var sfwScore = CalculateSfwScore(labelScores);
                    var nudeScore = CalculateNudeDetectionScore(labelScores);
                    results[item.Key] = new AiClassificationResult
                    {
                        IsAvailable = true,
                        ModelId = settings.IsNudeDetectorEnabled && File.Exists(nudeDetectorPath) ? ScoringModelId : DefaultModelId + ":explicit-nsfw-v2",
                        NsfwScore = Math.Clamp(nsfwScore, 0d, 1d),
                        SfwScore = Math.Clamp(sfwScore, 0d, 1d),
                        LabelScores = labelScores,
                        Message = nudeScore > 0d
                            ? "AI 已完成图片内容判断，并结合了 NudeNet 裸露部位检测。"
                            : "AI 已完成图片内容判断。",
                    };
                }

                return results.Select(result => result ?? CreateUnavailableResult("AI 视觉模型未返回结果。")).ToList();
            }, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            return filePaths.Select(_ => CreateUnavailableResult($"AI 视觉模型运行失败：{exception.Message}")).ToList();
        }
        finally
        {
            _runSemaphore.Release();
        }
    }

    public string ResolveModelPath(AiModelSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            return Path.GetFullPath(settings.ModelPath);
        }

        return AppRuntimePaths.ResolveDataPath("AiModels", "nsfw-image-detector", "model_quantized.onnx");
    }

    public string ResolveNudeDetectorModelPath(AiModelSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(settings.NudeDetectorModelPath))
        {
            return Path.GetFullPath(settings.NudeDetectorModelPath);
        }

        return AppRuntimePaths.ResolveDataPath("AiModels", "nudenet", "320n.onnx");
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ResetSessions();
        _runSemaphore.Dispose();
    }

    public static double CalculateWeightedNsfwScore(IReadOnlyDictionary<string, double> labelScores)
    {
        var genericExplicitScore = GetGenericExplicitNsfwScore(labelScores);
        var nudeScore = CalculateNudeDetectionScore(labelScores);
        var supportedGenericScore = CalculateSupportedGenericExplicitScore(labelScores, genericExplicitScore, nudeScore);
        var classifierScore = Math.Clamp(supportedGenericScore + (GetScore(labelScores, "sexy") * 0.1d), 0d, 1d);
        return Math.Clamp(1d - ((1d - classifierScore) * (1d - nudeScore)), 0d, 1d);
    }

    public static double CalculateExplicitNsfwScore(IReadOnlyDictionary<string, double> labelScores)
    {
        var genericExplicitScore = GetGenericExplicitNsfwScore(labelScores);
        var nudeScore = CalculateNudeDetectionScore(labelScores);
        return Math.Clamp(Math.Max(CalculateSupportedGenericExplicitScore(labelScores, genericExplicitScore, nudeScore), nudeScore), 0d, 1d);
    }

    public static double CalculateGenericExplicitNsfwScore(IReadOnlyDictionary<string, double> labelScores)
    {
        return GetGenericExplicitNsfwScore(labelScores);
    }

    public static double CalculateNudeDetectionScore(IReadOnlyDictionary<string, double> labelScores)
    {
        var breast = GetScore(labelScores, "nude.female_breast_exposed");
        var genitalia = GetScore(labelScores, "nude.genitalia_exposed");
        var anus = GetScore(labelScores, "nude.anus_exposed");
        var buttocks = GetScore(labelScores, "nude.buttocks_exposed");
        var highRiskExplicit = Math.Max(genitalia, anus);
        var softExplicit = Math.Max(breast, buttocks);
        var explicitArea = GetScore(labelScores, "nude.explicit_area");
        var explicitCount = GetScore(labelScores, "nude.explicit_count");
        var softEvidenceWeight = Math.Clamp(0.34d + (explicitArea * 0.45d) + (explicitCount * 0.18d), 0.28d, 0.78d);
        var softExplicitEvidence = softExplicit * softEvidenceWeight;

        return Math.Clamp(Math.Max(highRiskExplicit, softExplicitEvidence), 0d, 1d);
    }

    private static double CalculateSupportedGenericExplicitScore(
        IReadOnlyDictionary<string, double> labelScores,
        double genericExplicitScore,
        double nudeScore)
    {
        if (genericExplicitScore <= 0d)
        {
            return 0d;
        }

        if (!HasNudeDetectorEvidenceMetadata(labelScores))
        {
            return genericExplicitScore;
        }

        var supportWeight = Math.Clamp(
            0.28d +
            (nudeScore * 0.65d) +
            (GetScore(labelScores, "nude.explicit_area") * 0.25d) +
            (GetScore(labelScores, "sexy") * 0.2d) +
            (GetScore(labelScores, "nude.detection_count") * 0.1d),
            0.24d,
            1d);

        return Math.Clamp(genericExplicitScore * supportWeight, 0d, 1d);
    }

    private static bool HasNudeDetectorEvidenceMetadata(IReadOnlyDictionary<string, double> labelScores)
    {
        return labelScores.ContainsKey("nude.detection_count") ||
               labelScores.ContainsKey("nude.explicit_area") ||
               labelScores.ContainsKey("nude.strong_explicit") ||
               labelScores.Keys.Any(label => label.StartsWith("nude.", StringComparison.OrdinalIgnoreCase));
    }

    public static double CalculateSfwScore(IReadOnlyDictionary<string, double> labelScores)
    {
        return Math.Clamp(GetScore(labelScores, "drawings") + GetScore(labelScores, "neutral"), 0d, 1d);
    }

    public static double GetScore(IReadOnlyDictionary<string, double> labelScores, string label)
    {
        return labelScores.TryGetValue(label, out var score) ? Math.Clamp(score, 0d, 1d) : 0d;
    }

    private static string BuildStatusMessage(AiModelSettings settings, bool isClassifierInstalled, bool isNudeDetectorInstalled)
    {
        if (!settings.IsEnabled)
        {
            return "AI 视觉模型已关闭。";
        }

        if (isClassifierInstalled && (!settings.IsNudeDetectorEnabled || isNudeDetectorInstalled))
        {
            return settings.IsNudeDetectorEnabled
                ? "AI 模型已就绪：当前会结合通用 NSFW 分类、NudeNet 裸露检测和个人学习结果。"
                : "AI 模型已就绪：当前只启用通用 NSFW 分类和个人学习结果。";
        }

        if (isClassifierInstalled && !isNudeDetectorInstalled)
        {
            return "通用 NSFW 分类模型已安装，但 NudeNet 裸露检测模型缺失。点击下载会补齐缺失模型，不会重复下载已有模型。";
        }

        if (!isClassifierInstalled && isNudeDetectorInstalled)
        {
            return "NudeNet 裸露检测模型已安装，但通用 NSFW 分类模型缺失。点击下载会补齐缺失模型。";
        }

        return "AI 模型包尚未安装。点击下载后会安装通用 NSFW 分类模型和 NudeNet 裸露检测模型。";
    }

    private async Task DownloadClassifierModelAsync(string modelPath, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        var tempPath = modelPath + ".download";

        Exception? lastException = null;
        foreach (var downloadUrl in DefaultDownloadUrls)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                await DownloadFileAsync(downloadUrl, tempPath, cancellationToken, progress, "正在下载通用 NSFW 分类模型", "通用 NSFW 分类模型").ConfigureAwait(false);
                EnsureDownloadedModelLooksValid(tempPath, "通用 NSFW 分类模型");
                File.Move(tempPath, modelPath, overwrite: true);
                return;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                lastException = exception;
            }
        }

        throw new IOException("通用 NSFW 分类模型下载失败，请稍后重试或手动放入模型文件。", lastException);
    }

    private async Task DownloadNudeDetectorModelAsync(string modelPath, CancellationToken cancellationToken, IProgress<WorkspaceProgressInfo>? progress)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        var tempPath = modelPath + ".download";

        Exception? lastException = null;
        foreach (var downloadUrl in NudeDetectorOnnxDownloadUrls)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                await DownloadFileAsync(downloadUrl, tempPath, cancellationToken, progress, "正在下载 NudeNet 裸露检测模型", "NudeNet 320n 裸露检测模型").ConfigureAwait(false);
                EnsureDownloadedModelLooksValid(tempPath, "NudeNet 裸露检测模型");
                File.Move(tempPath, modelPath, overwrite: true);
                return;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                lastException = exception;
            }
        }

        foreach (var downloadUrl in NudeDetectorWheelDownloadUrls)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                var wheelPath = modelPath + ".wheel.download";
                await DownloadFileAsync(downloadUrl, wheelPath, cancellationToken, progress, "正在下载 NudeNet 官方包", "NudeNet PyPI 官方包").ConfigureAwait(false);
                await ExtractNudeDetectorFromWheelAsync(wheelPath, tempPath, cancellationToken).ConfigureAwait(false);
                EnsureDownloadedModelLooksValid(tempPath, "NudeNet 裸露检测模型");
                File.Move(tempPath, modelPath, overwrite: true);
                return;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                lastException = exception;
            }
        }

        throw new IOException("NudeNet 裸露检测模型下载失败，请稍后重试或手动放入 320n.onnx。", lastException);
    }

    private static void EnsureDownloadedModelLooksValid(string filePath, string modelName)
    {
        var fileInfo = new FileInfo(filePath);
        if (!fileInfo.Exists || fileInfo.Length < 1024 * 1024)
        {
            throw new InvalidDataException($"{modelName} 文件体积异常，可能下载到了错误页面。");
        }
    }

    private static async Task DownloadFileAsync(
        string downloadUrl,
        string tempPath,
        CancellationToken cancellationToken,
        IProgress<WorkspaceProgressInfo>? progress,
        string title,
        string modelName)
    {
        using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        using var response = await httpClient.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength;
        var readBytes = 0L;
        await using var responseStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        await using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, FileOptions.Asynchronous | FileOptions.SequentialScan);
        var buffer = new byte[81920];

        while (true)
        {
            var count = await responseStream.ReadAsync(buffer.AsMemory(0, buffer.Length), cancellationToken).ConfigureAwait(false);
            if (count == 0)
            {
                break;
            }

            await fileStream.WriteAsync(buffer.AsMemory(0, count), cancellationToken).ConfigureAwait(false);
            readBytes += count;

            if (totalBytes is > 0)
            {
                var percent = (int)Math.Clamp(Math.Round(readBytes * 100d / totalBytes.Value), 0d, 100d);
                progress?.Report(new WorkspaceProgressInfo
                {
                    Stage = WorkspaceProgressStage.DownloadingModel,
                    Title = title,
                    Detail = $"{modelName} 已下载 {percent}%。如果直连较慢，会自动尝试国内镜像源。",
                    CurrentItem = downloadUrl.Contains("hf-mirror", StringComparison.OrdinalIgnoreCase) ? "国内镜像源" : "官方源",
                    Current = percent,
                    Total = 100,
                });
            }
        }
    }

    private static async Task ExtractNudeDetectorFromWheelAsync(string wheelPath, string targetPath, CancellationToken cancellationToken)
    {
        using var archive = ZipFile.OpenRead(wheelPath);
        var entry = archive.GetEntry(NudeDetectorWheelEntryName)
                    ?? throw new FileNotFoundException("NudeNet 官方包中没有找到 320n.onnx。", NudeDetectorWheelEntryName);
        await using var entryStream = entry.Open();
        await using var fileStream = new FileStream(targetPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, FileOptions.Asynchronous | FileOptions.SequentialScan);
        await entryStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
    }

    private IReadOnlyList<Dictionary<string, double>> RunCombinedBatch(
        IReadOnlyList<string> filePaths,
        string classifierModelPath,
        string? nudeDetectorModelPath,
        int requestedNudeDetectorInputSize,
        AiModelSettings settings)
    {
        if (filePaths.Count == 0)
        {
            return [];
        }

        var classifierSession = GetOrCreateClassifierSession(classifierModelPath, settings);
        var classifierInput = classifierSession.InputMetadata.First();
        var classifierInputName = classifierInput.Key;
        var classifierDimensions = classifierInput.Value.Dimensions;
        var classifierIsNhwc = classifierDimensions.Length == 4 && classifierDimensions[3] == 3;
        var classifierImageSize = ResolveImageSize(classifierDimensions, DefaultClassifierInputSize);

        InferenceSession? nudeDetectorSession = null;
        string nudeDetectorInputName = string.Empty;
        var nudeDetectorImageSize = DefaultNudeDetectorInputSize;
        var shouldRunNudeDetector = !string.IsNullOrWhiteSpace(nudeDetectorModelPath);
        if (shouldRunNudeDetector)
        {
            nudeDetectorSession = GetOrCreateNudeDetectorSession(nudeDetectorModelPath!, settings);
            var nudeDetectorInput = nudeDetectorSession.InputMetadata.First();
            nudeDetectorInputName = nudeDetectorInput.Key;
            nudeDetectorImageSize = requestedNudeDetectorInputSize > 0
                ? requestedNudeDetectorInputSize
                : ResolveImageSize(nudeDetectorInput.Value.Dimensions, DefaultNudeDetectorInputSize);
        }

        var preparedInputs = CreatePreparedAiInputs(filePaths, classifierImageSize, classifierIsNhwc, shouldRunNudeDetector, nudeDetectorImageSize);
        var classifierBatchTensor = CreateClassifierBatchTensor(preparedInputs.Select(item => item.ClassifierTensor).ToList(), classifierImageSize, classifierIsNhwc);
        var classifierInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(classifierInputName, classifierBatchTensor) };
        using var classifierResults = classifierSession.Run(classifierInputs);
        var classifierOutputTensor = classifierResults.First().AsTensor<float>();
        var labelScores = ParseClassifierBatchScores(classifierOutputTensor, filePaths.Count).ToList();

        if (shouldRunNudeDetector && nudeDetectorSession is not null)
        {
            var nudeDetectorInputs = preparedInputs
                .Select(item => item.NudeDetectorInput ?? throw new InvalidOperationException("NudeNet 输入尚未准备完成。"))
                .ToList();
            var nudeDetectorBatchTensor = CreateNudeDetectorBatchTensor(nudeDetectorInputs, nudeDetectorImageSize);
            var nudeInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(nudeDetectorInputName, nudeDetectorBatchTensor) };
            using var nudeResults = nudeDetectorSession.Run(nudeInputs);
            var nudeOutputTensor = nudeResults.First().AsTensor<float>();
            var detectionsByImage = ParseNudeDetectionsBatch(nudeOutputTensor, nudeDetectorInputs);
            for (var index = 0; index < labelScores.Count; index++)
            {
                var detections = index < detectionsByImage.Count ? detectionsByImage[index] : Array.Empty<NudeDetection>();
                var nudeScores = BuildNudeDetectorLabelScores(detections, nudeDetectorInputs[index].OriginalWidth, nudeDetectorInputs[index].OriginalHeight);
                foreach (var score in nudeScores)
                {
                    labelScores[index][score.Key] = score.Value;
                }
            }
        }

        return labelScores;
    }

    private Dictionary<string, double> RunClassifier(string filePath, string modelPath, AiModelSettings settings)
    {
        return RunClassifierBatch([filePath], modelPath, settings)[0];
    }

    private IReadOnlyList<Dictionary<string, double>> RunClassifierBatch(IReadOnlyList<string> filePaths, string modelPath, AiModelSettings settings)
    {
        if (filePaths.Count == 0)
        {
            return [];
        }

        var session = GetOrCreateClassifierSession(modelPath, settings);
        var input = session.InputMetadata.First();
        var inputName = input.Key;
        var dimensions = input.Value.Dimensions;
        var isNhwc = dimensions.Length == 4 && dimensions[3] == 3;
        var imageSize = ResolveImageSize(dimensions, DefaultClassifierInputSize);
        var inputTensors = CreateClassifierInputTensors(filePaths, imageSize, isNhwc);
        var batchTensor = CreateClassifierBatchTensor(inputTensors, imageSize, isNhwc);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, batchTensor) };
        using var results = session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        return ParseClassifierBatchScores(outputTensor, filePaths.Count);
    }

    private Dictionary<string, double> RunNudeDetector(string filePath, string modelPath, int requestedInputSize, AiModelSettings settings)
    {
        return RunNudeDetectorBatch([filePath], modelPath, requestedInputSize, settings)[0];
    }

    private IReadOnlyList<Dictionary<string, double>> RunNudeDetectorBatch(IReadOnlyList<string> filePaths, string modelPath, int requestedInputSize, AiModelSettings settings)
    {
        if (filePaths.Count == 0)
        {
            return [];
        }

        var session = GetOrCreateNudeDetectorSession(modelPath, settings);
        var input = session.InputMetadata.First();
        var inputName = input.Key;
        var imageSize = requestedInputSize > 0 ? requestedInputSize : ResolveImageSize(input.Value.Dimensions, DefaultNudeDetectorInputSize);
        var preprocessItems = CreateNudeDetectorInputs(filePaths, imageSize);
        var batchTensor = CreateNudeDetectorBatchTensor(preprocessItems, imageSize);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, batchTensor) };
        using var results = session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var detectionsByImage = ParseNudeDetectionsBatch(outputTensor, preprocessItems);
        var labelScores = new List<Dictionary<string, double>>(preprocessItems.Count);
        for (var index = 0; index < preprocessItems.Count; index++)
        {
            var preprocess = preprocessItems[index];
            var detections = index < detectionsByImage.Count ? detectionsByImage[index] : Array.Empty<NudeDetection>();
            labelScores.Add(BuildNudeDetectorLabelScores(detections, preprocess.OriginalWidth, preprocess.OriginalHeight));
        }

        return labelScores;
    }

    private InferenceSession GetOrCreateClassifierSession(string modelPath, AiModelSettings settings)
    {
        var gpuAccelerationEnabled = ShouldUseGpuAcceleration(settings);
        lock (_classifierSessionLock)
        {
            if (_classifierSession is not null &&
                string.Equals(_loadedClassifierModelPath, modelPath, StringComparison.OrdinalIgnoreCase) &&
                _loadedClassifierGpuAccelerationEnabled == gpuAccelerationEnabled)
            {
                return _classifierSession;
            }

            if (string.Equals(_lastClassifierLoadErrorModelPath, modelPath, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(_lastClassifierLoadError))
            {
                throw new InvalidOperationException(_lastClassifierLoadError);
            }

            try
            {
                _classifierSession?.Dispose();
                _classifierSession = CreateInferenceSession(modelPath, gpuAccelerationEnabled, "通用 NSFW 分类模型");
                _loadedClassifierModelPath = modelPath;
                _loadedClassifierGpuAccelerationEnabled = gpuAccelerationEnabled;
                _lastClassifierLoadErrorModelPath = string.Empty;
                _lastClassifierLoadError = string.Empty;
                return _classifierSession;
            }
            catch (Exception exception)
            {
                _classifierSession = null;
                _loadedClassifierModelPath = string.Empty;
                _loadedClassifierGpuAccelerationEnabled = false;
                _lastClassifierLoadErrorModelPath = modelPath;
                _lastClassifierLoadError = exception.Message;
                throw;
            }
        }
    }

    private InferenceSession GetOrCreateNudeDetectorSession(string modelPath, AiModelSettings settings)
    {
        var gpuAccelerationEnabled = ShouldUseGpuAcceleration(settings);
        lock (_nudeDetectorSessionLock)
        {
            if (_nudeDetectorSession is not null &&
                string.Equals(_loadedNudeDetectorModelPath, modelPath, StringComparison.OrdinalIgnoreCase) &&
                _loadedNudeDetectorGpuAccelerationEnabled == gpuAccelerationEnabled)
            {
                return _nudeDetectorSession;
            }

            if (string.Equals(_lastNudeDetectorLoadErrorModelPath, modelPath, StringComparison.OrdinalIgnoreCase) &&
                !string.IsNullOrWhiteSpace(_lastNudeDetectorLoadError))
            {
                throw new InvalidOperationException(_lastNudeDetectorLoadError);
            }

            try
            {
                _nudeDetectorSession?.Dispose();
                _nudeDetectorSession = CreateInferenceSession(modelPath, gpuAccelerationEnabled, "NudeNet 裸露检测模型");
                _loadedNudeDetectorModelPath = modelPath;
                _loadedNudeDetectorGpuAccelerationEnabled = gpuAccelerationEnabled;
                _lastNudeDetectorLoadErrorModelPath = string.Empty;
                _lastNudeDetectorLoadError = string.Empty;
                return _nudeDetectorSession;
            }
            catch (Exception exception)
            {
                _nudeDetectorSession = null;
                _loadedNudeDetectorModelPath = string.Empty;
                _loadedNudeDetectorGpuAccelerationEnabled = false;
                _lastNudeDetectorLoadErrorModelPath = modelPath;
                _lastNudeDetectorLoadError = exception.Message;
                throw;
            }
        }
    }

    private InferenceSession CreateInferenceSession(string modelPath, bool preferGpuAcceleration, string modelName)
    {
        if (preferGpuAcceleration)
        {
            try
            {
                using var directMlOptions = CreateSessionOptions(useDirectMl: true);
                var session = new InferenceSession(modelPath, directMlOptions);
                QueueRuntimeLog($"{modelName} 已启用 DirectML GPU 加速。");
                return session;
            }
            catch (Exception exception)
            {
                QueueRuntimeLog($"{modelName} DirectML GPU 加速启动失败，已自动退回 CPU：{exception.Message}");
            }
        }

        using var cpuOptions = CreateSessionOptions(useDirectMl: false);
        var cpuSession = new InferenceSession(modelPath, cpuOptions);
        QueueRuntimeLog($"{modelName} 当前使用 CPU 推理。");
        return cpuSession;
    }

    private static SessionOptions CreateSessionOptions(bool useDirectMl)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            InterOpNumThreads = 1,
        };

        if (useDirectMl)
        {
            options.EnableMemoryPattern = false;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            options.IntraOpNumThreads = 1;
            options.AppendExecutionProvider_DML(0);
            return options;
        }

        options.IntraOpNumThreads = GetCpuInferenceThreadCount();
        return options;
    }

    private static int GetCpuInferenceThreadCount()
    {
        return Math.Clamp(Environment.ProcessorCount / 2, 2, 8);
    }

    private static int GetImagePreprocessingParallelism()
    {
        return Math.Clamp(Environment.ProcessorCount / 2, 2, 8);
    }

    private static bool ShouldUseGpuAcceleration(AiModelSettings settings)
    {
        return settings.IsGpuAccelerationEnabled && OperatingSystem.IsWindows();
    }

    private void QueueRuntimeLog(string message)
    {
        if (_performanceLogService is null)
        {
            return;
        }

        _ = LogRuntimeAsync(message);
    }

    private async Task LogRuntimeAsync(string message)
    {
        try
        {
            await _performanceLogService!.LogAsync("AiRuntime", message).ConfigureAwait(false);
        }
        catch
        {
        }
    }

    private void ResetSessions()
    {
        lock (_classifierSessionLock)
        {
            _classifierSession?.Dispose();
            _classifierSession = null;
            _loadedClassifierModelPath = string.Empty;
            _loadedClassifierGpuAccelerationEnabled = false;
            _lastClassifierLoadErrorModelPath = string.Empty;
            _lastClassifierLoadError = string.Empty;
        }

        lock (_nudeDetectorSessionLock)
        {
            _nudeDetectorSession?.Dispose();
            _nudeDetectorSession = null;
            _loadedNudeDetectorModelPath = string.Empty;
            _loadedNudeDetectorGpuAccelerationEnabled = false;
            _lastNudeDetectorLoadErrorModelPath = string.Empty;
            _lastNudeDetectorLoadError = string.Empty;
        }
    }

    private static DenseTensor<float> CreateClassifierInputTensor(string filePath, int imageSize, bool isNhwc)
    {
        using var source = new Bitmap(filePath);
        return CreateClassifierInputTensor(source, imageSize, isNhwc);
    }

    private static DenseTensor<float> CreateClassifierInputTensor(Bitmap source, int imageSize, bool isNhwc)
    {
        var dimensions = isNhwc
            ? new[] { 1, imageSize, imageSize, 3 }
            : new[] { 1, 3, imageSize, imageSize };
        var tensor = new DenseTensor<float>(dimensions);

        using var resized = new Bitmap(imageSize, imageSize, PixelFormat.Format24bppRgb);
        using (var graphics = Graphics.FromImage(resized))
        {
            graphics.Clear(Color.White);
            ConfigureInferenceGraphics(graphics);
            graphics.DrawImage(source, 0, 0, imageSize, imageSize);
        }

        CopyBitmapToTensor(resized, tensor, isNhwc, NormalizeClassifierChannel);
        return tensor;
    }

    private static PreparedAiInput CreatePreparedAiInput(string filePath, int classifierImageSize, bool classifierIsNhwc, bool includeNudeDetectorInput, int nudeDetectorImageSize)
    {
        using var source = new Bitmap(filePath);
        return new PreparedAiInput(
            CreateClassifierInputTensor(source, classifierImageSize, classifierIsNhwc),
            includeNudeDetectorInput ? CreateNudeDetectorInputTensor(source, nudeDetectorImageSize) : null);
    }

    private static List<PreparedAiInput> CreatePreparedAiInputs(
        IReadOnlyList<string> filePaths,
        int classifierImageSize,
        bool classifierIsNhwc,
        bool includeNudeDetectorInput,
        int nudeDetectorImageSize)
    {
        if (filePaths.Count < 3)
        {
            return filePaths
                .Select(filePath => CreatePreparedAiInput(filePath, classifierImageSize, classifierIsNhwc, includeNudeDetectorInput, nudeDetectorImageSize))
                .ToList();
        }

        return filePaths
            .AsParallel()
            .AsOrdered()
            .WithDegreeOfParallelism(GetImagePreprocessingParallelism())
            .Select(filePath => CreatePreparedAiInput(filePath, classifierImageSize, classifierIsNhwc, includeNudeDetectorInput, nudeDetectorImageSize))
            .ToList();
    }

    private static List<DenseTensor<float>> CreateClassifierInputTensors(IReadOnlyList<string> filePaths, int imageSize, bool isNhwc)
    {
        if (filePaths.Count < 3)
        {
            return filePaths.Select(filePath => CreateClassifierInputTensor(filePath, imageSize, isNhwc)).ToList();
        }

        return filePaths
            .AsParallel()
            .AsOrdered()
            .WithDegreeOfParallelism(GetImagePreprocessingParallelism())
            .Select(filePath => CreateClassifierInputTensor(filePath, imageSize, isNhwc))
            .ToList();
    }

    private static List<NudeDetectorInput> CreateNudeDetectorInputs(IReadOnlyList<string> filePaths, int imageSize)
    {
        if (filePaths.Count < 3)
        {
            return filePaths.Select(filePath => CreateNudeDetectorInputTensor(filePath, imageSize)).ToList();
        }

        return filePaths
            .AsParallel()
            .AsOrdered()
            .WithDegreeOfParallelism(GetImagePreprocessingParallelism())
            .Select(filePath => CreateNudeDetectorInputTensor(filePath, imageSize))
            .ToList();
    }

    private static DenseTensor<float> CreateClassifierBatchTensor(IReadOnlyList<DenseTensor<float>> inputTensors, int imageSize, bool isNhwc)
    {
        var dimensions = isNhwc
            ? new[] { inputTensors.Count, imageSize, imageSize, 3 }
            : new[] { inputTensors.Count, 3, imageSize, imageSize };
        var batchTensor = new DenseTensor<float>(dimensions);
        var imageValueCount = 3 * imageSize * imageSize;
        var batchSpan = batchTensor.Buffer.Span;

        for (var batchIndex = 0; batchIndex < inputTensors.Count; batchIndex++)
        {
            inputTensors[batchIndex].Buffer.Span.CopyTo(batchSpan.Slice(batchIndex * imageValueCount, imageValueCount));
        }

        return batchTensor;
    }

    private static IReadOnlyList<Dictionary<string, double>> ParseClassifierBatchScores(Tensor<float> outputTensor, int batchCount)
    {
        var values = outputTensor.ToArray();
        if (batchCount <= 0 || values.Length == 0)
        {
            return [];
        }

        if (values.Length < batchCount * DefaultLabels.Length)
        {
            throw new InvalidDataException("通用 NSFW 分类模型的批量输出数量异常。");
        }

        var featureCount = values.Length % batchCount == 0
            ? values.Length / batchCount
            : DefaultLabels.Length;
        if (featureCount < DefaultLabels.Length)
        {
            featureCount = DefaultLabels.Length;
        }

        var results = new List<Dictionary<string, double>>(batchCount);
        for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            var offset = batchIndex * featureCount;
            var rawScores = values
                .Skip(offset)
                .Take(Math.Min(DefaultLabels.Length, Math.Max(0, values.Length - offset)))
                .ToArray();
            results.Add(BuildClassifierLabelScores(NormalizeScores(rawScores)));
        }

        return results;
    }

    private static NudeDetectorInput CreateNudeDetectorInputTensor(string filePath, int imageSize)
    {
        using var source = new Bitmap(filePath);
        return CreateNudeDetectorInputTensor(source, imageSize);
    }

    private static NudeDetectorInput CreateNudeDetectorInputTensor(Bitmap source, int imageSize)
    {
        var maxSize = Math.Max(source.Width, source.Height);
        using var resized = new Bitmap(imageSize, imageSize, PixelFormat.Format24bppRgb);
        using (var graphics = Graphics.FromImage(resized))
        {
            graphics.Clear(Color.Black);
            ConfigureInferenceGraphics(graphics);
            var scaledWidth = Math.Max(1, (int)Math.Round(source.Width * imageSize / (double)maxSize));
            var scaledHeight = Math.Max(1, (int)Math.Round(source.Height * imageSize / (double)maxSize));
            graphics.DrawImage(source, 0, 0, scaledWidth, scaledHeight);
        }

        var tensor = new DenseTensor<float>(new[] { 1, 3, imageSize, imageSize });
        CopyBitmapToTensor(resized, tensor, isNhwc: false, NormalizeNudeDetectorChannel);
        return new NudeDetectorInput(tensor, imageSize, source.Width, source.Height, maxSize);
    }

    private static void ConfigureInferenceGraphics(Graphics graphics)
    {
        graphics.CompositingQuality = CompositingQuality.HighSpeed;
        graphics.InterpolationMode = InterpolationMode.Low;
        graphics.PixelOffsetMode = PixelOffsetMode.Half;
        graphics.SmoothingMode = SmoothingMode.None;
    }

    private static DenseTensor<float> CreateNudeDetectorBatchTensor(IReadOnlyList<NudeDetectorInput> preprocessItems, int imageSize)
    {
        var batchTensor = new DenseTensor<float>(new[] { preprocessItems.Count, 3, imageSize, imageSize });
        var imageValueCount = 3 * imageSize * imageSize;
        var batchSpan = batchTensor.Buffer.Span;

        for (var batchIndex = 0; batchIndex < preprocessItems.Count; batchIndex++)
        {
            var sourceSpan = preprocessItems[batchIndex].Tensor.Buffer.Span;
            sourceSpan.CopyTo(batchSpan.Slice(batchIndex * imageValueCount, imageValueCount));
        }

        return batchTensor;
    }

    private static void CopyBitmapToTensor(Bitmap bitmap, DenseTensor<float> tensor, bool isNhwc, Func<byte, float> normalizeChannel)
    {
        var imageSize = bitmap.Width;
        var bounds = new Rectangle(0, 0, imageSize, imageSize);
        var bitmapData = bitmap.LockBits(bounds, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        var stride = bitmapData.Stride;
        var rowLength = Math.Abs(stride);
        var bufferLength = rowLength * imageSize;
        var rentedBuffer = ArrayPool<byte>.Shared.Rent(bufferLength);

        try
        {
            Marshal.Copy(bitmapData.Scan0, rentedBuffer, 0, bufferLength);
            var tensorSpan = tensor.Buffer.Span;
            var planeSize = imageSize * imageSize;

            for (var y = 0; y < imageSize; y++)
            {
                var rowOffset = stride > 0 ? y * stride : (imageSize - 1 - y) * rowLength;
                for (var x = 0; x < imageSize; x++)
                {
                    var pixelOffset = rowOffset + (x * 3);
                    var blue = normalizeChannel(rentedBuffer[pixelOffset]);
                    var green = normalizeChannel(rentedBuffer[pixelOffset + 1]);
                    var red = normalizeChannel(rentedBuffer[pixelOffset + 2]);

                    if (isNhwc)
                    {
                        var tensorOffset = ((y * imageSize) + x) * 3;
                        tensorSpan[tensorOffset] = red;
                        tensorSpan[tensorOffset + 1] = green;
                        tensorSpan[tensorOffset + 2] = blue;
                    }
                    else
                    {
                        var tensorOffset = (y * imageSize) + x;
                        tensorSpan[tensorOffset] = red;
                        tensorSpan[planeSize + tensorOffset] = green;
                        tensorSpan[(planeSize * 2) + tensorOffset] = blue;
                    }
                }
            }
        }
        finally
        {
            bitmap.UnlockBits(bitmapData);
            ArrayPool<byte>.Shared.Return(rentedBuffer);
        }
    }

    private static IReadOnlyList<IReadOnlyList<NudeDetection>> ParseNudeDetectionsBatch(Tensor<float> outputTensor, IReadOnlyList<NudeDetectorInput> inputs)
    {
        var values = outputTensor.ToArray();
        var dimensions = outputTensor.Dimensions.ToArray();
        var emptyResults = inputs.Select(_ => (IReadOnlyList<NudeDetection>)Array.Empty<NudeDetection>()).ToList();
        if (!TryResolveDetectionLayout(dimensions, out var batchCount, out var rowCount, out var featureCount, out var layout))
        {
            return emptyResults;
        }

        if (inputs.Count > 1 && batchCount < inputs.Count)
        {
            throw new InvalidDataException("NudeNet 裸露检测模型的批量输出数量异常。");
        }

        var results = new List<IReadOnlyList<NudeDetection>>(inputs.Count);
        for (var batchIndex = 0; batchIndex < inputs.Count; batchIndex++)
        {
            if (batchIndex >= batchCount)
            {
                results.Add(Array.Empty<NudeDetection>());
                continue;
            }

            var input = inputs[batchIndex];
            var boxes = new List<NudeDetectionBox>();
            for (var row = 0; row < rowCount; row++)
            {
                var maxScore = 0f;
                var classId = -1;
                for (var labelIndex = 0; labelIndex < NudeDetectionLabels.Length; labelIndex++)
                {
                    var score = GetDetectionValue(values, batchIndex, row, 4 + labelIndex, rowCount, featureCount, layout);
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = labelIndex;
                    }
                }

                if (classId < 0 || maxScore < NudeDetectionConfidenceThreshold)
                {
                    continue;
                }

                var centerX = GetDetectionValue(values, batchIndex, row, 0, rowCount, featureCount, layout);
                var centerY = GetDetectionValue(values, batchIndex, row, 1, rowCount, featureCount, layout);
                var width = GetDetectionValue(values, batchIndex, row, 2, rowCount, featureCount, layout);
                var height = GetDetectionValue(values, batchIndex, row, 3, rowCount, featureCount, layout);
                var left = centerX - (width / 2f);
                var top = centerY - (height / 2f);
                var scale = input.PaddedSize / (double)input.InputSize;
                var x = Math.Clamp(left * scale, 0d, input.OriginalWidth);
                var y = Math.Clamp(top * scale, 0d, input.OriginalHeight);
                var w = Math.Clamp(width * scale, 0d, input.OriginalWidth - x);
                var h = Math.Clamp(height * scale, 0d, input.OriginalHeight - y);

                if (w <= 1d || h <= 1d)
                {
                    continue;
                }

                boxes.Add(new NudeDetectionBox(classId, maxScore, x, y, w, h));
            }

            results.Add(boxes.Count == 0
                ? Array.Empty<NudeDetection>()
                : ApplyNms(boxes)
                    .Select(box => new NudeDetection(NudeDetectionLabels[box.ClassId], box.Score, box.X, box.Y, box.Width, box.Height))
                    .ToList());
        }

        return results;
    }

    private static bool TryResolveDetectionLayout(int[] dimensions, out int batchCount, out int rowCount, out int featureCount, out DetectionOutputLayout layout)
    {
        batchCount = 1;
        rowCount = 0;
        featureCount = 0;
        layout = DetectionOutputLayout.ChannelFirst;

        var expectedFeatureCount = 4 + NudeDetectionLabels.Length;
        if (dimensions.Length == 3 && dimensions[1] == expectedFeatureCount)
        {
            batchCount = dimensions[0];
            featureCount = dimensions[1];
            rowCount = dimensions[2];
            layout = DetectionOutputLayout.ChannelFirst;
            return rowCount > 0;
        }

        if (dimensions.Length == 3 && dimensions[2] == expectedFeatureCount)
        {
            batchCount = dimensions[0];
            rowCount = dimensions[1];
            featureCount = dimensions[2];
            layout = DetectionOutputLayout.ChannelLast;
            return rowCount > 0;
        }

        if (dimensions.Length == 2 && dimensions[0] == expectedFeatureCount)
        {
            featureCount = dimensions[0];
            rowCount = dimensions[1];
            layout = DetectionOutputLayout.ChannelFirst;
            return rowCount > 0;
        }

        if (dimensions.Length == 2 && dimensions[1] == expectedFeatureCount)
        {
            rowCount = dimensions[0];
            featureCount = dimensions[1];
            layout = DetectionOutputLayout.ChannelLast;
            return rowCount > 0;
        }

        return false;
    }

    private static float GetDetectionValue(float[] values, int batchIndex, int row, int feature, int rowCount, int featureCount, DetectionOutputLayout layout)
    {
        var batchOffset = batchIndex * rowCount * featureCount;
        var index = layout == DetectionOutputLayout.ChannelFirst
            ? batchOffset + (feature * rowCount) + row
            : batchOffset + (row * featureCount) + feature;
        return index >= 0 && index < values.Length ? values[index] : 0f;
    }

    private static IReadOnlyList<NudeDetectionBox> ApplyNms(List<NudeDetectionBox> boxes)
    {
        var selected = new List<NudeDetectionBox>();
        foreach (var candidate in boxes
                     .Where(box => box.Score >= NudeDetectionConfidenceThreshold)
                     .OrderByDescending(box => box.Score))
        {
            if (selected.Any(existing => CalculateIntersectionOverUnion(candidate, existing) >= NudeDetectionNmsThreshold))
            {
                continue;
            }

            selected.Add(candidate);
        }

        return selected;
    }

    private static double CalculateIntersectionOverUnion(NudeDetectionBox left, NudeDetectionBox right)
    {
        var x1 = Math.Max(left.X, right.X);
        var y1 = Math.Max(left.Y, right.Y);
        var x2 = Math.Min(left.X + left.Width, right.X + right.Width);
        var y2 = Math.Min(left.Y + left.Height, right.Y + right.Height);
        var intersectionWidth = Math.Max(0d, x2 - x1);
        var intersectionHeight = Math.Max(0d, y2 - y1);
        var intersectionArea = intersectionWidth * intersectionHeight;
        if (intersectionArea <= 0d)
        {
            return 0d;
        }

        var leftArea = left.Width * left.Height;
        var rightArea = right.Width * right.Height;
        var unionArea = leftArea + rightArea - intersectionArea;
        return unionArea <= 0d ? 0d : intersectionArea / unionArea;
    }

    private static Dictionary<string, double> BuildNudeDetectorLabelScores(IReadOnlyList<NudeDetection> detections, int originalWidth, int originalHeight)
    {
        var labelScores = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        var imageArea = Math.Max(1d, originalWidth * (double)originalHeight);
        var explicitCount = 0;
        var maxExplicitAreaScore = 0d;

        foreach (var detection in detections)
        {
            var normalizedLabel = "nude." + detection.Label.ToLowerInvariant();
            labelScores[normalizedLabel] = Math.Max(GetScore(labelScores, normalizedLabel), detection.Score);

            if (IsExplicitNudeLabel(detection.Label))
            {
                explicitCount++;
                var areaScore = detection.Score * Math.Clamp((detection.Width * detection.Height) / imageArea * 8d, 0d, 1d);
                maxExplicitAreaScore = Math.Max(maxExplicitAreaScore, areaScore);
            }
        }

        var femaleBreast = GetScore(labelScores, "nude.female_breast_exposed");
        var genitalia = Math.Max(GetScore(labelScores, "nude.female_genitalia_exposed"), GetScore(labelScores, "nude.male_genitalia_exposed"));
        var anus = GetScore(labelScores, "nude.anus_exposed");
        var buttocks = GetScore(labelScores, "nude.buttocks_exposed");

        labelScores["nude.female_breast_exposed"] = femaleBreast;
        labelScores["nude.genitalia_exposed"] = genitalia;
        labelScores["nude.anus_exposed"] = anus;
        labelScores["nude.buttocks_exposed"] = buttocks;
        labelScores["nude.explicit_count"] = Math.Clamp(explicitCount / 3d, 0d, 1d);
        labelScores["nude.explicit_area"] = Math.Clamp(maxExplicitAreaScore, 0d, 1d);
        labelScores["nude.detection_count"] = Math.Clamp(detections.Count / 8d, 0d, 1d);
        labelScores["nude.strong_explicit"] = CalculateNudeDetectionScore(labelScores);
        return labelScores;
    }

    private static bool IsExplicitNudeLabel(string label)
    {
        return label is "FEMALE_BREAST_EXPOSED" or
            "FEMALE_GENITALIA_EXPOSED" or
            "MALE_GENITALIA_EXPOSED" or
            "ANUS_EXPOSED" or
            "BUTTOCKS_EXPOSED";
    }

    private static int ResolveImageSize(int[] dimensions, int defaultInputSize)
    {
        if (dimensions.Length != 4)
        {
            return defaultInputSize;
        }

        var candidates = dimensions.Where(value => value > 16 && value <= 2048).ToList();
        return candidates.Count > 0 ? candidates[^1] : defaultInputSize;
    }

    private static float NormalizeClassifierChannel(byte value)
    {
        return (value / 127.5f) - 1f;
    }

    private static float NormalizeNudeDetectorChannel(byte value)
    {
        return value / 255f;
    }

    private static double[] NormalizeScores(float[] outputValues)
    {
        var values = outputValues.Take(DefaultLabels.Length).Select(value => (double)value).ToArray();
        if (values.Length == 0)
        {
            return [];
        }

        var sum = values.Sum();
        if (values.All(value => value >= 0d && value <= 1d) && sum is > 0.98d and < 1.02d)
        {
            return values;
        }

        var max = values.Max();
        var expValues = values.Select(value => Math.Exp(value - max)).ToArray();
        var expSum = expValues.Sum();
        return expSum <= 0d ? values.Select(_ => 0d).ToArray() : expValues.Select(value => value / expSum).ToArray();
    }

    private static Dictionary<string, double> BuildClassifierLabelScores(double[] probabilities)
    {
        var labelScores = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < Math.Min(DefaultLabels.Length, probabilities.Length); index++)
        {
            labelScores[DefaultLabels[index]] = probabilities[index];
        }

        return labelScores;
    }

    private static double GetGenericExplicitNsfwScore(IReadOnlyDictionary<string, double> labelScores)
    {
        return Math.Clamp(GetScore(labelScores, "hentai") + GetScore(labelScores, "porn"), 0d, 1d);
    }

    private static AiClassificationResult CreateUnavailableResult(string message)
    {
        return new AiClassificationResult
        {
            IsAvailable = false,
            ModelId = ScoringModelId,
            Message = message,
        };
    }

    private enum DetectionOutputLayout
    {
        ChannelFirst,
        ChannelLast,
    }

    private sealed record PreparedAiInput(DenseTensor<float> ClassifierTensor, NudeDetectorInput? NudeDetectorInput);

    private sealed record NudeDetectorInput(DenseTensor<float> Tensor, int InputSize, int OriginalWidth, int OriginalHeight, int PaddedSize);

    private sealed record NudeDetection(string Label, double Score, double X, double Y, double Width, double Height);

    private sealed record NudeDetectionBox(int ClassId, double Score, double X, double Y, double Width, double Height);
}
