using System.Text.Json;
using JudgePicSFW.Models;
using System.Drawing;
using System.Drawing.Imaging;
using JudgePicSFW.Services;

var failures = new List<string>();

var imageFingerprintService = new ImageFingerprintService();
AssertTrue(imageFingerprintService.IsSupportedImage("sample.webp"), "WebP should be included in supported image scanning");
AssertTrue(imageFingerprintService.IsSupportedImage("sample.HEIC"), "HEIC should be included in supported image scanning");
AssertTrue(imageFingerprintService.IsSupportedImage("sample.heif"), "HEIF should be included in supported image scanning");
AssertTrue(imageFingerprintService.IsSupportedImage("sample.JFIF"), "JFIF should be included in supported image scanning");
AssertTrue(ImageDecodeCacheService.RequiresDedicatedDecoder("sample.webp"), "WebP should use the dedicated decoder path");
AssertTrue(ImageDecodeCacheService.RequiresDedicatedDecoder("sample.heic"), "HEIC should use the dedicated decoder path");
AssertTrue(ImageDecodeCacheService.RequiresDedicatedDecoder("sample.heif"), "HEIF should use the dedicated decoder path");
AssertFalse(ImageDecodeCacheService.RequiresDedicatedDecoder("sample.jpg"), "JPG should keep the existing decoder path");

AssertLessThan(
    AiNsfwClassifierService.CalculateNudeDetectionScore(new Dictionary<string, double>
    {
        ["nude.female_breast_exposed"] = 0.96d,
        ["nude.explicit_area"] = 0.08d,
        ["nude.explicit_count"] = 1d / 3d,
    }),
    0.5d,
    "breast single-point detection should not become strong explicit NSFW");

AssertLessThan(
    AiNsfwClassifierService.CalculateNudeDetectionScore(new Dictionary<string, double>
    {
        ["nude.buttocks_exposed"] = 0.94d,
        ["nude.explicit_area"] = 0.07d,
        ["nude.explicit_count"] = 1d / 3d,
    }),
    0.5d,
    "buttocks single-point detection should not become strong explicit NSFW");

AssertGreaterThanOrEqual(
    AiNsfwClassifierService.CalculateNudeDetectionScore(new Dictionary<string, double>
    {
        ["nude.genitalia_exposed"] = 0.82d,
        ["nude.explicit_area"] = 0.2d,
        ["nude.explicit_count"] = 1d / 3d,
    }),
    0.78d,
    "genitalia evidence should remain strong explicit NSFW");

AssertGreaterThanOrEqual(
    AiNsfwClassifierService.CalculateNudeDetectionScore(new Dictionary<string, double>
    {
        ["nude.anus_exposed"] = 0.78d,
        ["nude.explicit_area"] = 0.18d,
        ["nude.explicit_count"] = 1d / 3d,
    }),
    0.74d,
    "anus evidence should remain strong explicit NSFW");

AssertLessThan(
    AiNsfwClassifierService.CalculateExplicitNsfwScore(new Dictionary<string, double>
    {
        ["porn"] = 1d,
        ["hentai"] = 0d,
        ["sexy"] = 0d,
        ["neutral"] = 0d,
        ["nude.detection_count"] = 0d,
        ["nude.explicit_area"] = 0d,
    }),
    0.5d,
    "generic porn score alone should not become strong explicit NSFW when NudeDetector finds no evidence");

AssertLessThan(
    AiNsfwClassifierService.CalculateWeightedNsfwScore(new Dictionary<string, double>
    {
        ["porn"] = 1d,
        ["hentai"] = 0d,
        ["sexy"] = 0d,
        ["neutral"] = 0d,
        ["nude.detection_count"] = 0d,
        ["nude.explicit_area"] = 0d,
    }),
    0.6d,
    "generic porn score alone should not cross the default NSFW decision threshold");

AssertGreaterThanOrEqual(
    AiNsfwClassifierService.CalculateExplicitNsfwScore(new Dictionary<string, double>
    {
        ["porn"] = 0.92d,
        ["hentai"] = 0d,
        ["sexy"] = 0.35d,
        ["nude.genitalia_exposed"] = 0.72d,
        ["nude.explicit_area"] = 0.22d,
        ["nude.detection_count"] = 0.25d,
    }),
    0.78d,
    "generic porn score with explicit NudeDetector support should remain NSFW");

AssertFalse(
    WorkspaceService.CanPersonalAiPromoteNsfwForScores(
        explicitNsfwScore: 0d,
        suggestiveScore: 1d,
        hardNsfwSampleEvidence: false,
        strongLocalPersonalNsfw: true),
    "personal AI should not strongly promote NSFW when explicit evidence is zero");

AssertTrue(
    WorkspaceService.ShouldApplySuggestiveOnlySfwGuard(
        finalLabelIsNsfw: true,
        explicitNsfwScore: 0d,
        suggestiveScore: 1d,
        hardNsfwSampleEvidence: false,
        hasPersonalLoraNsfwOverride: false),
    "suggestive-only NSFW result should be guarded as SFW");

AssertTrue(
    WorkspaceService.CanPersonalAiPromoteNsfwForScores(
        explicitNsfwScore: 0.19d,
        suggestiveScore: 0.2d,
        hardNsfwSampleEvidence: false,
        strongLocalPersonalNsfw: false),
    "explicit evidence should still allow NSFW promotion");

AssertTrue(
    WorkspaceService.CanPersonalAiPromoteNsfwForScores(
        explicitNsfwScore: 0d,
        suggestiveScore: 1d,
        hardNsfwSampleEvidence: true,
        strongLocalPersonalNsfw: false),
    "hard NSFW sample evidence should still allow NSFW promotion");

AssertTrue(
    PersonalAiTrainingService.ShouldRecordTrainingSample(
        new PersonalAiModelSettings
        {
            TrainingRoots =
            [
                new PersonalAiTrainingRoot { FolderPath = @"D:\Wallpapers\SFW", Label = ImageLabel.Sfw },
                new PersonalAiTrainingRoot { FolderPath = @"D:\Wallpapers\NSFW", Label = ImageLabel.Nsfw },
            ],
        },
        @"D:\Wallpapers\SFW\a.jpg",
        ImageLabel.Sfw),
    "SFW sample under the durable SFW root should be recorded");

AssertEqual(
    @"D:\@Software\JudgePicSFW",
    AppRuntimePaths.ResolveApplicationFolder(
        @"D:\@Software\JudgePicSFW\JudgePicSFW.exe",
        @"C:\Users\freez\AppData\Local\Temp\.net\JudgePicSFW\extracted\"),
    "single-file exe runtime data should resolve beside the real exe, not the temporary extraction folder");

AssertEqual(
    Path.Combine(@"D:\@Software\JudgePicSFW", "Data", "PersonalAi"),
    new PersonalAiTrainingService(new PerformanceLogService()).ResolveRootFolder(
        new PersonalAiModelSettings
        {
            RootFolder = string.Empty,
        },
        @"D:\@Software\JudgePicSFW\JudgePicSFW.exe",
        @"C:\Users\freez\AppData\Local\Temp\.net\JudgePicSFW\extracted\"),
    "default personal AI root should live under exe-side Data folder");

var environmentInstallProgress = PersonalAiTrainingService.CreateEnvironmentInstallProgressFromLine(
    "ENV_PROGRESS {\"current\":2,\"total\":6,\"step\":\"安装 PyTorch\",\"detail\":\"正在通过清华镜像安装稳定版 PyTorch。\"}");
AssertTrue(environmentInstallProgress is not null, "environment installer progress lines should be parsed");
AssertEqual(
    WorkspaceProgressStage.TrainingPersonalModel,
    environmentInstallProgress!.Stage,
    "environment installer progress should use personal training stage");
AssertEqual(
    2,
    environmentInstallProgress.Current,
    "environment installer progress should preserve current step");
AssertEqual(
    "安装 PyTorch",
    environmentInstallProgress.CurrentItem,
    "environment installer progress should expose the current installer step");

var predictionProgress = PersonalAiTrainingService.CreatePredictionProgressFromLine(
    "PREDICT_PROGRESS {\"current\":128,\"total\":3982,\"currentItem\":\"a.jpg\",\"device\":\"cuda\",\"deviceDisplay\":\"CUDA (RTX 3060)\",\"elapsedSeconds\":12.5}");
AssertTrue(predictionProgress is not null, "personal LoRA prediction progress lines should be parsed");
AssertEqual(
    WorkspaceProgressStage.AnalyzingSource,
    predictionProgress!.Stage,
    "personal LoRA prediction progress should use source analysis stage");
AssertEqual(
    128,
    predictionProgress.Current,
    "personal LoRA prediction progress should preserve completed image count");
AssertEqual(
    3982,
    predictionProgress.Total,
    "personal LoRA prediction progress should preserve total image count");
AssertEqual(
    "a.jpg",
    predictionProgress.CurrentItem,
    "personal LoRA prediction progress should expose the current file name");
AssertEqual(
    2,
    PersonalAiTrainingService.NormalizePersonalAiBatchSize(8),
    "legacy personal AI batch size 8 should be migrated down to the conservative batch size 2");
AssertEqual(
    4,
    PersonalAiTrainingService.NormalizePersonalAiBatchSize(4),
    "personal AI batch size 4 should remain available after manual monitoring confirms enough GPU memory");
AssertEqual(
    16,
    PersonalAiTrainingService.NormalizePersonalAiPredictionBatchSize(8),
    "legacy personal AI batch size 8 should use monitored batch size 16 for prediction");
AssertEqual(
    16,
    PersonalAiTrainingService.NormalizePersonalAiPredictionBatchSize(2),
    "personal AI prediction should use batch size 16 even when training stays at batch size 2");

AssertFalse(
    PersonalAiTrainingService.ShouldRecordTrainingSample(
        new PersonalAiModelSettings
        {
            TrainingRoots =
            [
                new PersonalAiTrainingRoot { FolderPath = @"D:\Wallpapers\SFW", Label = ImageLabel.Sfw },
                new PersonalAiTrainingRoot { FolderPath = @"D:\Wallpapers\NSFW", Label = ImageLabel.Nsfw },
            ],
        },
        @"E:\PhoneWallpapers\temp.jpg",
        ImageLabel.Nsfw),
    "phone wallpaper outside durable roots should not be recorded for training");

AssertFalse(
    PersonalAiTrainingService.ShouldRecordTrainingSample(
        new PersonalAiModelSettings
        {
            TrainingRoots =
            [
                new PersonalAiTrainingRoot { FolderPath = @"D:\Wallpapers\SFW", Label = ImageLabel.Sfw },
            ],
        },
        @"D:\Wallpapers\SFW\actually-nsfw.jpg",
        ImageLabel.Nsfw),
    "a label must match the durable training root label");

AssertEqual(
    2,
    WorkspaceService.MergeTrainingRootsFromSampleFolders(
        new PersonalAiModelSettings(),
        [
            new SampleFolderConfig { FolderPath = @"D:\Wallpapers\SFW", Label = ImageLabel.Sfw, IsEnabled = true },
            new SampleFolderConfig { FolderPath = @"D:\Wallpapers\NSFW", Label = ImageLabel.Nsfw, IsEnabled = true },
            new SampleFolderConfig { FolderPath = @"E:\PhoneWallpapers", Label = ImageLabel.Unknown, IsEnabled = true },
        ]).TrainingRoots.Count,
    "only enabled SFW/NSFW sample folders should become durable training roots");

var desktopOnlyRoots = WorkspaceService.MergeTrainingRootsFromSampleFolders(
    new PersonalAiModelSettings
    {
        TrainingRoots =
        [
            new PersonalAiTrainingRoot { FolderPath = @"E:\@手机壁纸-待处理\@手机壁纸-SFW", Label = ImageLabel.Sfw },
        ],
    },
    [
        new SampleFolderConfig { FolderPath = @"E:\@手机壁纸-待处理\@手机壁纸-SFW", Label = ImageLabel.Sfw, IsEnabled = true },
        new SampleFolderConfig { FolderPath = @"E:\@百看不厌\@电脑壁纸\SFW", Label = ImageLabel.Sfw, IsEnabled = true },
        new SampleFolderConfig { FolderPath = @"E:\@手机壁纸-待处理\@手机壁纸-NSFW", Label = ImageLabel.Nsfw, IsEnabled = true },
        new SampleFolderConfig { FolderPath = @"E:\@百看不厌\@电脑壁纸\NSFW", Label = ImageLabel.Nsfw, IsEnabled = true },
    ]).TrainingRoots;
AssertEqual(2, desktopOnlyRoots.Count, "desktop wallpaper sample roots should replace temporary phone wallpaper training roots");
AssertFalse(
    desktopOnlyRoots.Any(root => root.FolderPath.Contains("手机壁纸", StringComparison.OrdinalIgnoreCase)),
    "temporary phone wallpaper folders should not be durable training roots when desktop wallpaper roots exist");

var desktopLearningSettings = new PersonalAiModelSettings
{
    TrainingRoots =
    [
        new PersonalAiTrainingRoot { FolderPath = @"E:\@百看不厌\@电脑壁纸\SFW", Label = ImageLabel.Sfw },
        new PersonalAiTrainingRoot { FolderPath = @"E:\@百看不厌\@电脑壁纸\NSFW", Label = ImageLabel.Nsfw },
    ],
};
AssertTrue(
    WorkspaceService.ShouldUseLearnedRecordForPersonalLearning(
        new LearnedImageRecord
        {
            TaskMode = ClassificationTaskMode.ContentSafety,
            LabelOrigin = LabelOrigin.SampleLibrary,
            CurrentLabel = ImageLabel.Sfw,
            LastKnownPath = @"E:\@百看不厌\@电脑壁纸\SFW\a.jpg",
        },
        ClassificationTaskMode.ContentSafety,
        desktopLearningSettings),
    "desktop wallpaper sample-library records should remain available for personal learning");
AssertFalse(
    WorkspaceService.ShouldUseLearnedRecordForPersonalLearning(
        new LearnedImageRecord
        {
            TaskMode = ClassificationTaskMode.ContentSafety,
            LabelOrigin = LabelOrigin.SampleLibrary,
            CurrentLabel = ImageLabel.Sfw,
            LastKnownPath = @"E:\@手机壁纸-待处理\@手机壁纸-SFW\a.jpg",
        },
        ClassificationTaskMode.ContentSafety,
        desktopLearningSettings),
    "temporary phone sample-library records should not feed personal learning when desktop roots exist");
AssertTrue(
    WorkspaceService.ShouldUseLearnedRecordForPersonalLearning(
        new LearnedImageRecord
        {
            TaskMode = ClassificationTaskMode.ContentSafety,
            LabelOrigin = LabelOrigin.ManualCorrection,
            CurrentLabel = ImageLabel.Nsfw,
            LastKnownPath = @"E:\@手机壁纸-待处理\@手机壁纸-Desktop\a.jpg",
        },
        ClassificationTaskMode.ContentSafety,
        desktopLearningSettings),
    "manual corrections should stay available as high-priority personal learning records");
AssertFalse(
    WorkspaceService.ShouldUseLearnedRecordForPersonalLearning(
        new LearnedImageRecord
        {
            TaskMode = ClassificationTaskMode.ContentSafety,
            LabelOrigin = LabelOrigin.ConfirmedMove,
            CurrentLabel = ImageLabel.Sfw,
            LastKnownPath = @"E:\@手机壁纸-待处理\@手机壁纸-SFW\a.jpg",
        },
        ClassificationTaskMode.ContentSafety,
        desktopLearningSettings),
    "ordinary phone wallpaper confirmations should not feed long-term personal learning");

await AssertFilteredTrainingDatasetUsesDurableRootsOnlyAsync();
await AssertManualMistakeUsesLightweightAssetOutsideDurableRootsAsync();
await AssertHistoricalManualCorrectionsBackfillIntoMistakeBookAsync();
AssertTrainingContinuationPrefersActiveModelOverPendingCandidate();

AssertEqual(
    2,
    WorkspaceService.SelectEvaluationSamples(
        [],
        [
            new LearnedImageRecord { ContentId = "c", CurrentLabel = ImageLabel.Sfw, LastKnownPath = @"D:\Wallpapers\SFW\c.jpg" },
            new LearnedImageRecord { ContentId = "a", CurrentLabel = ImageLabel.Sfw, LastKnownPath = @"D:\Wallpapers\SFW\a.jpg" },
            new LearnedImageRecord { ContentId = "b", CurrentLabel = ImageLabel.Sfw, LastKnownPath = @"D:\Wallpapers\SFW\b.jpg" },
        ],
        2).Count,
    "evaluation set selection should cap samples per label");

var eligibleReport = WorkspaceService.BuildPersonalAiEvaluationReport(
    "personal-lora-test",
    BuildEvaluationSamples(60, 60),
    BuildEvaluationPredictions(60, 60, sfwFalsePositiveCount: 2, nsfwFalseNegativeCount: 1),
    new PersonalAiModelSettings
    {
        MinimumConfidence = 0.56d,
        DecisionThreshold = 0.5d,
        EvaluationSamplesPerLabel = 300,
        PrimaryModelMinimumAccuracy = 0.95d,
        PrimaryModelMaximumSfwFalsePositiveRate = 0.05d,
        PrimaryModelMaximumNsfwFalseNegativeRate = 0.03d,
    });
AssertTrue(
    eligibleReport.Summary.AllowsPersonalModelPrimary,
    "evaluation report should allow primary mode when accuracy and error rates are strong");
AssertTrue(
    WorkspaceService.IsPersonalModelPrimaryEligible(
        eligibleReport.Summary,
        new PersonalAiModelSettings
        {
            PrimaryModelMinimumAccuracy = 0.95d,
            PrimaryModelMaximumSfwFalsePositiveRate = 0.05d,
            PrimaryModelMaximumNsfwFalseNegativeRate = 0.03d,
        }),
    "primary eligibility should follow the persisted evaluation summary");

var defaultedThresholdReport = WorkspaceService.BuildPersonalAiEvaluationReport(
    "personal-lora-test",
    BuildEvaluationSamples(60, 60),
    BuildEvaluationPredictions(60, 60, sfwFalsePositiveCount: 2, nsfwFalseNegativeCount: 1),
    new PersonalAiModelSettings
    {
        MinimumConfidence = 0d,
        DecisionThreshold = 0d,
        EvaluationSamplesPerLabel = 0,
        PrimaryModelMinimumAccuracy = 0d,
        PrimaryModelMaximumSfwFalsePositiveRate = 0d,
        PrimaryModelMaximumNsfwFalseNegativeRate = 0d,
    });
AssertTrue(
    defaultedThresholdReport.Summary.AllowsPersonalModelPrimary,
    "evaluation report should use safe defaults when older settings contain zero thresholds");

var weakReport = WorkspaceService.BuildPersonalAiEvaluationReport(
    "personal-lora-test",
    BuildEvaluationSamples(60, 60),
    BuildEvaluationPredictions(60, 60, sfwFalsePositiveCount: 1, nsfwFalseNegativeCount: 8),
    new PersonalAiModelSettings
    {
        MinimumConfidence = 0.56d,
        DecisionThreshold = 0.5d,
        EvaluationSamplesPerLabel = 300,
        PrimaryModelMinimumAccuracy = 0.9d,
        PrimaryModelMaximumSfwFalsePositiveRate = 0.05d,
        PrimaryModelMaximumNsfwFalseNegativeRate = 0.03d,
    });
AssertFalse(
    weakReport.Summary.AllowsPersonalModelPrimary,
    "evaluation report should block primary mode when NSFW false negatives are too high");

AssertThrowsInvalidOperation(
    () => PersonalAiTrainingService.EnsureTrainingProducedCandidateModel(new PersonalAiTrainingStatus
    {
        CandidateModelVersion = string.Empty,
        HasResumeCheckpoint = true,
        ResumeEpoch = 1,
        ResumeBatch = 51,
    }),
    "第 1 轮、第 51 批",
    "training without a candidate model should surface resumable checkpoint progress");

AssertThrowsInvalidOperation(
    () => PersonalAiTrainingService.EnsureTrainingProducedCandidateModel(new PersonalAiTrainingStatus
    {
        CandidateModelVersion = "personal-lora-20260712-110101",
        HasResumeCheckpoint = true,
        ResumeEpoch = 1,
        ResumeBatch = 51,
    }),
    null,
    "training with a candidate model should be accepted");

await AssertPreservesCreatedAtForUnchangedSampleAsync();

if (failures.Count > 0)
{
    foreach (var failure in failures)
    {
        Console.Error.WriteLine(failure);
    }

    return 1;
}

Console.WriteLine("classification scoring tests passed.");
return 0;

void AssertLessThan(double actual, double expectedExclusiveMaximum, string message)
{
    if (actual >= expectedExclusiveMaximum)
    {
        failures.Add($"{message}: actual {actual:P1}, expected less than {expectedExclusiveMaximum:P0}.");
    }
}

void AssertGreaterThanOrEqual(double actual, double expectedMinimum, string message)
{
    if (actual < expectedMinimum)
    {
        failures.Add($"{message}: actual {actual:P1}, expected at least {expectedMinimum:P0}.");
    }
}

void AssertTrue(bool actual, string message)
{
    if (!actual)
    {
        failures.Add($"{message}: expected true.");
    }
}

void AssertFalse(bool actual, string message)
{
    if (actual)
    {
        failures.Add($"{message}: expected false.");
    }
}

void AssertEqual<T>(T expected, T actual, string message)
{
    if (!EqualityComparer<T>.Default.Equals(expected, actual))
    {
        failures.Add($"{message}: actual {actual}, expected {expected}.");
    }
}

void AssertThrowsInvalidOperation(Action action, string? expectedMessageFragment, string message)
{
    try
    {
        action();
        if (expectedMessageFragment is not null)
        {
            failures.Add($"{message}: expected InvalidOperationException containing '{expectedMessageFragment}'.");
        }
    }
    catch (InvalidOperationException exception)
    {
        if (expectedMessageFragment is null)
        {
            failures.Add($"{message}: did not expect InvalidOperationException, but got '{exception.Message}'.");
            return;
        }

        if (!exception.Message.Contains(expectedMessageFragment, StringComparison.Ordinal))
        {
            failures.Add($"{message}: actual '{exception.Message}', expected to contain '{expectedMessageFragment}'.");
        }
    }
}

async Task AssertPreservesCreatedAtForUnchangedSampleAsync()
{
    var rootFolder = Path.Combine(Path.GetTempPath(), $"JudgePicSFW-Test-{Guid.NewGuid():N}");
    Directory.CreateDirectory(rootFolder);
    var datasetPath = Path.Combine(rootFolder, "training-labels.jsonl");

    try
    {
        const string existingCreatedAt = "2026-07-11T00:00:00.0000000Z";
        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
        var existingSample = new PersonalAiTrainingSample
        {
            FilePath = @"E:\Samples\same-file.jpg",
            ContentId = "old-content-id",
            Label = ImageLabel.Sfw,
            Source = "sample_library",
            CreatedAtUtc = DateTime.Parse(existingCreatedAt, null, System.Globalization.DateTimeStyles.RoundtripKind),
        };

        await File.WriteAllTextAsync(
            datasetPath,
            JsonSerializer.Serialize(existingSample, serializerOptions) + Environment.NewLine);

        var service = new PersonalAiTrainingService(new PerformanceLogService());
        await service.RecordTrainingSamplesAsync(
            new PersonalAiModelSettings
            {
                IsEnabled = true,
                RootFolder = rootFolder,
            },
            [(@"E:\Samples\same-file.jpg", "old-content-id", ImageLabel.Sfw, "sample_library")],
            CancellationToken.None);

        var updatedLine = (await File.ReadAllLinesAsync(datasetPath)).FirstOrDefault(line => !string.IsNullOrWhiteSpace(line));
        if (string.IsNullOrWhiteSpace(updatedLine))
        {
            failures.Add("unchanged sample timestamp should be preserved: training-labels.jsonl was unexpectedly empty.");
            return;
        }

        var updatedSample = JsonSerializer.Deserialize<PersonalAiTrainingSample>(updatedLine, serializerOptions);
        if (updatedSample is null)
        {
            failures.Add("unchanged sample timestamp should be preserved: updated sample could not be deserialized.");
            return;
        }

        if (updatedSample.CreatedAtUtc.ToUniversalTime().ToString("O") != existingCreatedAt)
        {
            failures.Add($"unchanged sample timestamp should be preserved: actual {updatedSample.CreatedAtUtc:O}, expected {existingCreatedAt}.");
        }
    }
    finally
    {
        if (File.Exists(datasetPath))
        {
            File.Delete(datasetPath);
        }

        if (Directory.Exists(rootFolder))
        {
            Directory.Delete(rootFolder, false);
        }
    }
}

async Task AssertFilteredTrainingDatasetUsesDurableRootsOnlyAsync()
{
    var rootFolder = Path.Combine(Path.GetTempPath(), $"JudgePicSFW-FilterTest-{Guid.NewGuid():N}");
    var sfwRoot = Path.Combine(rootFolder, "DesktopSfw");
    var nsfwRoot = Path.Combine(rootFolder, "DesktopNsfw");
    var phoneRoot = Path.Combine(rootFolder, "PhoneTemp");
    Directory.CreateDirectory(sfwRoot);
    Directory.CreateDirectory(nsfwRoot);
    Directory.CreateDirectory(phoneRoot);

    var sfwFile = Path.Combine(sfwRoot, "safe.jpg");
    var nsfwFile = Path.Combine(nsfwRoot, "manual.jpg");
    var phoneFile = Path.Combine(phoneRoot, "temporary.jpg");
    await File.WriteAllTextAsync(sfwFile, "sfw");
    await File.WriteAllTextAsync(nsfwFile, "nsfw");
    await File.WriteAllTextAsync(phoneFile, "phone");

    try
    {
        var filtered = PersonalAiTrainingService.SelectTrainingSamplesForRequest(
            new PersonalAiModelSettings
            {
                TrainingRoots =
                [
                    new PersonalAiTrainingRoot { FolderPath = sfwRoot, Label = ImageLabel.Sfw },
                    new PersonalAiTrainingRoot { FolderPath = nsfwRoot, Label = ImageLabel.Nsfw },
                ],
            },
            [
                new PersonalAiTrainingSample
                {
                    FilePath = sfwFile,
                    ContentId = "desktop-sfw",
                    Label = ImageLabel.Sfw,
                    Source = "sample_library",
                },
                new PersonalAiTrainingSample
                {
                    FilePath = phoneFile,
                    ContentId = "phone-temp",
                    Label = ImageLabel.Nsfw,
                    Source = "confirmed_move",
                },
                new PersonalAiTrainingSample
                {
                    FilePath = nsfwFile,
                    ContentId = "same-content",
                    Label = ImageLabel.Sfw,
                    Source = "sample_library",
                },
                new PersonalAiTrainingSample
                {
                    FilePath = nsfwFile,
                    ContentId = "same-content",
                    Label = ImageLabel.Nsfw,
                    Source = "manual_correction",
                },
            ],
            requireExistingFiles: true);

        AssertEqual(2, filtered.Count, "filtered training request should exclude temporary phone wallpaper samples");
        AssertTrue(
            filtered.Any(sample => sample.ContentId == "desktop-sfw" && sample.Label == ImageLabel.Sfw),
            "filtered training request should keep desktop SFW samples");
        AssertTrue(
            filtered.Any(sample => sample.ContentId == "same-content" && sample.Label == ImageLabel.Nsfw && sample.Source == "manual_correction"),
            "filtered training request should keep the highest-priority manual correction for a duplicated content id");
        AssertFalse(
            filtered.Any(sample => sample.ContentId == "phone-temp"),
            "filtered training request should not include phone wallpaper paths outside durable roots");
    }
    finally
    {
        foreach (var filePath in new[] { sfwFile, nsfwFile, phoneFile })
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }

        foreach (var folderPath in new[] { sfwRoot, nsfwRoot, phoneRoot, rootFolder })
        {
            if (Directory.Exists(folderPath))
            {
                Directory.Delete(folderPath, false);
            }
        }
    }
}

async Task AssertManualMistakeUsesLightweightAssetOutsideDurableRootsAsync()
{
    var rootFolder = Path.Combine(Path.GetTempPath(), $"JudgePicSFW-MistakeBookTest-{Guid.NewGuid():N}");
    var personalAiRoot = Path.Combine(rootFolder, "PersonalAi");
    var desktopSfwRoot = Path.Combine(rootFolder, "DesktopSfw");
    var desktopNsfwRoot = Path.Combine(rootFolder, "DesktopNsfw");
    var phoneRoot = Path.Combine(rootFolder, "PhoneTemp");
    Directory.CreateDirectory(desktopSfwRoot);
    Directory.CreateDirectory(desktopNsfwRoot);
    Directory.CreateDirectory(phoneRoot);

    var phoneImagePath = Path.Combine(phoneRoot, "manual-correction.jpg");
    using (var bitmap = new Bitmap(32, 20))
    using (var graphics = Graphics.FromImage(bitmap))
    {
        graphics.Clear(Color.HotPink);
        bitmap.Save(phoneImagePath, ImageFormat.Jpeg);
    }

    try
    {
        var service = new PersonalAiTrainingService(new PerformanceLogService());
        var settings = new PersonalAiModelSettings
        {
            IsEnabled = true,
            RootFolder = personalAiRoot,
            TrainingRoots =
            [
                new PersonalAiTrainingRoot { FolderPath = desktopSfwRoot, Label = ImageLabel.Sfw },
                new PersonalAiTrainingRoot { FolderPath = desktopNsfwRoot, Label = ImageLabel.Nsfw },
            ],
        };

        await service.RecordManualCorrectionSampleAsync(
            settings,
            phoneImagePath,
            "phone-content-id",
            ImageLabel.Sfw,
            ImageLabel.Uncertain,
            LabelOrigin.ModelPrediction,
            CancellationToken.None);

        var datasetPath = Path.Combine(personalAiRoot, "training-labels.jsonl");
        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
        var datasetSamples = (await File.ReadAllLinesAsync(datasetPath))
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => JsonSerializer.Deserialize<PersonalAiTrainingSample>(line, serializerOptions)!)
            .ToList();

        AssertEqual(1, datasetSamples.Count, "manual mistake should write one high-priority training sample");
        var mistakeSample = datasetSamples[0];
        AssertEqual(ImageLabel.Sfw, mistakeSample.Label, "manual mistake should keep the corrected SFW label");
        AssertEqual("mistake_book_manual", mistakeSample.Source, "manual mistake should use mistake-book source");
        AssertFalse(
            string.Equals(Path.GetFullPath(phoneImagePath), Path.GetFullPath(mistakeSample.FilePath), StringComparison.OrdinalIgnoreCase),
            "manual mistake should not train from the temporary phone wallpaper path when a lightweight asset can be created");
        AssertTrue(File.Exists(mistakeSample.FilePath), "manual mistake lightweight asset should exist on disk");
        AssertTrue(
            mistakeSample.FilePath.Contains("mistake-assets", StringComparison.OrdinalIgnoreCase),
            "manual mistake lightweight asset should live under the personal AI mistake-assets folder");

        var filtered = PersonalAiTrainingService.SelectTrainingSamplesForRequest(
            settings,
            datasetSamples,
            requireExistingFiles: true);

        AssertEqual(1, filtered.Count, "manual mistake outside durable roots should still be selected for LoRA training through its lightweight asset");
        AssertEqual("mistake_book_manual", filtered[0].Source, "manual mistake source should survive training request filtering");
        AssertTrue(File.Exists(Path.Combine(personalAiRoot, "mistakes.jsonl")), "manual mistake should be recorded in the mistake book index");
    }
    finally
    {
        if (Directory.Exists(rootFolder))
        {
            foreach (var filePath in Directory.EnumerateFiles(rootFolder, "*", SearchOption.AllDirectories).ToArray())
            {
                File.Delete(filePath);
            }

            foreach (var folderPath in Directory.EnumerateDirectories(rootFolder, "*", SearchOption.AllDirectories)
                         .OrderByDescending(path => path.Length)
                         .Concat([rootFolder])
                         .ToArray())
            {
                Directory.Delete(folderPath, false);
            }
        }
    }
}

async Task AssertHistoricalManualCorrectionsBackfillIntoMistakeBookAsync()
{
    var rootFolder = Path.Combine(Path.GetTempPath(), $"JudgePicSFW-BackfillTest-{Guid.NewGuid():N}");
    var personalAiRoot = Path.Combine(rootFolder, "PersonalAi");
    var desktopSfwRoot = Path.Combine(rootFolder, "DesktopSfw");
    var desktopNsfwRoot = Path.Combine(rootFolder, "DesktopNsfw");
    var phoneRoot = Path.Combine(rootFolder, "PhoneTemp");
    Directory.CreateDirectory(desktopSfwRoot);
    Directory.CreateDirectory(desktopNsfwRoot);
    Directory.CreateDirectory(phoneRoot);

    var manualImagePath = Path.Combine(phoneRoot, "historical-manual.jpg");
    using (var bitmap = new Bitmap(32, 20))
    using (var graphics = Graphics.FromImage(bitmap))
    {
        graphics.Clear(Color.DeepSkyBlue);
        bitmap.Save(manualImagePath, ImageFormat.Jpeg);
    }

    try
    {
        var state = new PersistedAppState();
        state.LearnedImagesById["manual-content-id"] = new LearnedImageRecord
        {
            ContentId = "manual-content-id",
            TaskMode = ClassificationTaskMode.ContentSafety,
            CurrentLabel = ImageLabel.Nsfw,
            LabelOrigin = LabelOrigin.ManualCorrection,
            LastKnownPath = manualImagePath,
            CorrectionCount = 1,
            UpdatedAtUtc = DateTime.UtcNow.AddMinutes(-20),
        };
        state.LearnedImagesById["ordinary-confirmed-id"] = new LearnedImageRecord
        {
            ContentId = "ordinary-confirmed-id",
            TaskMode = ClassificationTaskMode.ContentSafety,
            CurrentLabel = ImageLabel.Sfw,
            LabelOrigin = LabelOrigin.ConfirmedMove,
            LastKnownPath = manualImagePath,
            UpdatedAtUtc = DateTime.UtcNow.AddMinutes(-10),
        };

        var trainingService = new PersonalAiTrainingService(new PerformanceLogService());
        var workspaceService = new WorkspaceService(
            new AppStateStore(),
            new ImageFingerprintService(),
            new AiNsfwClassifierService(new PerformanceLogService()),
            trainingService,
            new WindowsShellFileMoveService(),
            new PerformanceLogService());
        var settings = new PersonalAiModelSettings
        {
            IsEnabled = true,
            RootFolder = personalAiRoot,
            TrainingRoots =
            [
                new PersonalAiTrainingRoot { FolderPath = desktopSfwRoot, Label = ImageLabel.Sfw },
                new PersonalAiTrainingRoot { FolderPath = desktopNsfwRoot, Label = ImageLabel.Nsfw },
            ],
        };

        var backfilledCount = await workspaceService.BackfillManualCorrectionsIntoMistakeBookAsync(
            state,
            settings,
            CancellationToken.None);
        var secondBackfillCount = await workspaceService.BackfillManualCorrectionsIntoMistakeBookAsync(
            state,
            settings,
            CancellationToken.None);

        AssertEqual(1, backfilledCount, "historical manual correction should be backfilled into the mistake book once");
        AssertEqual(0, secondBackfillCount, "historical manual correction backfill should be idempotent");

        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
        var datasetSamples = (await File.ReadAllLinesAsync(Path.Combine(personalAiRoot, "training-labels.jsonl")))
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => JsonSerializer.Deserialize<PersonalAiTrainingSample>(line, serializerOptions)!)
            .ToList();
        AssertEqual(1, datasetSamples.Count, "historical backfill should not import ordinary confirmed records");
        AssertEqual("mistake_book_manual", datasetSamples[0].Source, "historical backfill should use mistake-book training source");
        AssertEqual(ImageLabel.Nsfw, datasetSamples[0].Label, "historical backfill should keep the corrected manual label");
        AssertTrue(File.Exists(datasetSamples[0].FilePath), "historical backfill should create a lightweight training asset");
    }
    finally
    {
        if (Directory.Exists(rootFolder))
        {
            foreach (var filePath in Directory.EnumerateFiles(rootFolder, "*", SearchOption.AllDirectories).ToArray())
            {
                File.Delete(filePath);
            }

            foreach (var folderPath in Directory.EnumerateDirectories(rootFolder, "*", SearchOption.AllDirectories)
                         .OrderByDescending(path => path.Length)
                         .Concat([rootFolder])
                         .ToArray())
            {
                Directory.Delete(folderPath, false);
            }
        }
    }
}

void AssertTrainingContinuationPrefersActiveModelOverPendingCandidate()
{
    var rootFolder = Path.Combine(Path.GetTempPath(), $"JudgePicSFW-ContinuationTest-{Guid.NewGuid():N}");
    var activeModelFolder = Path.Combine(rootFolder, "models", "active-base");
    var candidateModelFolder = Path.Combine(rootFolder, "models", "pending-candidate");
    Directory.CreateDirectory(activeModelFolder);
    Directory.CreateDirectory(candidateModelFolder);

    try
    {
        var serializerOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
        File.WriteAllText(
            Path.Combine(rootFolder, "active-model.json"),
            JsonSerializer.Serialize(new
            {
                version = "personal-lora-active-base",
                modelPath = activeModelFolder,
                baseModelId = "google/vit-base-patch16-224",
                createdAt = "2026-07-12T09:00:00.0000000Z",
            }, serializerOptions));
        File.WriteAllText(
            Path.Combine(rootFolder, "candidate-model.json"),
            JsonSerializer.Serialize(new
            {
                version = "personal-lora-pending-candidate",
                modelPath = candidateModelFolder,
                baseModelId = "google/vit-base-patch16-224",
                createdAt = "2026-07-12T10:00:00.0000000Z",
            }, serializerOptions));

        var method = typeof(PersonalAiTrainingService).GetMethod(
            "ResolveTrainingContinuationModel",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        if (method is null)
        {
            failures.Add("training continuation should be testable: ResolveTrainingContinuationModel was not found.");
            return;
        }

        var selectedManifest = method.Invoke(null, [rootFolder]);
        var selectedVersion = selectedManifest?.GetType().GetProperty("Version")?.GetValue(selectedManifest)?.ToString();
        AssertEqual(
            "personal-lora-active-base",
            selectedVersion,
            "incremental training should continue from the enabled active model, not a newer pending candidate");
    }
    finally
    {
        if (Directory.Exists(rootFolder))
        {
            foreach (var filePath in Directory.EnumerateFiles(rootFolder, "*", SearchOption.AllDirectories).ToArray())
            {
                File.Delete(filePath);
            }

            foreach (var folderPath in Directory.EnumerateDirectories(rootFolder, "*", SearchOption.AllDirectories)
                         .OrderByDescending(path => path.Length)
                         .Concat([rootFolder])
                         .ToArray())
            {
                Directory.Delete(folderPath, false);
            }
        }
    }
}

static List<PersonalAiEvaluationSample> BuildEvaluationSamples(int sfwCount, int nsfwCount)
{
    var samples = new List<PersonalAiEvaluationSample>(sfwCount + nsfwCount);
    for (var index = 0; index < sfwCount; index++)
    {
        samples.Add(new PersonalAiEvaluationSample
        {
            ContentId = $"sfw-{index:D4}",
            FilePath = $@"D:\Wallpapers\SFW\{index:D4}.jpg",
            Label = ImageLabel.Sfw,
        });
    }

    for (var index = 0; index < nsfwCount; index++)
    {
        samples.Add(new PersonalAiEvaluationSample
        {
            ContentId = $"nsfw-{index:D4}",
            FilePath = $@"D:\Wallpapers\NSFW\{index:D4}.jpg",
            Label = ImageLabel.Nsfw,
        });
    }

    return samples;
}

static Dictionary<string, PersonalAiPredictionResult> BuildEvaluationPredictions(
    int sfwCount,
    int nsfwCount,
    int sfwFalsePositiveCount,
    int nsfwFalseNegativeCount)
{
    var predictions = new Dictionary<string, PersonalAiPredictionResult>(StringComparer.OrdinalIgnoreCase);
    for (var index = 0; index < sfwCount; index++)
    {
        var isFalsePositive = index < sfwFalsePositiveCount;
        predictions[$"sfw-{index:D4}"] = new PersonalAiPredictionResult
        {
            IsAvailable = true,
            ContentId = $"sfw-{index:D4}",
            NsfwProbability = isFalsePositive ? 0.8d : 0.1d,
            Confidence = 0.9d,
        };
    }

    for (var index = 0; index < nsfwCount; index++)
    {
        var isFalseNegative = index < nsfwFalseNegativeCount;
        predictions[$"nsfw-{index:D4}"] = new PersonalAiPredictionResult
        {
            IsAvailable = true,
            ContentId = $"nsfw-{index:D4}",
            NsfwProbability = isFalseNegative ? 0.1d : 0.85d,
            Confidence = 0.9d,
        };
    }

    return predictions;
}
