using System.Collections.Concurrent;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Channels;
using ImageMagick;

namespace JudgePicSFW.Services;

public sealed class ImageDecodeCacheService : IDisposable
{
    public const int DefaultPreviewWidth = 1440;
    public const int DefaultProcessingWidth = 1024;
    public const long MaxCacheBytes = 512L * 1024L * 1024L;

    private static readonly HashSet<string> DedicatedDecoderExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".webp",
        ".heic",
        ".heif",
    };

    private readonly Channel<DecodeJob> _queue = Channel.CreateUnbounded<DecodeJob>(new UnboundedChannelOptions
    {
        SingleReader = true,
        SingleWriter = false,
        AllowSynchronousContinuations = false,
    });
    private readonly ConcurrentDictionary<string, Lazy<DecodeJob>> _pendingJobs = new(StringComparer.OrdinalIgnoreCase);
    private readonly CancellationTokenSource _shutdownTokenSource = new();
    private readonly Task _workerTask;
    private readonly object _cacheGate = new();
    private bool _disposed;

    public ImageDecodeCacheService()
    {
        CacheFolder = Path.Combine(AppRuntimePaths.ApplicationFolder, "PicCache");
        Directory.CreateDirectory(CacheFolder);
        _workerTask = Task.Run(ProcessQueueAsync);
    }

    public string CacheFolder { get; }

    public static bool RequiresDedicatedDecoder(string filePath)
    {
        return DedicatedDecoderExtensions.Contains(Path.GetExtension(filePath));
    }

    public Task<string?> QueuePreviewDecodeAsync(string filePath, CancellationToken cancellationToken = default)
    {
        return QueueJpegDecodeAsync(filePath, DefaultPreviewWidth, cancellationToken);
    }

    public Task<string?> ResolveForProcessingAsync(
        string filePath,
        int targetWidth = DefaultProcessingWidth,
        CancellationToken cancellationToken = default)
    {
        if (!RequiresDedicatedDecoder(filePath))
        {
            return Task.FromResult<string?>(filePath);
        }

        return QueueJpegDecodeAsync(filePath, targetWidth, cancellationToken);
    }

    public async Task<IReadOnlyList<string>> ResolveForProcessingAsync(
        IReadOnlyList<string> filePaths,
        int targetWidth = DefaultProcessingWidth,
        CancellationToken cancellationToken = default)
    {
        if (filePaths.Count == 0)
        {
            return [];
        }

        var resolvedPaths = new string[filePaths.Count];
        for (var index = 0; index < filePaths.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            resolvedPaths[index] = await ResolveForProcessingAsync(filePaths[index], targetWidth, cancellationToken)
                .ConfigureAwait(false) ?? filePaths[index];
        }

        return resolvedPaths;
    }

    public string? TryGetCachedJpegPath(string filePath, int targetWidth)
    {
        if (!TryBuildCacheEntry(filePath, targetWidth, out var entry) || !File.Exists(entry.CachePath))
        {
            return null;
        }

        TouchCacheFile(entry.CachePath);
        return entry.CachePath;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _queue.Writer.TryComplete();
        _shutdownTokenSource.Cancel();
        try
        {
            _workerTask.GetAwaiter().GetResult();
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
        }

        _shutdownTokenSource.Dispose();
    }

    private Task<string?> QueueJpegDecodeAsync(string filePath, int targetWidth, CancellationToken cancellationToken)
    {
        if (!File.Exists(filePath))
        {
            return Task.FromResult<string?>(null);
        }

        if (!TryBuildCacheEntry(filePath, targetWidth, out var entry))
        {
            return Task.FromResult<string?>(null);
        }

        var cachedPath = TryGetCachedJpegPath(filePath, targetWidth);
        if (cachedPath is not null)
        {
            return Task.FromResult<string?>(cachedPath);
        }

        if (_disposed)
        {
            return Task.FromResult<string?>(null);
        }

        var lazyJob = _pendingJobs.GetOrAdd(
            entry.CacheKey,
            _ => new Lazy<DecodeJob>(
                () => CreateAndQueueJob(entry),
                LazyThreadSafetyMode.ExecutionAndPublication));
        var job = lazyJob.Value;
        return WaitWithCancellationAsync(job.Completion.Task, cancellationToken);
    }

    private DecodeJob CreateAndQueueJob(CacheEntry entry)
    {
        var job = new DecodeJob(entry, new TaskCompletionSource<string?>(TaskCreationOptions.RunContinuationsAsynchronously));
        if (!_queue.Writer.TryWrite(job))
        {
            _pendingJobs.TryRemove(entry.CacheKey, out _);
            job.Completion.TrySetResult(null);
        }

        return job;
    }

    private async Task ProcessQueueAsync()
    {
        try
        {
            await foreach (var job in _queue.Reader.ReadAllAsync(_shutdownTokenSource.Token).ConfigureAwait(false))
            {
                string? result = null;
                try
                {
                    result = ProcessDecodeJob(job.Entry);
                }
                catch
                {
                    result = null;
                }
                finally
                {
                    _pendingJobs.TryRemove(job.Entry.CacheKey, out _);
                    job.Completion.TrySetResult(result);
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
    }

    private string? ProcessDecodeJob(CacheEntry entry)
    {
        if (!File.Exists(entry.SourcePath))
        {
            return null;
        }

        var cachedPath = TryGetCachedJpegPath(entry.SourcePath, entry.TargetWidth);
        if (cachedPath is not null)
        {
            return cachedPath;
        }

        var currentInfo = new FileInfo(entry.SourcePath);
        if (!currentInfo.Exists ||
            currentInfo.Length != entry.SourceSize ||
            currentInfo.LastWriteTimeUtc != entry.SourceLastWriteUtc)
        {
            return null;
        }

        Directory.CreateDirectory(CacheFolder);
        var temporaryPath = Path.Combine(CacheFolder, $".{entry.CacheKey}.{Guid.NewGuid():N}.tmp");
        try
        {
            using var image = new MagickImage(entry.SourcePath);
            image.AutoOrient();
            if (entry.TargetWidth > 0 && image.Width > (uint)entry.TargetWidth)
            {
                image.Resize(new MagickGeometry((uint)entry.TargetWidth, 0u));
            }

            image.BackgroundColor = MagickColors.White;
            image.Alpha(AlphaOption.Remove);
            image.Quality = 88;
            image.Write(temporaryPath, MagickFormat.Jpeg);

            var writtenInfo = new FileInfo(temporaryPath);
            if (!writtenInfo.Exists || writtenInfo.Length <= 0 || writtenInfo.Length > MaxCacheBytes)
            {
                return null;
            }

            lock (_cacheGate)
            {
                File.Move(temporaryPath, entry.CachePath, overwrite: true);
                EnforceCacheLimit(entry.CachePath);
            }

            return File.Exists(entry.CachePath) ? entry.CachePath : null;
        }
        finally
        {
            TryDeleteFile(temporaryPath);
        }
    }

    private void EnforceCacheLimit(string protectedPath)
    {
        List<FileInfo> cacheFiles;
        try
        {
            cacheFiles = Directory
                .EnumerateFiles(CacheFolder, "*.jpg", SearchOption.TopDirectoryOnly)
                .Select(path => new FileInfo(path))
                .Where(info => info.Exists)
                .ToList();
        }
        catch
        {
            return;
        }

        var totalBytes = cacheFiles.Sum(info => info.Length);
        if (totalBytes <= MaxCacheBytes)
        {
            return;
        }

        foreach (var cacheFile in cacheFiles
                     .Where(info => !string.Equals(info.FullName, protectedPath, StringComparison.OrdinalIgnoreCase))
                     .OrderBy(info => info.LastAccessTimeUtc)
                     .ThenBy(info => info.LastWriteTimeUtc))
        {
            if (totalBytes <= MaxCacheBytes)
            {
                break;
            }

            try
            {
                totalBytes -= cacheFile.Length;
                cacheFile.Delete();
            }
            catch
            {
            }
        }
    }

    private bool TryBuildCacheEntry(string filePath, int targetWidth, out CacheEntry entry)
    {
        entry = null!;
        if (string.IsNullOrWhiteSpace(filePath) || targetWidth < 0)
        {
            return false;
        }

        try
        {
            var sourcePath = Path.GetFullPath(filePath);
            var sourceInfo = new FileInfo(sourcePath);
            if (!sourceInfo.Exists)
            {
                return false;
            }

            var cacheKeyText = string.Join(
                "|",
                "v1",
                sourcePath,
                sourceInfo.Length.ToString(System.Globalization.CultureInfo.InvariantCulture),
                sourceInfo.LastWriteTimeUtc.Ticks.ToString(System.Globalization.CultureInfo.InvariantCulture),
                targetWidth.ToString(System.Globalization.CultureInfo.InvariantCulture));
            var cacheKey = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(cacheKeyText))).ToLowerInvariant();
            entry = new CacheEntry(
                sourcePath,
                sourceInfo.Length,
                sourceInfo.LastWriteTimeUtc,
                targetWidth,
                cacheKey,
                Path.Combine(CacheFolder, $"{cacheKey}.jpg"));
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static async Task<string?> WaitWithCancellationAsync(Task<string?> task, CancellationToken cancellationToken)
    {
        if (!cancellationToken.CanBeCanceled)
        {
            return await task.ConfigureAwait(false);
        }

        return await task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private static void TouchCacheFile(string path)
    {
        try
        {
            File.SetLastAccessTimeUtc(path, DateTime.UtcNow);
        }
        catch
        {
        }
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
        }
    }

    private sealed record CacheEntry(
        string SourcePath,
        long SourceSize,
        DateTime SourceLastWriteUtc,
        int TargetWidth,
        string CacheKey,
        string CachePath);

    private sealed record DecodeJob(
        CacheEntry Entry,
        TaskCompletionSource<string?> Completion);
}
