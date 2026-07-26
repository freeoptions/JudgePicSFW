using System.IO;
using System.Text;

namespace JudgePicSFW.Services;

public sealed class PerformanceLogService
{
    private readonly string _logFilePath;
    private readonly SemaphoreSlim _syncLock = new(1, 1);

    public PerformanceLogService()
    {
        var logFolder = AppRuntimePaths.ResolveDataPath("Logs");
        Directory.CreateDirectory(logFolder);
        _logFilePath = Path.Combine(logFolder, "performance.log");
    }

    public async Task LogAsync(string category, string message)
    {
        var line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] [{category}] {message}{Environment.NewLine}";
        await _syncLock.WaitAsync();

        try
        {
            await File.AppendAllTextAsync(_logFilePath, line, Encoding.UTF8);
        }
        finally
        {
            _syncLock.Release();
        }
    }
}
