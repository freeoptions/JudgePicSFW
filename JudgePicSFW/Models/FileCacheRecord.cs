namespace JudgePicSFW.Models;

public sealed class FileCacheRecord
{
    public string FilePath { get; set; } = string.Empty;

    public long FileSize { get; set; }

    public DateTime LastWriteUtc { get; set; }

    public string ContentId { get; set; } = string.Empty;

    public string AverageHash { get; set; } = string.Empty;

    public ulong AverageHashBits { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }

    public DateTime LastSeenUtc { get; set; }
}
