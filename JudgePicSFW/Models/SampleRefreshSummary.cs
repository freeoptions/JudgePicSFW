namespace JudgePicSFW.Models;

public sealed class SampleRefreshSummary
{
    public int ScannedFiles { get; set; }

    public int ReusedCacheFiles { get; set; }

    public int ImportedFiles { get; set; }

    public int IgnoredFiles { get; set; }
}
