namespace JudgePicSFW.Models;

public sealed class ImageFingerprint
{
    public required string ContentId { get; init; }

    public required string AverageHash { get; init; }

    public required ulong AverageHashBits { get; init; }

    public required int Width { get; init; }

    public required int Height { get; init; }
}
