using System.IO;
using System.Numerics;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class ImageFingerprintService
{
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
    };

    public bool IsSupportedImage(string path)
    {
        return SupportedExtensions.Contains(Path.GetExtension(path));
    }

    public Task<ImageFingerprint> CreateFingerprintAsync(string filePath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var fileInfo = new FileInfo(filePath);

        int width;
        int height;
        using (var metadataStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 4096, FileOptions.SequentialScan))
        {
            var decoder = BitmapDecoder.Create(metadataStream, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None);
            var frame = decoder.Frames[0];
            width = frame.PixelWidth;
            height = frame.PixelHeight;
        }

        BitmapSource thumbnailSource;
        using (var thumbnailStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 4096, FileOptions.SequentialScan))
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.CreateOptions = BitmapCreateOptions.IgnoreColorProfile;
            bitmap.DecodePixelWidth = 8;
            bitmap.DecodePixelHeight = 8;
            bitmap.StreamSource = thumbnailStream;
            bitmap.EndInit();
            bitmap.Freeze();
            thumbnailSource = bitmap;
        }

        var scaledBitmap = new TransformedBitmap(thumbnailSource, new ScaleTransform(8d / thumbnailSource.PixelWidth, 8d / thumbnailSource.PixelHeight));
        scaledBitmap.Freeze();

        var grayscaleBitmap = new FormatConvertedBitmap(scaledBitmap, PixelFormats.Gray8, null, 0);
        grayscaleBitmap.Freeze();

        var pixels = new byte[64];
        grayscaleBitmap.CopyPixels(pixels, 8, 0);

        var total = 0;
        for (var index = 0; index < pixels.Length; index++)
        {
            total += pixels[index];
        }

        var average = total / (double)pixels.Length;
        ulong averageHashBits = 0;
        var hashBuilder = new StringBuilder(64);
        for (var index = 0; index < pixels.Length; index++)
        {
            if (pixels[index] >= average)
            {
                averageHashBits |= 1UL << (63 - index);
                hashBuilder.Append('1');
            }
            else
            {
                hashBuilder.Append('0');
            }
        }

        cancellationToken.ThrowIfCancellationRequested();
        var contentId = BuildFastContentId(fileInfo.Length, width, height, averageHashBits);

        return Task.FromResult(new ImageFingerprint
        {
            ContentId = contentId,
            AverageHash = hashBuilder.ToString(),
            AverageHashBits = averageHashBits,
            Width = width,
            Height = height,
        });
    }

    public ulong ParseAverageHashBits(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return 0UL;
        }

        ulong bits = 0;
        var length = Math.Min(64, value.Length);
        for (var index = 0; index < length; index++)
        {
            if (value[index] == '1')
            {
                bits |= 1UL << (63 - index);
            }
        }

        return bits;
    }

    public int ComputeHammingDistance(ulong left, ulong right)
    {
        return BitOperations.PopCount(left ^ right);
    }

    public int ComputeHammingDistance(string left, string right)
    {
        return ComputeHammingDistance(ParseAverageHashBits(left), ParseAverageHashBits(right));
    }

    private static string BuildFastContentId(long fileSize, int width, int height, ulong averageHashBits)
    {
        return $"VISUAL-V2:{width}x{height}:{fileSize:X}:{averageHashBits:X16}";
    }
}
