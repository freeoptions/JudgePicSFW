using System.IO;
using System.Numerics;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ImageMagick;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class ImageFingerprintService
{
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".jfif",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".heic",
        ".heif",
    };

    public bool IsSupportedImage(string path)
    {
        return SupportedExtensions.Contains(Path.GetExtension(path));
    }

    public Task<ImageFingerprint> CreateFingerprintAsync(string filePath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(
            () => ImageDecodeCacheService.RequiresDedicatedDecoder(filePath)
                ? CreateDedicatedDecoderFingerprint(filePath, cancellationToken)
                : CreateWpfFingerprint(filePath, cancellationToken),
            cancellationToken);
    }

    private static ImageFingerprint CreateDedicatedDecoderFingerprint(string filePath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var fileInfo = new FileInfo(filePath);

        using var image = new MagickImage(filePath);
        image.AutoOrient();
        var width = checked((int)image.Width);
        var height = checked((int)image.Height);
        var thumbnailGeometry = new MagickGeometry(8u, 8u)
        {
            IgnoreAspectRatio = true,
        };
        image.Resize(thumbnailGeometry);
        image.Format = MagickFormat.Gray;
        var pixels = image.ToByteArray(MagickFormat.Gray);
        if (pixels.Length < 64)
        {
            throw new InvalidDataException("Dedicated image decoder did not return a valid thumbnail.");
        }

        return BuildFingerprint(fileInfo, width, height, pixels[..64], cancellationToken);
    }

    private static ImageFingerprint CreateWpfFingerprint(string filePath, CancellationToken cancellationToken)
    {
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
        return BuildFingerprint(fileInfo, width, height, pixels, cancellationToken);
    }

    private static ImageFingerprint BuildFingerprint(
        FileInfo fileInfo,
        int width,
        int height,
        byte[] pixels,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

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

        return new ImageFingerprint
        {
            ContentId = contentId,
            AverageHash = hashBuilder.ToString(),
            AverageHashBits = averageHashBits,
            Width = width,
            Height = height,
        };
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
