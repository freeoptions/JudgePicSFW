using System.IO;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class WindowsShellFileMoveService
{
    public Task<FileMoveResult> MoveFileAsync(string sourcePath, string destinationFolderPath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Directory.CreateDirectory(destinationFolderPath);

        return Task.Run(() => MoveWithAutoRename(sourcePath, destinationFolderPath, cancellationToken), cancellationToken);
    }

    private static FileMoveResult MoveWithAutoRename(string sourcePath, string destinationFolderPath, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var originalDestinationPath = Path.Combine(destinationFolderPath, Path.GetFileName(sourcePath));
        if (!File.Exists(sourcePath))
        {
            return new FileMoveResult
            {
                SourcePath = sourcePath,
                DestinationPath = originalDestinationPath,
                Status = FileMoveStatus.SourceMissing,
                Message = "Skipped because the source file no longer exists.",
            };
        }

        var finalDestinationPath = originalDestinationPath;
        var wasRenamed = false;
        if (File.Exists(originalDestinationPath))
        {
            finalDestinationPath = BuildRenamedDestinationPath(destinationFolderPath, sourcePath);
            wasRenamed = true;
        }

        try
        {
            File.Move(sourcePath, finalDestinationPath, overwrite: false);
        }
        catch (IOException moveException) when (!File.Exists(finalDestinationPath))
        {
            CopyThenDeleteSource(sourcePath, finalDestinationPath, moveException, cancellationToken);
        }

        if (File.Exists(sourcePath) || !File.Exists(finalDestinationPath))
        {
            throw new IOException($"Unexpected move result for file: {sourcePath}");
        }

        return new FileMoveResult
        {
            SourcePath = sourcePath,
            DestinationPath = finalDestinationPath,
            Status = wasRenamed ? FileMoveStatus.RenamedDueToConflict : FileMoveStatus.Moved,
            Message = wasRenamed
                ? $"Renamed due to conflict: {Path.GetFileName(finalDestinationPath)}"
                : "Moved to target folder.",
        };
    }

    private static void CopyThenDeleteSource(string sourcePath, string destinationPath, Exception moveException, CancellationToken cancellationToken)
    {
        var copiedTarget = false;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            File.Copy(sourcePath, destinationPath, overwrite: false);
            copiedTarget = true;

            cancellationToken.ThrowIfCancellationRequested();
            File.Delete(sourcePath);
        }
        catch (Exception copyException)
        {
            if (copiedTarget && File.Exists(destinationPath))
            {
                try
                {
                    File.Delete(destinationPath);
                }
                catch
                {
                }
            }

            throw new IOException($"Fast move fallback failed: {moveException.Message}", copyException);
        }
    }

    private static string BuildRenamedDestinationPath(string destinationFolderPath, string sourcePath)
    {
        var originalName = Path.GetFileNameWithoutExtension(sourcePath);
        var extension = Path.GetExtension(sourcePath);

        while (true)
        {
            var candidateName = $"{originalName}_rename_{Random.Shared.Next(1000)}{extension}";
            var candidatePath = Path.Combine(destinationFolderPath, candidateName);
            if (!File.Exists(candidatePath))
            {
                return candidatePath;
            }
        }
    }
}
