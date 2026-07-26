using System.IO;

namespace JudgePicSFW.Services;

internal static class AppRuntimePaths
{
    public static string ApplicationFolder => ResolveApplicationFolder(Environment.ProcessPath, AppContext.BaseDirectory);

    public static string DataFolder => Path.Combine(ApplicationFolder, "Data");

    public static string ResolveDataPath(params string[] segments)
    {
        return segments.Length == 0
            ? DataFolder
            : Path.Combine([DataFolder, .. segments]);
    }

    internal static string ResolveApplicationFolder(string? processPath, string? baseDirectory)
    {
        if (!string.IsNullOrWhiteSpace(processPath))
        {
            try
            {
                var processFolder = Path.GetDirectoryName(Path.GetFullPath(processPath));
                if (!string.IsNullOrWhiteSpace(processFolder))
                {
                    return TrimDirectoryEnd(processFolder);
                }
            }
            catch
            {
            }
        }

        if (!string.IsNullOrWhiteSpace(baseDirectory))
        {
            return TrimDirectoryEnd(Path.GetFullPath(baseDirectory));
        }

        return TrimDirectoryEnd(Directory.GetCurrentDirectory());
    }

    private static string TrimDirectoryEnd(string path)
    {
        var root = Path.GetPathRoot(path);
        var trimmed = path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return string.IsNullOrWhiteSpace(trimmed) && !string.IsNullOrWhiteSpace(root)
            ? root
            : trimmed;
    }
}
