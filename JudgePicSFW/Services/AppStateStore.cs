using System.IO;
using System.Text.Json;
using JudgePicSFW.Models;

namespace JudgePicSFW.Services;

public sealed class AppStateStore
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    private readonly string _stateFilePath;
    private readonly SemaphoreSlim _syncLock = new(1, 1);

    public AppStateStore()
    {
        var appDataRoot = AppRuntimePaths.DataFolder;

        Directory.CreateDirectory(appDataRoot);
        _stateFilePath = Path.Combine(appDataRoot, "app-state.json");
    }

    public async Task<PersistedAppState> LoadAsync()
    {
        if (!File.Exists(_stateFilePath))
        {
            return new PersistedAppState();
        }

        try
        {
            await using var stream = File.OpenRead(_stateFilePath);
            var state = await JsonSerializer.DeserializeAsync<PersistedAppState>(stream, SerializerOptions);
            return state ?? new PersistedAppState();
        }
        catch (JsonException)
        {
            return new PersistedAppState();
        }
        catch (IOException)
        {
            return new PersistedAppState();
        }
    }

    public async Task SaveAsync(PersistedAppState state)
    {
        await _syncLock.WaitAsync();

        try
        {
            await using var stream = File.Create(_stateFilePath);
            await JsonSerializer.SerializeAsync(stream, state, SerializerOptions);
        }
        finally
        {
            _syncLock.Release();
        }
    }
}
