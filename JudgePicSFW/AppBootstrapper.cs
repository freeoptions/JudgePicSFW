using System.ComponentModel;
using System.Windows;
using System.Windows.Threading;
using JudgePicSFW.Services;
using JudgePicSFW.ViewModels;

namespace JudgePicSFW;

public sealed class AppBootstrapper : IDisposable
{
    private const double ScreenPadding = 24d;

    private readonly AppStateStore _appStateStore;
    private readonly ImageDecodeCacheService _imageDecodeCacheService;
    private readonly ImageFingerprintService _imageFingerprintService;
    private readonly AiNsfwClassifierService _aiNsfwClassifierService;
    private readonly PersonalAiTrainingService _personalAiTrainingService;
    private readonly WindowsShellFileMoveService _fileMoveService;
    private readonly PerformanceLogService _performanceLogService;
    private readonly WorkspaceService _workspaceService;

    private MainWindow? _mainWindow;
    private MainViewModel? _mainViewModel;
    private TrayIconService? _trayIconService;
    private bool _isExitRequested;
    private bool _isDisposed;

    public AppBootstrapper()
    {
        _appStateStore = new AppStateStore();
        _imageDecodeCacheService = new ImageDecodeCacheService();
        _imageFingerprintService = new ImageFingerprintService();
        _performanceLogService = new PerformanceLogService();
        _aiNsfwClassifierService = new AiNsfwClassifierService(_performanceLogService, _imageDecodeCacheService);
        _personalAiTrainingService = new PersonalAiTrainingService(_performanceLogService, _imageDecodeCacheService);
        _fileMoveService = new WindowsShellFileMoveService();
        _workspaceService = new WorkspaceService(_appStateStore, _imageFingerprintService, _aiNsfwClassifierService, _personalAiTrainingService, _fileMoveService, _performanceLogService);
    }

    public void Run()
    {
        _mainViewModel = new MainViewModel(_workspaceService, _imageDecodeCacheService);
        _mainWindow = new MainWindow(_mainViewModel);
        _mainWindow.WindowState = WindowState.Maximized;
        _mainWindow.Closing += OnMainWindowClosing;
        System.Windows.Application.Current.ShutdownMode = ShutdownMode.OnExplicitShutdown;
        System.Windows.Application.Current.MainWindow = _mainWindow;

        ShowMainWindow();
        _mainWindow.Dispatcher.BeginInvoke(() =>
        {
            if (_mainWindow is not null)
            {
                ShowMainWindow();
            }
        }, DispatcherPriority.ApplicationIdle);

        _trayIconService = new TrayIconService(
            "JudgePicSFW - 图片安全判断",
            ShowMainWindow,
            HideMainWindow,
            ExitApplication);
    }

    public void Dispose()
    {
        if (_isDisposed)
        {
            return;
        }

        _isDisposed = true;
        if (_mainWindow is not null)
        {
            _mainWindow.Closing -= OnMainWindowClosing;
        }

        _trayIconService?.Dispose();
        _aiNsfwClassifierService.Dispose();
        _mainViewModel?.Dispose();
        _imageDecodeCacheService.Dispose();
        _trayIconService = null;
        _mainViewModel = null;
    }

    private void OnMainWindowClosing(object? sender, CancelEventArgs e)
    {
        if (_isExitRequested)
        {
            return;
        }

        e.Cancel = true;
        HideMainWindow();
    }

    private void ShowMainWindow()
    {
        if (_mainWindow is null)
        {
            return;
        }

        _mainWindow.ShowInTaskbar = true;
        if (!_mainWindow.IsVisible)
        {
            _mainWindow.Show();
        }

        _mainWindow.WindowState = WindowState.Maximized;

        EnsureWindowVisible(_mainWindow);
        _mainWindow.Activate();
        _mainWindow.Topmost = true;
        _mainWindow.Topmost = false;
        _mainWindow.Focus();
    }

    private static void EnsureWindowVisible(Window window)
    {
        if (window.WindowState != WindowState.Normal)
        {
            return;
        }

        var workArea = SystemParameters.WorkArea;
        var maxWidth = Math.Max(640d, workArea.Width - (ScreenPadding * 2d));
        var maxHeight = Math.Max(480d, workArea.Height - (ScreenPadding * 2d));

        window.MinWidth = Math.Min(window.MinWidth, maxWidth);
        window.MinHeight = Math.Min(window.MinHeight, maxHeight);
        window.Width = Math.Min(window.Width, maxWidth);
        window.Height = Math.Min(window.Height, maxHeight);

        var isOutside = double.IsNaN(window.Left) ||
                        double.IsNaN(window.Top) ||
                        window.Left + 120d < workArea.Left ||
                        window.Top + 80d < workArea.Top ||
                        window.Left > workArea.Right - 120d ||
                        window.Top > workArea.Bottom - 80d;

        if (isOutside)
        {
            window.Left = workArea.Left + Math.Max(ScreenPadding, (workArea.Width - window.Width) / 2d);
            window.Top = workArea.Top + Math.Max(ScreenPadding, (workArea.Height - window.Height) / 2d);
        }
    }

    private void HideMainWindow()
    {
        if (_mainWindow is null)
        {
            return;
        }

        _mainWindow.Hide();
        _mainWindow.ShowInTaskbar = false;
    }

    private void ExitApplication()
    {
        _isExitRequested = true;
        Dispose();

        if (_mainWindow is not null)
        {
            _mainWindow.Closing -= OnMainWindowClosing;
            _mainWindow.Close();
        }

        System.Windows.Application.Current.Shutdown();
    }
}
