using System.Drawing;
using Forms = System.Windows.Forms;

namespace JudgePicSFW.Services;

public sealed class TrayIconService : IDisposable
{
    private readonly Forms.ContextMenuStrip _contextMenu;
    private readonly Forms.NotifyIcon _notifyIcon;
    private readonly Icon _trayIcon;
    private bool _disposed;

    public TrayIconService(string tooltipText, Action showWindow, Action hideWindow, Action exitApplication)
    {
        _contextMenu = new Forms.ContextMenuStrip();
        _contextMenu.Items.Add("\u663E\u793A\u8F6F\u4EF6", null, (_, _) => showWindow());
        _contextMenu.Items.Add("\u9000\u51FA", null, (_, _) => exitApplication());

        _trayIcon = CreateTrayIcon();

        _notifyIcon = new Forms.NotifyIcon
        {
            Text = tooltipText,
            Icon = _trayIcon,
            Visible = true,
            ContextMenuStrip = _contextMenu,
        };

        _notifyIcon.MouseClick += (_, eventArgs) =>
        {
            if (eventArgs.Button == Forms.MouseButtons.Left)
            {
                var mainWindow = System.Windows.Application.Current.MainWindow;
                if (mainWindow?.IsVisible == true)
                {
                    hideWindow();
                    return;
                }

                showWindow();
            }
        };
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _notifyIcon.Visible = false;
        _notifyIcon.Dispose();
        _trayIcon.Dispose();
        _contextMenu.Dispose();
    }

    private static Icon CreateTrayIcon()
    {
        var streamInfo = System.Windows.Application.GetResourceStream(
            new Uri("pack://application:,,,/Assets/JudgePicSFW.ico", UriKind.Absolute));
        if (streamInfo?.Stream is not null)
        {
            using var stream = streamInfo.Stream;
            using var icon = new Icon(stream);
            return (Icon)icon.Clone();
        }

        return SystemIcons.Application;
    }
}
