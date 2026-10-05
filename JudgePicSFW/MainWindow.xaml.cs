using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shell;
using System.Windows.Threading;
using JudgePicSFW.ViewModels;

namespace JudgePicSFW;

public partial class MainWindow : Window
{
    public static readonly DependencyProperty IsCompactLayoutProperty = DependencyProperty.Register(
        nameof(IsCompactLayout), typeof(bool), typeof(MainWindow), new PropertyMetadata(false));

    public bool IsCompactLayout
    {
        get => (bool)GetValue(IsCompactLayoutProperty);
        private set => SetValue(IsCompactLayoutProperty, value);
    }

    private readonly MainViewModel _viewModel;

    public MainWindow(MainViewModel viewModel)
    {
        InitializeComponent();
        TextOptions.SetTextFormattingMode(this, TextFormattingMode.Display);
        TextOptions.SetTextRenderingMode(this, TextRenderingMode.ClearType);
        TextOptions.SetTextHintingMode(this, TextHintingMode.Fixed);
        RenderOptions.SetClearTypeHint(this, ClearTypeHint.Enabled);
        UseLayoutRounding = true;
        SnapsToDevicePixels = true;
        Background = (System.Windows.Media.Brush)FindResource("SurfaceBrush");
        StateChanged += OnWindowStateChanged;
        SizeChanged += OnWindowSizeChanged;
        UpdateWindowChromeForState();

        DataContext = viewModel;
        _viewModel = viewModel;

        Loaded += OnLoaded;
        Closed += OnClosed;
        _viewModel.PropertyChanged += OnViewModelPropertyChanged;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        Loaded -= OnLoaded;
        _viewModel.InitializeCommand.Execute(null);
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
        StateChanged -= OnWindowStateChanged;
        SizeChanged -= OnWindowSizeChanged;
    }

    private void OnWindowSizeChanged(object sender, SizeChangedEventArgs e)
    {
        // ActualWidth is measured in DIP, so the breakpoint follows the available space at any DPI.
        var compact = ActualWidth < 1500d;
        if (compact == IsCompactLayout)
        {
            return;
        }

        IsCompactLayout = compact;
        Resources["ResultResolutionColumnWidth"] = new GridLength(compact ? 90d : 104d);
        Resources["ResultConfidenceColumnWidth"] = new GridLength(compact ? 54d : 64d);
        Resources["ResultLabelColumnWidth"] = new GridLength(compact ? 64d : 72d);
        Resources["ResultOriginColumnWidth"] = new GridLength(compact ? 80d : 88d);
    }

    private void OnWindowStateChanged(object? sender, EventArgs e)
    {
        UpdateWindowChromeForState();
    }

    private void UpdateWindowChromeForState()
    {
        var chrome = WindowChrome.GetWindowChrome(this);
        if (chrome is null)
        {
            return;
        }

        chrome.ResizeBorderThickness = WindowState == WindowState.Maximized
            ? new Thickness(0)
            : new Thickness(6);
    }

    private void OnReviewAreaPreviewMouseWheel(object sender, MouseWheelEventArgs e)
    {
        var direction = e.Delta < 0 ? 1 : -1;
        e.Handled = _viewModel.SelectAdjacentVisibleResult(direction);
    }

    private void OnTitleBarMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (e.ClickCount == 2)
        {
            ToggleWindowState();
            return;
        }

        try
        {
            DragMove();
        }
        catch (InvalidOperationException)
        {
        }
    }

    private void OnMinimizeButtonClick(object sender, RoutedEventArgs e)
    {
        WindowState = WindowState.Minimized;
    }

    private void OnMaximizeRestoreButtonClick(object sender, RoutedEventArgs e)
    {
        ToggleWindowState();
    }

    private void OnCloseButtonClick(object sender, RoutedEventArgs e)
    {
        Close();
    }

    private void ToggleWindowState()
    {
        WindowState = WindowState == WindowState.Maximized
            ? WindowState.Normal
            : WindowState.Maximized;
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName != nameof(MainViewModel.SelectedResult))
        {
            return;
        }

        Dispatcher.BeginInvoke(() =>
        {
            if (_viewModel.SelectedResult is not null)
            {
                CenterSelectedResultInList();
            }
        }, DispatcherPriority.Background);
    }

    private void CenterSelectedResultInList(int attempt = 0)
    {
        var selectedResult = _viewModel.SelectedResult;
        if (selectedResult is null || !IsLoaded)
        {
            return;
        }

        var itemContainer = ResultsListBox.ItemContainerGenerator.ContainerFromItem(selectedResult) as ListBoxItem;
        if (itemContainer is null)
        {
            ResultsListBox.ScrollIntoView(selectedResult);
            ScheduleCenteredSelection(attempt);
            return;
        }

        var scrollViewer = FindVisualChild<ScrollViewer>(ResultsListBox);
        if (scrollViewer is null || scrollViewer.ViewportHeight <= 0d)
        {
            ScheduleCenteredSelection(attempt);
            return;
        }

        var itemIndex = ResultsListBox.ItemContainerGenerator.IndexFromContainer(itemContainer);
        if (itemIndex < 0)
        {
            return;
        }

        var targetOffset = itemIndex - Math.Max(0d, (scrollViewer.ViewportHeight - 1d) / 2d);
        scrollViewer.ScrollToVerticalOffset(Math.Clamp(targetOffset, 0d, scrollViewer.ScrollableHeight));
    }

    private void ScheduleCenteredSelection(int attempt)
    {
        if (attempt >= 4)
        {
            return;
        }

        Dispatcher.BeginInvoke(
            DispatcherPriority.ContextIdle,
            new Action(() => CenterSelectedResultInList(attempt + 1)));
    }

    private static T? FindVisualChild<T>(DependencyObject parent)
        where T : DependencyObject
    {
        for (var index = 0; index < VisualTreeHelper.GetChildrenCount(parent); index++)
        {
            var child = VisualTreeHelper.GetChild(parent, index);
            if (child is T match)
            {
                return match;
            }

            var descendant = FindVisualChild<T>(child);
            if (descendant is not null)
            {
                return descendant;
            }
        }

        return null;
    }

    private void OnWindowPreviewKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (_viewModel.SelectedResult is null || _viewModel.IsBusy)
        {
            return;
        }

        switch (e.Key)
        {
            case Key.S:
                if (_viewModel.MarkAsSfwCommand.CanExecute(null))
                {
                    _viewModel.MarkAsSfwCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.N:
                if (_viewModel.MarkAsNsfwCommand.CanExecute(null))
                {
                    _viewModel.MarkAsNsfwCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.U:
                if (_viewModel.MarkAsUncertainCommand.CanExecute(null))
                {
                    _viewModel.MarkAsUncertainCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.Left:
            case Key.Up:
                e.Handled = _viewModel.SelectAdjacentVisibleResult(-1);
                break;
            case Key.Right:
            case Key.Down:
                e.Handled = _viewModel.SelectAdjacentVisibleResult(1);
                break;
            case Key.Enter:
                if (_viewModel.ConfirmMoveCommand.CanExecute(null))
                {
                    _viewModel.ConfirmMoveCommand.Execute(null);
                    e.Handled = true;
                }
                break;
        }
    }
}
