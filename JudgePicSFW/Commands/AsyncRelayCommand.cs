using System.Windows.Input;

namespace JudgePicSFW.Commands;

public sealed class AsyncRelayCommand : ICommand
{
    private readonly Func<Task>? _executeAsync;
    private readonly Func<object?, Task>? _executeWithParameterAsync;
    private readonly Func<bool>? _canExecute;
    private readonly Func<object?, bool>? _canExecuteWithParameter;
    private bool _isRunning;

    public AsyncRelayCommand(Func<Task> executeAsync, Func<bool>? canExecute = null)
    {
        _executeAsync = executeAsync;
        _canExecute = canExecute;
    }

    public AsyncRelayCommand(Func<object?, Task> executeAsync, Func<object?, bool>? canExecute = null)
    {
        _executeWithParameterAsync = executeAsync;
        _canExecuteWithParameter = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter)
    {
        return !_isRunning &&
               (_canExecuteWithParameter?.Invoke(parameter) ?? _canExecute?.Invoke() ?? true);
    }

    public async void Execute(object? parameter)
    {
        if (!CanExecute(parameter))
        {
            return;
        }

        _isRunning = true;
        RaiseCanExecuteChanged();

        try
        {
            if (_executeWithParameterAsync is not null)
            {
                await _executeWithParameterAsync(parameter);
            }
            else
            {
                await _executeAsync!();
            }
        }
        finally
        {
            _isRunning = false;
            RaiseCanExecuteChanged();
        }
    }

    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
