namespace JudgePicSFW.ViewModels;

public enum OperationLogLevel
{
    Info,
    Success,
    Warning,
    Error,
}

public sealed class OperationLogItemViewModel
{
    public OperationLogItemViewModel(OperationLogLevel level, string message, string detail)
    {
        Level = level;
        Message = message;
        Detail = detail;
        TimestampText = DateTime.Now.ToString("HH:mm:ss");
    }

    public OperationLogLevel Level { get; }

    public string Message { get; }

    public string Detail { get; }

    public string TimestampText { get; }

    public string CopyText => string.IsNullOrWhiteSpace(Detail)
        ? $"[{TimestampText}] {Message}"
        : $"[{TimestampText}] {Message}{Environment.NewLine}{Detail}";
}
