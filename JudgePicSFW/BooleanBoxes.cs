using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace JudgePicSFW;

public static class BooleanBoxes
{
    public static IValueConverter NullToVisible { get; } = new NullToVisibleConverter();
    public static IValueConverter NotNullToVisible { get; } = new NotNullToVisibleConverter();
    public static IValueConverter BoolToVisible { get; } = new BoolToVisibleConverter();
    public static IValueConverter ProgressToRingDashArray { get; } = new ProgressToRingDashArrayConverter();
    public static IValueConverter ConfidenceToNumber { get; } = new ConfidenceToNumberConverter();

    private sealed class ConfidenceToNumberConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value is double confidence && double.IsFinite(confidence) && confidence > 0d
                ? confidence.ToString("0.00", CultureInfo.InvariantCulture)
                : "--";
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    private sealed class NullToVisibleConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value is null ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    private sealed class NotNullToVisibleConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value is null ? Visibility.Collapsed : Visibility.Visible;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    private sealed class BoolToVisibleConverter : IValueConverter
    {
        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            return value is true ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }

    private sealed class ProgressToRingDashArrayConverter : IValueConverter
    {
        private const double RingLength = 30d;

        public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            var progress = value switch
            {
                double doubleValue => doubleValue,
                float floatValue => floatValue,
                decimal decimalValue => (double)decimalValue,
                int intValue => intValue,
                _ => 0d,
            };

            progress = Math.Clamp(progress, 0d, 1d);
            var ringLength = parameter is string text
                && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsedLength)
                && double.IsFinite(parsedLength) && parsedLength > 0d
                    ? parsedLength
                    : RingLength;
            return new DoubleCollection { progress * ringLength, ringLength };
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }
}
