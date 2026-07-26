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
            return new DoubleCollection { progress * RingLength, RingLength };
        }

        public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            throw new NotSupportedException();
        }
    }
}
