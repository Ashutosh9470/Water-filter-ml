
RATED_CAPACITY_LITRES = 9463.53  # Provisional: 2500 US gallons


def calculate_filter_life(
    cumulative_litres: float,
    average_daily_usage: float
):
    if cumulative_litres < 0:
        raise ValueError("Cumulative litres cannot be negative")

    if average_daily_usage <= 0:
        raise ValueError("Daily usage must be positive")

    remaining_litres = max(
        0.0,
        RATED_CAPACITY_LITRES - cumulative_litres
    )

    remaining_percentage = (
        remaining_litres / RATED_CAPACITY_LITRES
    ) * 100

    remaining_days = (
        remaining_litres / average_daily_usage
    )

    return {
        "rated_capacity_litres": RATED_CAPACITY_LITRES,
        "cumulative_litres": cumulative_litres,
        "remaining_litres": round(remaining_litres, 2),
        "remaining_life_pct": round(remaining_percentage, 2),
        "remaining_days": round(remaining_days, 2)
    }
