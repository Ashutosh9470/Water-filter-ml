
from pathlib import Path

# ==========================================
# HORIZON FILTER LIFE PREDICTION ENGINE
# ==========================================

RATED_CAPACITY_LITRES = 9463.53  # Provisional: 2500 US gallons


def calculate_filter_life(
    cumulative_litres,
    average_daily_usage
):
    if cumulative_litres < 0:
        raise ValueError("Cumulative litres cannot be negative")

    if average_daily_usage <= 0:
        raise ValueError("Daily usage must be greater than zero")

    remaining_litres = max(
        0,
        RATED_CAPACITY_LITRES - cumulative_litres
    )

    remaining_percentage = (
        remaining_litres / RATED_CAPACITY_LITRES
    ) * 100

    remaining_days = (
        remaining_litres / average_daily_usage
    )

    return {
        "remaining_litres": remaining_litres,
        "remaining_life_pct": remaining_percentage,
        "remaining_days": remaining_days
    }


def main():
    print("\n================================")
    print(" HORIZON FILTER LIFE ENGINE")
    print("================================")

    try:
        cumulative_litres = float(
            input("Enter cumulative water processed (L): ")
        )

        average_daily_usage = float(
            input("Enter average daily water usage (L/day): ")
        )

        result = calculate_filter_life(
            cumulative_litres,
            average_daily_usage
        )

        print("\nFILTER LIFE RESULTS")
        print("--------------------------------")

        print(
            f"Rated capacity: "
            f"{RATED_CAPACITY_LITRES:.2f} L"
        )

        print(
            f"Processed water: "
            f"{cumulative_litres:.2f} L"
        )

        print(
            f"Remaining capacity: "
            f"{result['remaining_litres']:.2f} L"
        )

        print(
            f"Remaining life: "
            f"{result['remaining_life_pct']:.2f}%"
        )

        print(
            f"Estimated remaining days: "
            f"{result['remaining_days']:.2f}"
        )

        print("--------------------------------")

        if result["remaining_life_pct"] <= 0:
            print("STATUS: RATED CAPACITY REACHED")
        elif result["remaining_life_pct"] <= 10:
            print("STATUS: REPLACEMENT DUE SOON")
        elif result["remaining_life_pct"] <= 25:
            print("STATUS: LOW REMAINING CAPACITY")
        else:
            print("STATUS: CAPACITY WITHIN RATED LIMIT")

        print("================================")

    except ValueError as error:
        print("Input error:", error)


if __name__ == "__main__":
    main()
