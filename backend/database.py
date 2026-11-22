import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "horizon_sensor_data.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pulse_count INTEGER NOT NULL,
                interval_seconds REAL NOT NULL,
                litres_processed REAL NOT NULL,
                flow_rate REAL NOT NULL,
                temperature REAL NOT NULL,
                outlet_turbidity REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_timestamp
            ON sensor_readings(device_id, timestamp)
        """)


def save_reading(
    device_id,
    timestamp,
    pulse_count,
    interval_seconds,
    litres_processed,
    flow_rate,
    temperature,
    outlet_turbidity
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sensor_readings (
                device_id,
                timestamp,
                pulse_count,
                interval_seconds,
                litres_processed,
                flow_rate,
                temperature,
                outlet_turbidity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_id,
            timestamp,
            pulse_count,
            interval_seconds,
            litres_processed,
            flow_rate,
            temperature,
            outlet_turbidity
        ))


def get_device_summary(device_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COALESCE(SUM(litres_processed), 0) AS cumulative_litres,
                COALESCE(AVG(flow_rate), 0) AS average_flow_rate,
                COALESCE(AVG(temperature), 0) AS average_temperature,
                COALESCE(AVG(outlet_turbidity), 0) AS average_turbidity,
                MIN(timestamp) AS first_reading,
                MAX(timestamp) AS last_reading
            FROM sensor_readings
            WHERE device_id = ?
        """, (device_id,)).fetchone()

        return dict(row)


def get_average_daily_usage(device_id, lookback_days=7, minimum_observed_days=2):
    """Return average litres per observed calendar day over the recent window.

    Returns None until readings exist on at least minimum_observed_days distinct
    dates. Zero-flow readings count as observed days, which is useful when the
    device is online but the pump is idle.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                substr(timestamp, 1, 10) AS usage_date,
                SUM(litres_processed) AS daily_litres
            FROM sensor_readings
            WHERE device_id = ? AND timestamp >= ?
            GROUP BY substr(timestamp, 1, 10)
            ORDER BY usage_date
        """, (device_id, cutoff)).fetchall()

    observed_days = len(rows)
    if observed_days < minimum_observed_days:
        return {
            "average_daily_usage": None,
            "observed_days": observed_days,
            "lookback_days": lookback_days
        }

    total_litres = sum(float(row["daily_litres"] or 0.0) for row in rows)
    average_daily_usage = total_litres / observed_days

    # A zero average means no measured consumption; a finite days estimate
    # cannot be calculated until water usage is observed.
    return {
        "average_daily_usage": round(average_daily_usage, 4),
        "observed_days": observed_days,
        "lookback_days": lookback_days
    }


def get_recent_readings(device_id, limit=100):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT *
            FROM sensor_readings
            WHERE device_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (device_id, limit)).fetchall()

        return [dict(row) for row in rows]
