from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.inference import calculate_filter_life
from backend.database import (
    initialize_database,
    save_reading,
    get_device_summary,
    get_average_daily_usage,
    get_recent_readings
)

app = FastAPI(
    title="HORIZON API",
    description="Smart Water Purifier Monitoring System",
    version="2.1.0"
)

# Provisional YF-S201 calibration. Calibrate the physical sensor before
# relying on these values.
PULSES_PER_LITRE = 450.0
USAGE_LOOKBACK_DAYS = 7
MINIMUM_USAGE_DAYS = 2


@app.on_event("startup")
def startup():
    initialize_database()


class SensorData(BaseModel):
    device_id: str = Field(min_length=1, max_length=64)
    pulse_count: int = Field(ge=0)
    interval_seconds: float = Field(gt=0, le=3600)
    temperature: float = Field(ge=0, le=100)
    outlet_turbidity: float = Field(ge=0)


class PredictionData(BaseModel):
    device_id: str
    cumulative_litres: float = Field(ge=0)
    average_daily_usage: float = Field(gt=0)
    flow_rate: float = Field(ge=0)
    temperature: float = Field(ge=0, le=100)
    outlet_turbidity: float = Field(ge=0)


def build_filter_estimate(device_id, cumulative_litres):
    """Build capacity and days estimates from measured recent usage."""
    usage = get_average_daily_usage(
        device_id,
        lookback_days=USAGE_LOOKBACK_DAYS,
        minimum_observed_days=MINIMUM_USAGE_DAYS
    )

    average_daily_usage = usage["average_daily_usage"]
    if average_daily_usage is not None and average_daily_usage > 0:
        life = calculate_filter_life(cumulative_litres, average_daily_usage)
        remaining_days = life["remaining_days"]
    else:
        # Not enough history or no measured water consumption yet.
        life = calculate_filter_life(cumulative_litres, 1.0)
        remaining_days = None

    return {
        "cumulative_litres": round(cumulative_litres, 3),
        "remaining_litres": life["remaining_litres"],
        "remaining_life_pct": life["remaining_life_pct"],
        "remaining_days": remaining_days,
        "average_daily_usage_litres": average_daily_usage,
        "usage_days_observed": usage["observed_days"],
        "usage_lookback_days": usage["lookback_days"]
    }


@app.get("/")
def home():
    return {
        "project": "HORIZON",
        "status": "online",
        "message": "Backend is running"
    }


@app.post("/sensor-data")
def receive_sensor_data(data: SensorData):
    litres_processed = data.pulse_count / PULSES_PER_LITRE
    flow_rate = (litres_processed / data.interval_seconds) * 60
    timestamp = datetime.now(timezone.utc).isoformat()

    save_reading(
        device_id=data.device_id,
        timestamp=timestamp,
        pulse_count=data.pulse_count,
        interval_seconds=data.interval_seconds,
        litres_processed=litres_processed,
        flow_rate=flow_rate,
        temperature=data.temperature,
        outlet_turbidity=data.outlet_turbidity
    )

    summary = get_device_summary(data.device_id)
    filter_estimate = build_filter_estimate(
        data.device_id,
        summary["cumulative_litres"]
    )

    return {
        "device_id": data.device_id,
        "status": "success",
        "timestamp": timestamp,
        "current_reading": {
            "pulse_count": data.pulse_count,
            "litres_processed": round(litres_processed, 4),
            "flow_rate": round(flow_rate, 3),
            "temperature": data.temperature,
            "outlet_turbidity": data.outlet_turbidity
        },
        "filter": filter_estimate,
        "condition_monitoring": {
            "mode": "MONITORING_ONLY",
            "message": "No validated condition classification available."
        }
    }


@app.get("/device-summary/{device_id}")
def device_summary(device_id: str):
    summary = get_device_summary(device_id)

    if summary["first_reading"] is None:
        raise HTTPException(
            status_code=404,
            detail="No readings found for this device"
        )

    filter_estimate = build_filter_estimate(
        device_id,
        summary["cumulative_litres"]
    )

    return {
        "device_id": device_id,
        "total_water_processed_litres": round(summary["cumulative_litres"], 3),
        "remaining_capacity_litres": filter_estimate["remaining_litres"],
        "remaining_life_pct": filter_estimate["remaining_life_pct"],
        "remaining_days": filter_estimate["remaining_days"],
        "average_daily_usage_litres": filter_estimate["average_daily_usage_litres"],
        "usage_days_observed": filter_estimate["usage_days_observed"],
        "average_flow_rate": round(summary["average_flow_rate"], 3),
        "average_temperature": round(summary["average_temperature"], 2),
        "average_outlet_turbidity": round(summary["average_turbidity"], 3),
        "first_reading": summary["first_reading"],
        "last_reading": summary["last_reading"]
    }


@app.get("/readings/{device_id}")
def readings(device_id: str, limit: int = 100):
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000"
        )

    return {
        "device_id": device_id,
        "readings": get_recent_readings(device_id, limit)
    }


@app.post("/predict")
def predict(data: PredictionData):
    result = calculate_filter_life(
        data.cumulative_litres,
        data.average_daily_usage
    )

    return {
        "device_id": data.device_id,
        "status": "success",
        "filter": result,
        "sensors": {
            "flow_rate": data.flow_rate,
            "temperature": data.temperature,
            "outlet_turbidity": data.outlet_turbidity
        }
    }
