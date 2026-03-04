import pandas as pd
import numpy as np

COMPLETED = {"dropped-off", "dropped off", "completed", "complete"}
ABANDONED = {"cancelled", "canceled", "abandoned"}

def norm_status(x):
    return "" if pd.isna(x) else str(x).strip().lower()

def compute_kpis_from_excels(riders_path="riders.xlsx", drivers_path="drivers.xlsx"):
    riders = pd.read_excel(riders_path)
    drivers = pd.read_excel(drivers_path)  # loaded for completeness / future KPIs

    # --- required rider columns (per your schema) ---
    required_rider_cols = {
        "id","request_time","pickup_time","dropoff_time",
        "pickup_location","dropoff_location","status"
    }
    missing = required_rider_cols - set(riders.columns)
    if missing:
        raise KeyError(f"riders.xlsx missing columns: {sorted(missing)}")

    riders["_status_norm"] = riders["status"].map(norm_status)

    completed_rides = int(riders["_status_norm"].isin(COMPLETED).sum())
    abandoned_rides = int(riders["_status_norm"].isin(ABANDONED).sum())
    total_requests  = int(len(riders))

    abandonment_rate = (abandoned_rides / total_requests) if total_requests else 0.0

    # Avg wait (minutes): (pickup_time - request_time) * 60, only for completed rides
    avg_wait_min = 0.0
    if completed_rides > 0:
        comp = riders[riders["_status_norm"].isin(COMPLETED)].copy()
        # protect against missing timestamps
        comp = comp.dropna(subset=["request_time", "pickup_time"])
        if len(comp) > 0:
            avg_wait_min = float(((comp["pickup_time"] - comp["request_time"]) * 60).mean())

    # Driver KPIs not computable from your drivers.xlsx schema (no earnings, no busy time)
    avg_earnings_per_hr = 0.0
    avg_utilisation     = 0.0
    earnings_std        = 0.0

    return {
        "completed_rides": completed_rides,
        "abandoned_rides": abandoned_rides,
        "total_requests": total_requests,
        "abandonment_rate": round(abandonment_rate, 4),
        "avg_wait_min": round(avg_wait_min, 2),
        "avg_earnings_per_hr": round(avg_earnings_per_hr, 2),
        "avg_utilisation": round(avg_utilisation, 4),
        "earnings_std": round(earnings_std, 2),
    }

def print_results_like_model(results, seed=42):
    print("=" * 55)
    print(f"REAL DATA RESULTS")
    print("=" * 55)
    print(f"  Completed rides    : {results['completed_rides']:,}")
    print(f"  Abandoned rides    : {results['abandoned_rides']:,}")
    print(f"  Total requests     : {results['total_requests']:,}")
    print(f"  Abandonment rate   : {results['abandonment_rate']*100:.1f}%")
    print(f"  Avg wait time      : {results['avg_wait_min']:.2f} minutes")
    print(f"  Avg earnings/hr    : £{results['avg_earnings_per_hr']:.2f}")
    print(f"  Avg utilisation    : {results['avg_utilisation']*100:.1f}%")
    print(f"  Earnings std/hr    : £{results['earnings_std']:.2f}")

# ---- run ----
results = compute_kpis_from_excels("riders.xlsx", "drivers.xlsx")
print_results_like_model(results, seed=42)