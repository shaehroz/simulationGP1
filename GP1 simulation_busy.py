import numpy as np
import pandas as pd
import heapq
import ast
from dataclasses import dataclass
from scipy.stats import chisquare, kstest, expon, uniform, chi2 as chi2_dist
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import t as t_dist


# =========================
# LOAD DATA
# =========================
riders = pd.read_excel("riders.xlsx")
drivers = pd.read_excel("drivers.xlsx")

print(riders.head())
print(drivers.head())
print(riders.dtypes)
print(drivers.dtypes)


def convert_to_xy(v):
    if isinstance(v, tuple):
        return v
    return ast.literal_eval(v)


drivers["initial_location"] = drivers["initial_location"].apply(convert_to_xy)
drivers["current_location"] = drivers["current_location"].apply(convert_to_xy)
drivers[["x", "y"]] = pd.DataFrame(drivers["current_location"].tolist(), index=drivers.index)
drivers[["initial_x", "initial_y"]] = pd.DataFrame(drivers["initial_location"].tolist(), index=drivers.index)

riders["pickup_location"] = riders["pickup_location"].apply(convert_to_xy)
riders["dropoff_location"] = riders["dropoff_location"].apply(convert_to_xy)
riders[["pickup_x", "pickup_y"]] = pd.DataFrame(riders["pickup_location"].tolist(), index=riders.index)
riders[["dropoff_x", "dropoff_y"]] = pd.DataFrame(riders["dropoff_location"].tolist(), index=riders.index)

ALPHA = 0.05 #significance level
SEED = 42 #fixes randomness


# =========================
# ASSUMPTION TESTING
# =========================
rider_iat = (
    riders["request_time"]
    .sort_values()
    .diff()
    .dropna()
    .values
)

driver_iat = (
    drivers["arrival_time"]
    .sort_values()
    .diff()
    .dropna()
    .values
)

driver_dur = (drivers["offline_time"] - drivers["arrival_time"]).values

print(f"Rider IAT mean: {rider_iat.mean()*60:.2f} minutes  (brief implies 2.00 min)")
print(f"Driver IAT mean: {driver_iat.mean()*60:.2f} minutes  (brief implies 20.00 min)")


def split(arr, frac=0.70, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(arr))
    cut = int(len(arr) * frac)
    return arr[idx[:cut]], arr[idx[cut:]]


rider_iat_train, rider_iat_test = split(rider_iat)
driver_iat_train, driver_iat_test = split(driver_iat)
driver_dur_train, driver_dur_test = split(driver_dur)

rider_rate_mle = 1.0 / rider_iat_train.mean()
driver_rate_mle = 1.0 / driver_iat_train.mean()
dur_a_mle = driver_dur_train.min()
dur_b_mle = driver_dur_train.max()

print("=" * 70)
print("MLE PARAMETER ESTIMATES (from 70% training data)")
print("=" * 70)
print(f"  Rider IAT  – MLE rate : {rider_rate_mle:.2f}/hr  (brief: 30.00/hr)  mean={60/rider_rate_mle:.2f} min")
print(f"  Driver IAT – MLE rate : {driver_rate_mle:.2f}/hr  (brief: 3.00/hr)   mean={60/driver_rate_mle:.2f} min")
print(f"  Driver dur – MLE range: [{dur_a_mle:.2f}, {dur_b_mle:.2f}] hrs  (brief: [5, 8])")
print()


def run_chi2(data, dist, params, n_bins=20):
    n = len(data)

    if dist == "exponential":
        rate = params["rate"]
        lo, hi = 0, np.percentile(data, 99)
        edges = np.linspace(lo, hi, n_bins + 1)
        cdf = expon.cdf(edges, scale=1 / rate)

    elif dist == "uniform":
        a, b = params["a"], params["b"]
        edges = np.linspace(a, b, n_bins + 1)
        cdf = uniform.cdf(edges, loc=a, scale=b - a)

    observed = np.histogram(data, bins=edges)[0].astype(float)
    expected = np.diff(cdf) * n
    expected = expected * (observed.sum() / expected.sum())

    obs_m, exp_m = list(observed), list(expected)
    i = 0
    while i < len(exp_m):
        if exp_m[i] < 5:
            j = i + 1 if i < len(exp_m) - 1 else i - 1
            obs_m[j] += obs_m.pop(i)
            exp_m[j] += exp_m.pop(i)
            i = max(0, i - 1)
        else:
            i += 1

    stat, _ = chisquare(f_obs=obs_m, f_exp=exp_m)

    n_params = 1 if dist == "exponential" else 2
    df = len(obs_m) - 1 - n_params
    df = max(df, 1)
    p = 1 - chi2_dist.cdf(stat, df)

    return round(stat, 3), round(p, 4)


def run_all_tests(label, data_test, dist, brief_params, mle_params):
    dec = lambda p: "REJECT H0" if p < ALPHA else "Fail to reject H0"

    chi2_s_b, chi2_p_b = run_chi2(data_test, dist, brief_params)
    chi2_s_m, chi2_p_m = run_chi2(data_test, dist, mle_params)

    if dist == "exponential":
        ks_b = kstest(data_test, "expon", args=(0, 1 / brief_params["rate"]))
        ks_m = kstest(data_test, "expon", args=(0, 1 / mle_params["rate"]))
    else:
        ks_b = kstest(data_test, "uniform", args=(brief_params["a"], brief_params["b"] - brief_params["a"]))
        ks_m = kstest(data_test, "uniform", args=(mle_params["a"], mle_params["b"] - mle_params["a"]))

    print(f"\n{'─'*65}")
    print(f"  {label}   (n_test = {len(data_test):,})")
    print(f"{'─'*65}")
    print(f"  Testing against BRIEF parameters:")
    print(f"    Chi-squared : stat={chi2_s_b},  p={chi2_p_b}  → {dec(chi2_p_b)}")
    print(f"    KS test     : stat={round(ks_b.statistic,4)},  p={round(ks_b.pvalue,4)}  → {dec(ks_b.pvalue)}")
    print(f"  Testing against MLE parameters:")
    print(f"    Chi-squared : stat={chi2_s_m},  p={chi2_p_m}  → {dec(chi2_p_m)}")
    print(f"    KS test     : stat={round(ks_m.statistic,4)},  p={round(ks_m.pvalue,4)}  → {dec(ks_m.pvalue)}")


print("\n" + "=" * 65)
print("GOODNESS-OF-FIT TEST RESULTS  (alpha = 0.05)")
print("=" * 65)

run_all_tests(
    label="Rider Inter-Arrival Time",
    data_test=rider_iat_test,
    dist="exponential",
    brief_params={"rate": 30.0},
    mle_params={"rate": rider_rate_mle}
)

run_all_tests(
    label="Driver Inter-Arrival Time",
    data_test=driver_iat_test,
    dist="exponential",
    brief_params={"rate": 3.0},
    mle_params={"rate": driver_rate_mle}
)

run_all_tests(
    label="Driver Online Duration",
    data_test=driver_dur_test,
    dist="uniform",
    brief_params={"a": 5.0, "b": 8.0},
    mle_params={"a": dur_a_mle, "b": dur_b_mle}
)


# =========================
# DATACLASSES
# =========================
@dataclass
class Driver:
    id: int
    x: float
    y: float
    pending_dropoff_x: float | None = None
    pending_dropoff_y: float | None = None
    pending_dropoff_time: float | None = None
    pending_pickup: bool = False
    online: bool = False
    busy: bool = False
    status: str = "idle"            # idle / deadhead / in_ride

    wants_offline: bool = False        # flag only (True => offline after current ride/pickup attempt)
    offline_time: float | None = None  # scheduled offline time (float)

    busy_since: float | None = None    # when matched (for busy time)
    arrival_time: float | None = None  # when came online (for online time)


@dataclass
class Rider:
    id: int
    request_time: float
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float

    pickup_dist: float = 0.0           # deadhead (driver->pickup) distance stored at match time

    status: str = "waiting"            # waiting / matched / cancelled / in_ride / completed
    cancel_time: float | None = None
    driver_id: int | None = None


@dataclass(order=True)
class Event:
    time: float
    kind: str
    rider_id: int | None = None
    driver_id: int | None = None


# =========================
# SIMULATION
# =========================
def run_simulation(
    T_end: float = 200.0,
    seed: int = 1,
    square_min: float = 0.0,
    square_max: float = 20.0,
    lam_driver_on: float = 4.65,
    lam_rider_arrival: float = 34.92,
    lam_patience: float = 5.0,
    speed_mph: float = 20.0,
    fare_base: float = 3.0,
    fare_per_mile: float = 2.0,
    cost_per_mile: float = 0.20,
    burn_in: float = 5.0,
) -> dict:
    rng = np.random.default_rng(seed)

    def sample_driver_interarrival_hr():
        return float(rng.exponential(scale=1 / lam_driver_on))

    def sample_driver_online_duration_hr():
        return float(rng.uniform(6.0, 8.0))

    def sample_rider_interarrival_hr():
        return float(rng.exponential(scale=1 / lam_rider_arrival))

    def sample_patience_hr():
        return float(rng.exponential(scale=1 / lam_patience))

    def sample_location():
        return (float(rng.uniform(square_min, square_max)),
                float(rng.uniform(square_min, square_max)))

    def euclid(x1, y1, x2, y2):
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def mean_drive_time_hr(dist_miles):
        return dist_miles / speed_mph

    def actual_drive_time_hr(mu):
        return float(rng.uniform(0.8 * mu, 1.2 * mu))

    driver_objects = {}
    rider_objects = {}
    idle_drivers = []
    busy_drivers = []
    waiting_riders = []

    next_driver_id = 0
    next_rider_id = 0

    heap = []

    def push(ev: Event):
        heapq.heappush(heap, ev)

    def pop():
        return heapq.heappop(heap)

    push(Event(time=sample_driver_interarrival_hr(), kind="driver_arrives"))
    push(Event(time=sample_rider_interarrival_hr(), kind="rider_request"))

    completed_waits = []
    abandoned_count = 0
    completed_count = 0

    driver_earnings = {}
    driver_busy_time = {}
    driver_online_time = {}

    def after_burnin(t: float) -> bool:
        return t >= burn_in

    def record_online_time(driver_id: int, t: float):
        d = driver_objects[driver_id]
        start = max(d.arrival_time, burn_in)
        if t > start:
            driver_online_time[driver_id] = driver_online_time.get(driver_id, 0.0) + (t - start)

    def add_busy_time(driver_id: int, start_t: float, end_t: float):
        """
        Adds busy time but only counts the portion after burn-in.
        """
        if end_t <= burn_in:
            return
        s = max(start_t, burn_in)
        if end_t > s:
            driver_busy_time[driver_id] = driver_busy_time.get(driver_id, 0.0) + (end_t - s)
    
    ### with our improvement
    # if we allow riders to match with a driver that is not yet finished a drive but will be finished within a chosen amount of time, we can
    # increase rider satisfaction as more riders will be matched before their patience runs out
    def find_closest_driver_improved(rx, ry, max_minutes_to_dropoff=10.0):
        # Prefer idle drivers if any exist
        if idle_drivers:
            best_id, best_dist = None, float("inf")
            for did in idle_drivers:
                d = driver_objects[did]
                dist = euclid(d.x, d.y, rx, ry)
                if dist < best_dist:
                    best_dist, best_id = dist, did
            return best_id

        # Otherwise consider drivers currently in a ride who will finish soon,
        # are not going offline, and don't already have a next ride assigned.
        best_id, best_dist = None, float("inf")
        max_hr = max_minutes_to_dropoff / 60.0

        for did in busy_drivers:
            d = driver_objects.get(did)
            if d is None:
                continue

            if d.wants_offline:
                continue

            # If shift ends before (or at) current dropoff, don't assign a next ride
            if d.offline_time is not None and d.pending_dropoff_time is not None and d.offline_time <= d.pending_dropoff_time:
                continue

            # Must be actively in a ride with a known dropoff time
            if d.status != "in_ride" or d.pending_dropoff_time is None:
                continue

            # Already has next ride assigned? skip
            if d.pending_pickup:
                continue

            # Must finish soon enough
            time_to_dropoff = d.pending_dropoff_time - current_time
            if time_to_dropoff < 0:
                continue
            if time_to_dropoff > max_hr:
                continue

            # Choose by closeness of *future dropoff location* to the new pickup
            dist = euclid(d.pending_dropoff_x, d.pending_dropoff_y, rx, ry)
            if dist < best_dist:
                best_dist, best_id = dist, did

        return best_id
    
    def find_closest_rider(dx, dy):
        if not waiting_riders:
            return None
        best_id = None
        best_dist = float("inf")
        for rid in waiting_riders:
            r = rider_objects[rid]
            dist = euclid(r.pickup_x, r.pickup_y, dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_id = rid
        return best_id

    def match_driver_to_rider(driver_id: int, rider_id: int, t: float):
        d = driver_objects[driver_id]
        r = rider_objects[rider_id]

        # mark match
        d.busy = True
        d.busy_since = t
        r.status = "matched"
        r.driver_id = driver_id
        if d.pending_dropoff_time != None:
            dist_to_pickup = euclid(d.pending_dropoff_x, d.pending_dropoff_y, r.pickup_x, r.pickup_y)
        else:
            # store deadhead distance for later petrol accounting
            d.status = "deadhead"
            dist_to_pickup = euclid(d.x, d.y, r.pickup_x, r.pickup_y)
        r.pickup_dist = dist_to_pickup

        # remove from pools
        if driver_id in idle_drivers:
            idle_drivers.remove(driver_id)
            busy_drivers.append(driver_id)
        if rider_id in waiting_riders:
            waiting_riders.remove(rider_id)

        # schedule pickup

        # if the driver is going straight from one dropoff to one pickup, we must add their current time to dropoff to the pickup time
        mu_to_pickup = mean_drive_time_hr(dist_to_pickup)
        travel_time = actual_drive_time_hr(mu_to_pickup)
        if d.pending_dropoff_time != None: 
            pickup_time = d.pending_dropoff_time + travel_time
            d.pending_pickup = True  # mark that this driver has a next ride queued
        else:
            pickup_time = t + travel_time

        push(Event(time=pickup_time, kind="pickup", rider_id=rider_id, driver_id=driver_id))

    current_time = 0.0

    while heap:
        ev = pop()
        current_time = ev.time

        if current_time > T_end:
            break

        # -------------------
        # DRIVER ARRIVES
        # -------------------
        if ev.kind == "driver_arrives":
            driver_id = next_driver_id
            next_driver_id += 1
            x, y = sample_location()

            driver_objects[driver_id] = Driver(
                id=driver_id, x=x, y=y, online=True,
                arrival_time=current_time
            )
            idle_drivers.append(driver_id)

            driver_earnings[driver_id] = 0.0
            driver_busy_time[driver_id] = 0.0
            driver_online_time[driver_id] = 0.0

            online_dur = sample_driver_online_duration_hr()
            off_time = current_time + online_dur
            driver_objects[driver_id].offline_time = off_time
            driver_objects[driver_id].wants_offline = False

            push(Event(time=off_time, kind="driver_offline", driver_id=driver_id))
            push(Event(time=current_time + sample_driver_interarrival_hr(), kind="driver_arrives"))

            # try match immediately
            rid = find_closest_rider(x, y)
            if rid is not None:
                match_driver_to_rider(driver_id, rid, current_time)

        # -------------------
        # DRIVER OFFLINE
        # -------------------
        elif ev.kind == "driver_offline":
            driver_id = ev.driver_id
            if driver_id not in driver_objects:
                continue
            d = driver_objects[driver_id]
            if d.busy:
                d.wants_offline = True
                if driver_id in busy_drivers:
                    busy_drivers.remove(driver_id)
            else:
                if driver_id in idle_drivers:
                    idle_drivers.remove(driver_id)
                elif driver_id in busy_drivers:
                    busy_drivers.remove(driver_id)
                record_online_time(driver_id, current_time)
                del driver_objects[driver_id]

        # -------------------
        # RIDER REQUEST
        # -------------------
        elif ev.kind == "rider_request":
            rider_id = next_rider_id
            next_rider_id += 1
            px, py = sample_location()
            dx, dy = sample_location()

            rider_objects[rider_id] = Rider(
                id=rider_id, request_time=current_time,
                pickup_x=px, pickup_y=py, dropoff_x=dx, dropoff_y=dy,
                status="waiting"
            )

            push(Event(time=current_time + sample_rider_interarrival_hr(), kind="rider_request"))

            did = find_closest_driver_improved(px, py)
            if did is not None:
                match_driver_to_rider(did, rider_id, current_time)
            else:
                waiting_riders.append(rider_id)
                patience = sample_patience_hr()
                cancel_time = current_time + patience
                rider_objects[rider_id].cancel_time = cancel_time
                push(Event(time=cancel_time, kind="rider_cancel", rider_id=rider_id))

        # -------------------
        # RIDER CANCEL
        # -------------------
        elif ev.kind == "rider_cancel":
            rider_id = ev.rider_id
            if rider_id not in rider_objects:
                continue
            r = rider_objects[rider_id]
            if r.status == "waiting":
                r.status = "cancelled"
                if rider_id in waiting_riders:
                    waiting_riders.remove(rider_id)
                if after_burnin(current_time):
                    abandoned_count += 1

                # Clear rider state
                del rider_objects[rider_id]

        # -------------------
        # PICKUP
        # -------------------
        elif ev.kind == "pickup":
            rider_id = ev.rider_id
            driver_id = ev.driver_id

            if driver_id not in driver_objects:
                continue

            d = driver_objects[driver_id]

            # Im confused on when the scenario below would actually happen and running with it commented out doesn't seem to change the results
            # so I think we don't need it

            # If rider object missing entirely, just free the driver safely
            #if rider_id not in rider_objects:
            #    # Driver arrives somewhere unknown; safest is to just idle them where they are.
            #    d.busy = False
            #    d.busy_since = None
            #    if d.wants_offline:
            #        record_online_time(driver_id, current_time)
            #        del driver_objects[driver_id]
            #    else:
            #        if driver_id not in idle_drivers:
            #            idle_drivers.append(driver_id)
            #    continue

            r = rider_objects[rider_id]

            # Driver arrives at pickup location either way
            d.x = r.pickup_x
            d.y = r.pickup_y

            ## Normal pickup (rider matched)
            r.status = "in_ride"
            d.status = "in_ride"
            d.pending_pickup = False

            # record wait time after burn-in
            if after_burnin(current_time):
                completed_waits.append(current_time - r.request_time)

            dist_trip = euclid(r.pickup_x, r.pickup_y, r.dropoff_x, r.dropoff_y)
            mu_trip = mean_drive_time_hr(dist_trip)
            trip_time = actual_drive_time_hr(mu_trip)
            dropoff_t = current_time + trip_time
            d.pending_dropoff_time = dropoff_t
            d.pending_dropoff_x = r.dropoff_x
            d.pending_dropoff_y = r.dropoff_y

            push(Event(time=dropoff_t, kind="dropoff", rider_id=rider_id, driver_id=driver_id))

        # -------------------
        # DROPOFF
        # -------------------
        elif ev.kind == "dropoff":
            rider_id = ev.rider_id
            driver_id = ev.driver_id

            if driver_id not in driver_objects:
                continue
            if rider_id not in rider_objects:
                continue

            d = driver_objects[driver_id]
            #print(current_time)
            #print(d)
            r = rider_objects[rider_id]

            dist_trip = euclid(r.pickup_x, r.pickup_y, r.dropoff_x, r.dropoff_y)
            total_miles = r.pickup_dist + dist_trip

            fare = fare_base + fare_per_mile * dist_trip
            cost = cost_per_mile * total_miles

            # move driver to dropoff location
            d.x = r.dropoff_x
            d.y = r.dropoff_y

            if after_burnin(current_time):
                driver_earnings[driver_id] = driver_earnings.get(driver_id, 0.0) + (fare - cost)

            # busy time from match to dropoff (count only after burn-in)
            if d.busy_since is not None:
                add_busy_time(driver_id, d.busy_since, current_time)

            completed_count += 1 if after_burnin(current_time) else 0

            if d.pending_pickup == True:
                d.status = "deadhead"
            else:
                d.busy = False
                d.busy_since = None
                d.status = "idle"

            d.pending_dropoff_x = None
            d.pending_dropoff_y = None
            d.pending_dropoff_time = None

            del rider_objects[rider_id]

            if d.wants_offline:
                record_online_time(driver_id, current_time)
                if driver_id in busy_drivers:
                    busy_drivers.remove(driver_id)
                del driver_objects[driver_id]
            elif d.pending_pickup == False:
                next_rid = find_closest_rider(d.x, d.y)
                if next_rid is not None:
                    match_driver_to_rider(driver_id, next_rid, current_time)
                else:
                    if (driver_id not in idle_drivers) and (d.pending_pickup == False):
                        idle_drivers.append(driver_id)

                        

    # cleanup online time for drivers still online at T_end
    for did, d in list(driver_objects.items()):
        start = max(d.arrival_time, burn_in)
        if T_end > start:
            driver_online_time[did] = driver_online_time.get(did, 0.0) + (T_end - start)

    total_requests = completed_count + abandoned_count
    avg_wait = float(np.mean(completed_waits)) * 60 if completed_waits else 0.0
    abandonment_rate = abandoned_count / total_requests if total_requests > 0 else 0.0

    earnings_per_hr = []
    utilisation = []

    for did, earned in driver_earnings.items():
        online = driver_online_time.get(did, 0.0)
        busy = driver_busy_time.get(did, 0.0)
        if online > 0:
            earnings_per_hr.append(earned / online)
            utilisation.append(busy / online)

    avg_earnings_per_hr = float(np.mean(earnings_per_hr)) if earnings_per_hr else 0.0
    avg_utilisation = float(np.mean(utilisation)) if utilisation else 0.0
    earnings_std = float(np.std(earnings_per_hr)) if earnings_per_hr else 0.0

    return {
        "completed_rides": completed_count,
        "abandoned_rides": abandoned_count,
        "total_requests": total_requests,
        "abandonment_rate": round(abandonment_rate, 4),
        "avg_wait_min": round(avg_wait, 2),
        "avg_earnings_per_hr": round(avg_earnings_per_hr, 2),
        "avg_utilisation": round(avg_utilisation, 4),
        "earnings_std": round(earnings_std, 2),
        "driver_earnings": driver_earnings,
        "driver_online_time": driver_online_time,
        "driver_busy_time": driver_busy_time,
    }


# =========================
# SINGLE RUN
# =========================
results = run_simulation(T_end=200, seed=42, burn_in=5.0)

print("=" * 55)
print("SIMULATION RESULTS  (single run, seed=42)")
print("=" * 55)
print(f"  Completed rides    : {results['completed_rides']:,}")
print(f"  Abandoned rides    : {results['abandoned_rides']:,}")
print(f"  Total requests     : {results['total_requests']:,}")
print(f"  Abandonment rate   : {results['abandonment_rate']*100:.1f}%")
print(f"  Avg wait time      : {results['avg_wait_min']:.2f} minutes")
print(f"  Avg earnings/hr    : £{results['avg_earnings_per_hr']:.2f}")
print(f"  Avg utilisation    : {results['avg_utilisation']*100:.1f}%")
print(f"  Earnings std/hr    : £{results['earnings_std']:.2f}")


# =========================
# REPLICATIONS + 95% CI
# =========================
N_REPS = 100

rep_abandonment = []
rep_wait = []
rep_earnings = []
rep_utilisation = []
rep_earnings_std = []

print(f"\nRunning {N_REPS} replications...")

for rep in range(N_REPS):
    r = run_simulation(T_end=200, seed=rep, burn_in=5.0)
    rep_abandonment.append(r["abandonment_rate"])
    rep_wait.append(r["avg_wait_min"])
    rep_earnings.append(r["avg_earnings_per_hr"])
    rep_utilisation.append(r["avg_utilisation"])
    rep_earnings_std.append(r["earnings_std"])


def confidence_interval(data):
    n = len(data)
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1)) / np.sqrt(n)
    t_c = t_dist.ppf(0.975, df=n - 1)
    return mean, mean - t_c * se, mean + t_c * se


ab_mean, ab_lo, ab_hi = confidence_interval(rep_abandonment)
wt_mean, wt_lo, wt_hi = confidence_interval(rep_wait)
er_mean, er_lo, er_hi = confidence_interval(rep_earnings)
ut_mean, ut_lo, ut_hi = confidence_interval(rep_utilisation)
es_mean, es_lo, es_hi = confidence_interval(rep_earnings_std)

print("\n" + "=" * 65)
print(f"REPLICATION RESULTS  ({N_REPS} runs, mean ± 95% CI)")
print("=" * 65)
print(f"  Abandonment rate   : {ab_mean*100:.1f}%  [{ab_lo*100:.1f}%, {ab_hi*100:.1f}%]")
print(f"  Avg wait time      : {wt_mean:.2f} min  [{wt_lo:.2f}, {wt_hi:.2f}]")
print(f"  Avg earnings/hr    : £{er_mean:.2f}  [£{er_lo:.2f}, £{er_hi:.2f}]")
print(f"  Avg utilisation    : {ut_mean*100:.1f}%  [{ut_lo*100:.1f}%, {ut_hi*100:.1f}%]")
print(f"  Earnings std/hr    :git £{es_mean:.2f}  [£{es_lo:.2f}, £{es_hi:.2f}]")