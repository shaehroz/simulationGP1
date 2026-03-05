import numpy as np
import pandas as pd
import heapq
import ast
from dataclasses import dataclass
from scipy.stats import (chisquare, kstest, expon, uniform,
                         chi2 as chi2_dist, gaussian_kde,
                         multivariate_normal, truncnorm, norm,
                         t as t_dist, pearsonr)
from scipy import stats
import matplotlib
matplotlib.use("TkAgg")        # shows plots on screen; change to "Agg" to save only
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.05
SEED  = 42

# Update these paths to wherever your data files live
RIDERS_PATH  = r"riders.xlsx"
DRIVERS_PATH = r"drivers.xlsx"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND PARSE DATA
# ══════════════════════════════════════════════════════════════════════════════

riders  = pd.read_excel(RIDERS_PATH)
drivers = pd.read_excel(DRIVERS_PATH)

def convert_to_xy(v):
    if isinstance(v, tuple):
        return v
    return ast.literal_eval(v)

drivers["initial_location"] = drivers["initial_location"].apply(convert_to_xy)
drivers["current_location"] = drivers["current_location"].apply(convert_to_xy)
drivers[["x", "y"]]                = pd.DataFrame(drivers["current_location"].tolist(),  index=drivers.index)
drivers[["initial_x", "initial_y"]] = pd.DataFrame(drivers["initial_location"].tolist(), index=drivers.index)

riders["pickup_location"]  = riders["pickup_location"].apply(convert_to_xy)
riders["dropoff_location"] = riders["dropoff_location"].apply(convert_to_xy)
riders[["pickup_x",  "pickup_y"]]  = pd.DataFrame(riders["pickup_location"].tolist(),  index=riders.index)
riders[["dropoff_x", "dropoff_y"]] = pd.DataFrame(riders["dropoff_location"].tolist(), index=riders.index)

print(f"Loaded {len(riders):,} rider records and {len(drivers):,} driver records.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRAIN / TEST SPLIT UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def split(arr, frac=0.70, seed=SEED):
    """70/30 train-test split with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(arr))
    cut = int(len(arr) * frac)
    return arr[idx[:cut]], arr[idx[cut:]]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ARRIVAL AND DURATION DISTRIBUTION TESTING
# ══════════════════════════════════════════════════════════════════════════════

rider_iat  = riders["request_time"].sort_values().diff().dropna().values
driver_iat = drivers["arrival_time"].sort_values().diff().dropna().values
driver_dur = (drivers["offline_time"] - drivers["arrival_time"]).values

rider_iat_train,  rider_iat_test  = split(rider_iat)
driver_iat_train, driver_iat_test = split(driver_iat)
driver_dur_train, driver_dur_test = split(driver_dur)

rider_rate_mle  = 1.0 / rider_iat_train.mean()
driver_rate_mle = 1.0 / driver_iat_train.mean()
dur_a_mle       = driver_dur_train.min()
dur_b_mle       = driver_dur_train.max()

print("\n" + "=" * 70)
print("MLE PARAMETER ESTIMATES  (70% training data)")
print("=" * 70)
print(f"  Rider IAT   MLE rate : {rider_rate_mle:.2f}/hr  (brief: 30.00/hr)  mean = {60/rider_rate_mle:.2f} min")
print(f"  Driver IAT  MLE rate : {driver_rate_mle:.2f}/hr  (brief: 3.00/hr)   mean = {60/driver_rate_mle:.2f} min")
print(f"  Driver dur  MLE range: [{dur_a_mle:.2f}, {dur_b_mle:.2f}] hrs  (brief: [5, 8])")


def run_chi2(data, dist, params, n_bins=20):
    """Chi-squared GOF test. Merges bins with expected < 5. Adjusts df for params."""
    n = len(data)
    if dist == "exponential":
        rate   = params["rate"]
        lo, hi = 0, np.percentile(data, 99)
        edges  = np.linspace(lo, hi, n_bins + 1)
        cdf    = expon.cdf(edges, scale=1 / rate)
    elif dist == "uniform":
        a, b  = params["a"], params["b"]
        edges = np.linspace(a, b, n_bins + 1)
        cdf   = uniform.cdf(edges, loc=a, scale=b - a)
    observed = np.histogram(data, bins=edges)[0].astype(float)
    expected = np.diff(cdf) * n
    expected = expected * (observed.sum() / expected.sum())   # normalise to avoid float errors
    obs_m, exp_m = list(observed), list(expected)
    i = 0
    while i < len(exp_m):
        if exp_m[i] < 5:
            j = i + 1 if i < len(exp_m) - 1 else i - 1
            obs_m[j] += obs_m.pop(i); exp_m[j] += exp_m.pop(i); i = max(0, i - 1)
        else:
            i += 1
    stat, _ = chisquare(f_obs=obs_m, f_exp=exp_m)
    n_params = 1 if dist == "exponential" else 2
    df = max(len(obs_m) - 1 - n_params, 1)
    p  = 1 - chi2_dist.cdf(stat, df)
    return round(stat, 3), round(p, 4)


def run_all_tests(label, data_test, dist, brief_params, mle_params):
    """Runs chi-squared and KS against both brief and MLE parameters."""
    dec = lambda p: "REJECT H0" if p < ALPHA else "Fail to reject H0"
    chi2_s_b, chi2_p_b = run_chi2(data_test, dist, brief_params)
    chi2_s_m, chi2_p_m = run_chi2(data_test, dist, mle_params)
    if dist == "exponential":
        ks_b = kstest(data_test, "expon", args=(0, 1 / brief_params["rate"]))
        ks_m = kstest(data_test, "expon", args=(0, 1 / mle_params["rate"]))
    else:
        ks_b = kstest(data_test, "uniform", args=(brief_params["a"], brief_params["b"] - brief_params["a"]))
        ks_m = kstest(data_test, "uniform", args=(mle_params["a"],   mle_params["b"]   - mle_params["a"]))
    print(f"\n{'─' * 65}")
    print(f"  {label}   (n_test = {len(data_test):,})")
    print(f"{'─' * 65}")
    print(f"  BRIEF parameters:")
    print(f"    Chi-squared : stat = {chi2_s_b},  p = {chi2_p_b}  →  {dec(chi2_p_b)}")
    print(f"    KS test     : stat = {round(ks_b.statistic, 4)},  p = {round(ks_b.pvalue, 4)}  →  {dec(ks_b.pvalue)}")
    print(f"  MLE parameters:")
    print(f"    Chi-squared : stat = {chi2_s_m},  p = {chi2_p_m}  →  {dec(chi2_p_m)}")
    print(f"    KS test     : stat = {round(ks_m.statistic, 4)},  p = {round(ks_m.pvalue, 4)}  →  {dec(ks_m.pvalue)}")


print("\n" + "=" * 65)
print("GOODNESS-OF-FIT TESTS — ARRIVAL AND DURATION  (alpha = 0.05)")
print("=" * 65)

run_all_tests("Rider Inter-Arrival Time",  rider_iat_test,  "exponential",
              {"rate": 30.0}, {"rate": rider_rate_mle})
run_all_tests("Driver Inter-Arrival Time", driver_iat_test, "exponential",
              {"rate": 3.0},  {"rate": driver_rate_mle})
run_all_tests("Driver Online Duration",    driver_dur_test, "uniform",
              {"a": 5.0, "b": 8.0}, {"a": dur_a_mle, "b": dur_b_mle})

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPATIAL MODEL SELECTION
#
# Narrative:
#   Step 1 — Reject Uniform(0,20) by KS test
#   Step 2 — Reject Truncated Normal: (a) marginals fail KS + chi2,
#                                      (b) independence rejected by Pearson r
#   Step 3 — Reject Bivariate Normal: necessary condition (normal marginals)
#             violated; TN and BVN fail for identical reason (marginal shape)
#   Step 4 — Adopt 4D KDE as the only statistically justified model
# ══════════════════════════════════════════════════════════════════════════════

def make_grid(resolution=60):
    """Returns meshgrid X, Y and flattened grid_coords for density evaluation."""
    xg = np.linspace(0, 20, resolution)
    yg = np.linspace(0, 20, resolution)
    X, Y = np.meshgrid(xg, yg)
    return X, Y, np.vstack([X.ravel(), Y.ravel()])


def chi2_1d(data, cdf_fn, n_bins=20, n_params=2):
    """Chi-squared GOF for a 1D distribution over [0,20]."""
    n     = len(data)
    edges = np.linspace(0, 20, n_bins + 1)
    cdf   = np.clip(cdf_fn(edges), 0, 1)
    exp   = np.diff(cdf) * n
    obs   = np.histogram(data, bins=edges)[0].astype(float)
    exp   = exp * (obs.sum() / exp.sum())
    obs_m, exp_m = list(obs), list(exp)
    i = 0
    while i < len(exp_m):
        if exp_m[i] < 5:
            j = i + 1 if i < len(exp_m) - 1 else i - 1
            obs_m[j] += obs_m.pop(i); exp_m[j] += exp_m.pop(i); i = max(0, i - 1)
        else:
            i += 1
    stat, _ = chisquare(f_obs=obs_m, f_exp=exp_m)
    df = max(len(obs_m) - 1 - n_params, 1)
    p  = 1 - chi2_dist.cdf(stat, df)
    return round(stat, 3), round(p, 4), df


def ks_1d(data, cdf_fn):
    """KS test: returns D statistic and p-value."""
    ks = kstest(data, cdf_fn)
    return round(ks.statistic, 5), round(ks.pvalue, 6)


def kl_divergence_kde_vs_uniform(xs, ys, resolution=60):
    """
    Approximate KL(KDE || Uniform) on [0,20]^2 using a grid.
    KL = integral f(x) log(f(x)/g(x)) dx  where g = 1/400 (uniform density).
    """
    X, Y, grid = make_grid(resolution)
    kde  = gaussian_kde(np.vstack([xs, ys]))
    f    = kde(grid).reshape(resolution, resolution)
    g    = 1.0 / 400.0   # uniform density on 20x20 area
    cell = (20.0 / resolution) ** 2
    mask = f > 0
    kl   = np.sum(f[mask] * np.log(f[mask] / g) * cell)
    return round(float(kl), 3)


def kl_divergence_two_kdes(xs1, ys1, xs2, ys2, resolution=60):
    """Approximate KL(KDE1 || KDE2) on [0,20]^2."""
    X, Y, grid = make_grid(resolution)
    kde1 = gaussian_kde(np.vstack([xs1, ys1]))
    kde2 = gaussian_kde(np.vstack([xs2, ys2]))
    f1   = kde1(grid).reshape(resolution, resolution)
    f2   = kde2(grid).reshape(resolution, resolution)
    cell = (20.0 / resolution) ** 2
    mask = (f1 > 0) & (f2 > 0)
    kl   = np.sum(f1[mask] * np.log(f1[mask] / f2[mask]) * cell)
    return round(float(kl), 3)


def spatial_model_selection(riders_df, drivers_df):
    """
    Full spatial model selection pipeline.

    Tests Uniform → Truncated Normal → Bivariate Normal, eliminates each
    with formal reasoning, then adopts the 4D KDE.

    Produces five figures:
      1. spatial_kde_heatmaps.png         — KDE heatmaps for all 3 location types
      2. spatial_pickup_vs_dropoff.png    — directional travel pattern
      3. all_model_comparison_contours.png — 3x3 contour grid (KDE/BVN/TN x 3 types)
      4. all_marginals_comparison.png     — 3x2 marginal histograms
      5. spatial_kde_vs_bvn.png           — KDE vs BVN for pickup (legacy figure)
    """

    print("\n" + "=" * 70)
    print("SPATIAL MODEL SELECTION")
    print("Parameters fitted on full data (MLE); GOF tests on 30% test split")
    print("=" * 70)

    X, Y, grid_coords = make_grid(resolution=60)

    loc_datasets = [
        ("Rider Pickup",   riders_df["pickup_x"].values,   riders_df["pickup_y"].values),
        ("Rider Dropoff",  riders_df["dropoff_x"].values,  riders_df["dropoff_y"].values),
        ("Driver Initial", drivers_df["initial_x"].values, drivers_df["initial_y"].values),
    ]

    dec = lambda p: "REJECT" if p < ALPHA else "fail to reject"

    # ── Step 1: Uniform rejection ─────────────────────────────────────────────
    print("\n── STEP 1: Uniform(0,20) ────────────────────────────────────────────")
    for label, xs, ys in loc_datasets:
        _, xs_t = split(xs)
        _, ys_t = split(ys)
        ksx = kstest(xs_t, "uniform", args=(0, 20))
        ksy = kstest(ys_t, "uniform", args=(0, 20))
        print(f"  {label}: X KS p={round(ksx.pvalue,4)} → {dec(ksx.pvalue)} | "
              f"Y KS p={round(ksy.pvalue,4)} → {dec(ksy.pvalue)}")
    print("  → Uniform(0,20) REJECTED for all location types.")

    # ── Step 2: Truncated Normal ──────────────────────────────────────────────
    print("\n── STEP 2: Truncated Normal (independent marginals) ────────────────")
    for label, xs_all, ys_all in loc_datasets:
        _, xs = split(xs_all)
        _, ys = split(ys_all)
        mu_x, sig_x = xs_all.mean(), xs_all.std()
        mu_y, sig_y = ys_all.mean(), ys_all.std()
        a_x, b_x = (0 - mu_x) / sig_x, (20 - mu_x) / sig_x
        a_y, b_y = (0 - mu_y) / sig_y, (20 - mu_y) / sig_y
        tn_x = truncnorm(a_x, b_x, loc=mu_x, scale=sig_x)
        tn_y = truncnorm(a_y, b_y, loc=mu_y, scale=sig_y)
        corr, p_corr = pearsonr(xs_all, ys_all)

        cx, pcx, dfx = chi2_1d(xs, tn_x.cdf, n_params=2)
        cy, pcy, dfy = chi2_1d(ys, tn_y.cdf, n_params=2)
        Dx, pks_x    = ks_1d(xs, tn_x.cdf)
        Dy, pks_y    = ks_1d(ys, tn_y.cdf)

        print(f"\n  {label}:")
        print(f"    TN X: chi2={cx} (df={dfx}) p={pcx} → {dec(pcx)} | "
              f"KS D={Dx} p={pks_x} → {dec(pks_x)}")
        print(f"    TN Y: chi2={cy} (df={dfy}) p={pcy} → {dec(pcy)} | "
              f"KS D={Dy} p={pks_y} → {dec(pks_y)}")
        print(f"    Pearson r={corr:.4f}, p={p_corr:.2e}  "
              f"→ independence {'REJECTED' if p_corr < ALPHA else 'not rejected'}")

    print("\n  → TN eliminated: (1) marginals rejected by both tests,")
    print("                   (2) X⊥Y independence rejected for rider locations.")

    # ── Step 3: Bivariate Normal ──────────────────────────────────────────────
    print("\n── STEP 3: Bivariate Normal ─────────────────────────────────────────")
    print("  BVN requires normal marginals (necessary condition, not assumption).")
    print("  Testing BVN marginals = testing normality of each coordinate.\n")
    for label, xs_all, ys_all in loc_datasets:
        _, xs = split(xs_all)
        _, ys = split(ys_all)
        mu_x  = xs_all.mean(); sig_x = xs_all.std()
        mu_y  = ys_all.mean(); sig_y = ys_all.std()
        cov   = np.cov(xs_all, ys_all)
        rho   = cov[0, 1] / (np.sqrt(cov[0, 0]) * np.sqrt(cov[1, 1]))
        bvn_x_cdf = lambda v, m=mu_x, s=np.sqrt(cov[0,0]): norm.cdf(v, loc=m, scale=s)
        bvn_y_cdf = lambda v, m=mu_y, s=np.sqrt(cov[1,1]): norm.cdf(v, loc=m, scale=s)

        cx, pcx, dfx = chi2_1d(xs, bvn_x_cdf, n_params=2)
        cy, pcy, dfy = chi2_1d(ys, bvn_y_cdf, n_params=2)
        Dx, pks_x    = ks_1d(xs, bvn_x_cdf)
        Dy, pks_y    = ks_1d(ys, bvn_y_cdf)

        print(f"  {label}  (rho={rho:.4f}):")
        print(f"    BVN X marginal: chi2={cx} p={pcx} → {dec(pcx)} | KS D={Dx} p={pks_x} → {dec(pks_x)}")
        print(f"    BVN Y marginal: chi2={cy} p={pcy} → {dec(pcy)} | KS D={Dy} p={pks_y} → {dec(pks_y)}")

    print("\n  → BVN eliminated: normal marginals required but rejected.")
    print("    Note: TN and BVN chi2 statistics are near-identical — both")
    print("    fail for the same reason (marginal shape), not correlation.")

    # ── Step 4: Spatial mismatch and KL divergence ────────────────────────────
    print("\n── STEP 4: Spatial Mismatch Quantification ──────────────────────────")

    px = riders_df["pickup_x"].values;  py = riders_df["pickup_y"].values
    dx = riders_df["dropoff_x"].values; dy = riders_df["dropoff_y"].values
    ix = drivers_df["initial_x"].values; iy = drivers_df["initial_y"].values

    mu_pu  = np.array([px.mean(), py.mean()])
    mu_do  = np.array([dx.mean(), dy.mean()])
    mu_dr  = np.array([ix.mean(), iy.mean()])
    mu_uni = np.array([10.0, 10.0])

    corr_pu, p_pu = pearsonr(px, py)
    corr_do, p_do = pearsonr(dx, dy)
    corr_dr, p_dr = pearsonr(ix, iy)

    kl_pu_uni = kl_divergence_kde_vs_uniform(px, py)
    kl_do_uni = kl_divergence_kde_vs_uniform(dx, dy)
    kl_dr_uni = kl_divergence_kde_vs_uniform(ix, iy)
    kl_pu_dr  = kl_divergence_two_kdes(px, py, ix, iy)
    kl_do_dr  = kl_divergence_two_kdes(dx, dy, ix, iy)

    dist_pu_dr  = float(np.linalg.norm(mu_pu - mu_dr))
    dist_do_dr  = float(np.linalg.norm(mu_do - mu_dr))
    dist_pu_uni = float(np.linalg.norm(mu_pu - mu_uni))
    dist_pu_do  = float(np.linalg.norm(mu_pu - mu_do))

    print(f"\n  Centroid locations:")
    print(f"    Rider pickup  : ({mu_pu[0]:.2f}, {mu_pu[1]:.2f})  σ=({px.std():.2f}, {py.std():.2f})  ρ={corr_pu:.4f} (p={p_pu:.2e})")
    print(f"    Rider dropoff : ({mu_do[0]:.2f}, {mu_do[1]:.2f})  σ=({dx.std():.2f}, {dy.std():.2f})  ρ={corr_do:.4f} (p={p_do:.2e})")
    print(f"    Driver initial: ({mu_dr[0]:.2f}, {mu_dr[1]:.2f})  σ=({ix.std():.2f}, {iy.std():.2f})  ρ={corr_dr:.4f} (p={p_dr:.2e})")
    print(f"    Uniform(0,20) : (10.00, 10.00)  σ=(5.77, 5.77)  ρ=0.000")

    print(f"\n  Centroid distances:")
    print(f"    ||pickup − driver||  = {dist_pu_dr:.2f} miles  ← structural deadhead mismatch")
    print(f"    ||dropoff − driver|| = {dist_do_dr:.2f} miles")
    print(f"    ||pickup − uniform|| = {dist_pu_uni:.2f} miles")
    print(f"    ||pickup − dropoff|| = {dist_pu_do:.2f} miles  ← directional travel pattern")

    print(f"\n  KL divergences (higher = more different from reference):")
    print(f"    KL(pickup  || uniform) = {kl_pu_uni}  ← riders far from uniform")
    print(f"    KL(dropoff || uniform) = {kl_do_uni}")
    print(f"    KL(driver  || uniform) = {kl_dr_uni}")
    print(f"    KL(pickup  || driver)  = {kl_pu_dr}   ← riders/drivers more similar to each other")
    print(f"    KL(dropoff || driver)  = {kl_do_dr}")
    print(f"\n  → KL(pickup||uniform) = {kl_pu_uni} is {kl_pu_uni/kl_pu_dr:.1f}x larger than")
    print(f"    KL(pickup||driver) = {kl_pu_dr}. Uniform assumption underestimates")
    print(f"    spatial alignment and therefore overestimates deadhead distances.")
    print(f"\n  → Adopted model: 4D KDE (only model not rejected by formal testing)")

    # ── FIGURE 1: KDE heatmaps ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Spatial Density of Locations across Squareshire\n"
        "KDE Heatmap — brighter = higher density",
        fontsize=12, fontweight="bold"
    )
    for ax, (label, xs, ys) in zip(axes, loc_datasets):
        kde_2d = gaussian_kde(np.vstack([xs, ys]))
        Z = kde_2d(grid_coords).reshape(60, 60)
        im = ax.imshow(Z, origin="lower", extent=[0, 20, 0, 20],
                       aspect="auto", cmap="YlOrRd")
        ax.scatter(xs, ys, s=1, alpha=0.12, color="white")
        plt.colorbar(im, ax=ax, label="Density")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("X (miles)"); ax.set_ylabel("Y (miles)")
    plt.tight_layout()
    plt.savefig("spatial_kde_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: spatial_kde_heatmaps.png")

    # ── FIGURE 2: Pickup vs Dropoff directional pattern ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Pickup vs Dropoff Spatial Density\n"
        "Riders travel northward on average — systematic directional pattern",
        fontsize=11, fontweight="bold"
    )
    for ax, xs, ys, title in [
        (axes[0], riders_df["pickup_x"].values,  riders_df["pickup_y"].values,  "Pickup Locations"),
        (axes[1], riders_df["dropoff_x"].values, riders_df["dropoff_y"].values, "Dropoff Locations"),
    ]:
        kde_2d = gaussian_kde(np.vstack([xs, ys]))
        Z = kde_2d(grid_coords).reshape(60, 60)
        ax.contourf(X, Y, Z, levels=12, cmap="Blues", alpha=0.85)
        ax.scatter(xs, ys, s=1, alpha=0.08, color="navy")
        ax.axhline(ys.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"Mean y = {ys.mean():.2f}")
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_title(f"{title}\nMean y = {ys.mean():.2f}  (Uniform expectation: 10.00)",
                     fontweight="bold")
        ax.set_xlabel("X (miles)"); ax.set_ylabel("Y (miles)")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("spatial_pickup_vs_dropoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: spatial_pickup_vs_dropoff.png")

    # ── FIGURE 3: 3x3 contour comparison (KDE / BVN / TN) ────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(18, 17))
    fig.suptitle(
        "Spatial Model Comparison — Rider Pickup, Rider Dropoff, Driver Initial\n"
        "Columns: KDE  |  Bivariate Normal  |  Truncated Normal (independent)",
        fontsize=13, fontweight="bold", y=1.01
    )
    col_titles = [
        "KDE\n(non-parametric)",
        "Bivariate Normal\n(captures correlation)",
        "Truncated Normal\n(assumes X ⊥ Y)"
    ]
    scatter_kw = dict(s=1, alpha=0.06, color="steelblue")

    for row, (label, xs_all, ys_all) in enumerate(loc_datasets):
        mu_x, sig_x = xs_all.mean(), xs_all.std()
        mu_y, sig_y = ys_all.mean(), ys_all.std()
        a_x, b_x    = (0 - mu_x) / sig_x, (20 - mu_x) / sig_x
        a_y, b_y    = (0 - mu_y) / sig_y, (20 - mu_y) / sig_y
        corr_r, _   = pearsonr(xs_all, ys_all)

        tn_x  = truncnorm(a_x, b_x, loc=mu_x, scale=sig_x)
        tn_y  = truncnorm(a_y, b_y, loc=mu_y, scale=sig_y)
        bvn   = multivariate_normal([mu_x, mu_y], np.cov(xs_all, ys_all))
        kde2d = gaussian_kde(np.vstack([xs_all, ys_all]))

        Z_kde = kde2d(grid_coords).reshape(60, 60)
        Z_bvn = bvn.pdf(np.dstack((X, Y)))
        Z_tn  = tn_x.pdf(X) * tn_y.pdf(Y)

        notes = [
            "No assumption\nFits true shape",
            f"ρ={corr_r:.3f} captured\nElliptical contours",
            "ρ ignored\nProduct of marginals"
        ]

        for col, (Z, note) in enumerate(zip([Z_kde, Z_bvn, Z_tn], notes)):
            ax = axes[row][col]
            ax.contourf(X, Y, Z, levels=12, cmap="YlOrRd", alpha=0.85)
            ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.4, alpha=0.35)
            ax.scatter(xs_all, ys_all, **scatter_kw)
            ax.set_xlim(0, 20); ax.set_ylim(0, 20)
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
            ax.set_xlabel("X (miles)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{label}\nY (miles)", fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel("Y (miles)", fontsize=8)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("all_model_comparison_contours.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: all_model_comparison_contours.png")

    # ── FIGURE 4: 3x2 marginal histograms ────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        "Marginal Distributions — Observed vs Truncated Normal vs KDE\n"
        "Rows: Rider Pickup  |  Rider Dropoff  |  Driver Initial",
        fontsize=12, fontweight="bold"
    )
    xplot = np.linspace(0, 20, 300)

    for row, (label, xs_all, ys_all) in enumerate(loc_datasets):
        for col, (data, coord) in enumerate([(xs_all, "X"), (ys_all, "Y")]):
            ax   = axes[row][col]
            mu   = data.mean(); sig = data.std()
            a, b = (0 - mu) / sig, (20 - mu) / sig
            tn   = truncnorm(a, b, loc=mu, scale=sig)
            kde1 = gaussian_kde(data)
            ax.hist(data, bins=50, density=True, alpha=0.35,
                    color="steelblue", label="Observed")
            ax.plot(xplot, tn.pdf(xplot),  color="crimson",  lw=2,
                    label=f"TruncNorm (μ={mu:.1f}, σ={sig:.1f})")
            ax.plot(xplot, kde1(xplot),    color="darkgreen", lw=2,
                    label="KDE (1D)")
            ax.set_xlabel(f"{coord} (miles)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"{label} — {coord} coordinate", fontsize=9, fontweight="bold")
            ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig("all_marginals_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: all_marginals_comparison.png")

    # ── FIGURE 5: KDE vs BVN for pickup only (used in report) ────────────────
    pickup_x = riders_df["pickup_x"].values
    pickup_y = riders_df["pickup_y"].values
    mean_vec = np.array([pickup_x.mean(), pickup_y.mean()])
    cov_mat  = np.cov(pickup_x, pickup_y)
    corr_pu2 = np.corrcoef(pickup_x, pickup_y)[0, 1]
    kde_pu   = gaussian_kde(np.vstack([pickup_x, pickup_y]))
    Z_kde_pu = kde_pu(grid_coords).reshape(60, 60)
    Z_bvn_pu = multivariate_normal(mean_vec, cov_mat).pdf(np.dstack((X, Y)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Rider Pickup Locations — KDE vs Bivariate Normal\n"
        "Both appear elliptical, but BVN is formally invalid (normal marginals rejected)",
        fontsize=11, fontweight="bold"
    )
    for ax, Z, title in [
        (axes[0], Z_kde_pu, "KDE (non-parametric, no distributional assumption)"),
        (axes[1], Z_bvn_pu, f"Bivariate Normal  (μ=({mean_vec[0]:.1f},{mean_vec[1]:.1f}), ρ={corr_pu2:.3f})"),
    ]:
        ax.contourf(X, Y, Z, levels=12, cmap="YlOrRd", alpha=0.8)
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.5, alpha=0.5)
        ax.scatter(pickup_x, pickup_y, s=1, alpha=0.08, color="steelblue")
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (miles)"); ax.set_ylabel("Y (miles)")
    plt.tight_layout()
    plt.savefig("spatial_kde_vs_bvn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: spatial_kde_vs_bvn.png")

    print("\n" + "=" * 70)
    print("SPATIAL MODEL SELECTION COMPLETE — 4D KDE adopted")
    print("=" * 70)


# Run spatial model selection
spatial_model_selection(riders, drivers)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — BVN DIAGNOSTIC FIGURES
#
# Generates three figures used in the report:
#   1. bvn_fitted_contours.png      — fitted BVN density over observed data
#   2. bvn_acceptance_rejection.png — accepted vs rejected samples
#   3. bvn_marginals.png            — observed vs BVN marginal vs KDE reference
# ══════════════════════════════════════════════════════════════════════════════

def generate_bvn_figures(riders_df, save_dir="."):
    """
    Generates and saves all three BVN diagnostic figures.
    Requires bvn_params to already be fitted (called after fit_truncated_bvn).
    """
    import os
    px  = riders_df["pickup_x"].values;  py  = riders_df["pickup_y"].values
    dx  = riders_df["dropoff_x"].values; dy  = riders_df["dropoff_y"].values
    mu_pu  = np.array([px.mean(), py.mean()])
    cov_pu = np.cov(px, py)
    mu_do  = np.array([dx.mean(), dy.mean()])
    cov_do = np.cov(dx, dy)
    rho_pu = cov_pu[0,1] / np.sqrt(cov_pu[0,0] * cov_pu[1,1])
    rho_do = cov_do[0,1] / np.sqrt(cov_do[0,0] * cov_do[1,1])

    xg = np.linspace(0, 20, 60);  yg = np.linspace(0, 20, 60)
    X, Y = np.meshgrid(xg, yg);   pos = np.dstack((X, Y))

    # ── Figure 1: Fitted BVN contours ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Fitted Truncated Bivariate Normal — Pickup and Dropoff Locations\n"
        r"Contours show BVN density; boundary $[0,20]^2$ enforced by acceptance-rejection",
        fontsize=11, fontweight="bold"
    )
    for ax, mu, cov, rho, xs, ys, label, cmap in [
        (axes[0], mu_pu, cov_pu, rho_pu, px, py, "Rider Pickup",  "YlOrRd"),
        (axes[1], mu_do, cov_do, rho_do, dx, dy, "Rider Dropoff", "Blues"),
    ]:
        bvn = multivariate_normal(mean=mu, cov=cov)
        Z   = bvn.pdf(pos)
        cf  = ax.contourf(X, Y, Z, levels=12, cmap=cmap, alpha=0.85)
        ax.contour(X, Y, Z, levels=12, colors="k", linewidths=0.4, alpha=0.4)
        ax.scatter(xs, ys, s=1, alpha=0.07, color="steelblue")
        rect = plt.Rectangle((0,0), 20, 20, fill=False,
                              edgecolor="black", linewidth=2, linestyle="--")
        ax.add_patch(rect)
        plt.colorbar(cf, ax=ax, label="Density")
        ax.set_xlim(-1, 21); ax.set_ylim(-1, 21)
        ax.set_title(f"{label}\nrho={rho:.3f},  mu=({mu[0]:.2f},{mu[1]:.2f})",
                     fontweight="bold")
        ax.set_xlabel("X (miles)"); ax.set_ylabel("Y (miles)")
        ax.legend(fontsize=9)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "bvn_fitted_contours.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path1}")

    # ── Figure 2: Acceptance-rejection illustration ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Acceptance-Rejection Sampling — Enforcing the [0,20]^2 Boundary\n"
        "Red = rejected (outside boundary);  Blue = accepted",
        fontsize=11, fontweight="bold"
    )
    rng_fig = np.random.default_rng(99)
    n_demo  = 2000
    for ax, mu, cov, label in [
        (axes[0], mu_pu, cov_pu, "Pickup"),
        (axes[1], mu_do, cov_do, "Dropoff"),
    ]:
        bvn     = multivariate_normal(mean=mu, cov=cov)
        samples = bvn.rvs(size=n_demo, random_state=rng_fig)
        inside  = ((samples[:,0]>=0)&(samples[:,0]<=20)&
                   (samples[:,1]>=0)&(samples[:,1]<=20))
        ax.scatter(samples[~inside,0], samples[~inside,1],
                   s=4, alpha=0.5, color="crimson",
                   label=f"Rejected ({(~inside).sum()})")
        ax.scatter(samples[inside,0],  samples[inside,1],
                   s=4, alpha=0.3, color="steelblue",
                   label=f"Accepted ({inside.sum()})")
        rect = plt.Rectangle((0,0), 20, 20, fill=False,
                              edgecolor="black", linewidth=2.5, linestyle="--",
                              label="[0,20] boundary")
        ax.add_patch(rect)
        ax.set_xlim(-5, 25); ax.set_ylim(-5, 25)
        ax.set_xlabel("X (miles)"); ax.set_ylabel("Y (miles)")
        acc_rate = inside.sum() / n_demo * 100
        ax.set_title(f"{label}  acceptance rate: {acc_rate:.1f}%", fontweight="bold")
        ax.legend(fontsize=9)
        for v in [0, 20]:
            ax.axvline(v, color="gray", lw=0.8, ls=":")
            ax.axhline(v, color="gray", lw=0.8, ls=":")
    plt.tight_layout()
    path2 = os.path.join(save_dir, "bvn_acceptance_rejection.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path2}")

    # ── Figure 3: Marginals — observed vs BVN vs KDE reference ───────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Marginal Distributions — Observed vs BVN Marginal vs KDE Reference\n"
        "KDE (green dashed) = non-parametric benchmark",
        fontsize=11, fontweight="bold"
    )
    xplot = np.linspace(0, 20, 300)
    cov_do_ = np.cov(dx, dy)
    configs = [
        (axes[0,0], px, px.mean(), np.sqrt(cov_pu[0,0]), "Pickup X"),
        (axes[0,1], py, py.mean(), np.sqrt(cov_pu[1,1]), "Pickup Y"),
        (axes[1,0], dx, dx.mean(), np.sqrt(cov_do_[0,0]), "Dropoff X"),
        (axes[1,1], dy, dy.mean(), np.sqrt(cov_do_[1,1]), "Dropoff Y"),
    ]
    for ax, data, mu_1d, sig_1d, label in configs:
        kde1d = gaussian_kde(data)
        ax.hist(data, bins=60, density=True, alpha=0.35,
                color="steelblue", label="Observed data")
        ax.plot(xplot, norm.pdf(xplot, loc=mu_1d, scale=sig_1d),
                color="crimson", lw=2.5,
                label=f"BVN marginal  N({mu_1d:.2f}, {sig_1d:.2f})")
        ax.plot(xplot, kde1d(xplot),
                color="darkgreen", lw=2.0, linestyle="--",
                label="KDE (non-parametric reference)")
        ax.set_xlabel(f"{label} (miles)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=8); ax.set_xlim(0, 20)
    plt.tight_layout()
    path3 = os.path.join(save_dir, "bvn_marginals.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path3}")
    print("\n  All BVN diagnostic figures generated.")

generate_bvn_figures(riders)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Driver:
    id:              int
    x:               float
    y:               float
    online:          bool        = False
    busy:            bool        = False
    wants_offline:   bool        = False    # True = go offline after current ride completes
    repositioning:   bool        = False    # True = en route to quadrant centroid
    offline_time:    float | None = None    # scheduled offline time
    busy_since:      float | None = None    # time of last match (for busy time accounting)
    arrival_time:    float | None = None    # time driver came online
    pending_dropoff_x: float | None = None
    pending_dropoff_y: float | None = None
    pending_dropoff_time: float | None = None
    pending_pickup: bool = False
    status: str = "idle"

@dataclass
class Rider:
    id:           int
    request_time: float
    pickup_x:     float
    pickup_y:     float
    dropoff_x:    float
    dropoff_y:    float
    pickup_dist:  float       = 0.0         # deadhead distance: driver current loc → pickup
    status:       str         = "waiting"
    cancel_time:  float | None = None
    driver_id:    int   | None = None
    surge_mult:   float        = 1.0        # fare multiplier locked in at matching time

@dataclass(order=True)
class Event:
    time:      float
    kind:      str
    rider_id:  int | None = None
    driver_id: int | None = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRUNCATED BIVARIATE NORMAL TRIP SAMPLER
#
# Following supervisor guidance: fit a Bivariate Normal jointly to (x, y) for
# each of pickup and dropoff, then enforce the [0,20] boundary using
# acceptance-rejection sampling.
#
# Method:
#   1. Fit BVN(mu, Sigma) to observed (pickup_x, pickup_y) via MLE
#      (MLE for BVN = sample mean and sample covariance matrix).
#   2. Fit BVN(mu, Sigma) to observed (dropoff_x, dropoff_y) via MLE.
#   3. To sample one trip: draw (px,py) from pickup BVN, draw (dx,dy) from
#      dropoff BVN, reject any draw where any coordinate falls outside [0,20].
#      Repeat until n accepted draws obtained.
#   4. Pre-sample a pool of n trips before each replication so the event loop
#      never calls the sampler at runtime.
#
# Acceptance rate: empirically ~85% (matching the ~8.5% out-of-bounds rate
# observed with the KDE). Oversample by 25% to guarantee n accepted draws
# in a single batch without looping.
#
# Preserves: x-y correlation (via BVN covariance), spatial bias (via BVN mean),
#            hard [0,20] boundary (via acceptance-rejection).
# Does not preserve: non-elliptical density structure (BVN is elliptical by
#                    construction) — acknowledged in report.
# ══════════════════════════════════════════════════════════════════════════════

def fit_truncated_bvn(riders_df):
    """
    Fits two independent BVNs:
      - pickup  BVN: MLE on (pickup_x,  pickup_y)
      - dropoff BVN: MLE on (dropoff_x, dropoff_y)

    MLE for a BVN is simply the sample mean vector and sample covariance matrix.

    Returns a dict with keys 'pickup' and 'dropoff', each containing
    {'mean': array(2,), 'cov': array(2,2)}.
    Prints fitted parameters and Pearson correlations.
    """
    results = {}
    for label, xcol, ycol in [
        ("pickup",  "pickup_x",  "pickup_y"),
        ("dropoff", "dropoff_x", "dropoff_y"),
    ]:
        xs  = riders_df[xcol].values
        ys  = riders_df[ycol].values
        mu  = np.array([xs.mean(), ys.mean()])
        cov = np.cov(xs, ys)
        rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        results[label] = {"mean": mu, "cov": cov}

        print(f"\n  Truncated BVN — {label}:")
        print(f"    μ  = ({mu[0]:.4f}, {mu[1]:.4f})")
        print(f"    σ  = ({np.sqrt(cov[0,0]):.4f}, {np.sqrt(cov[1,1]):.4f})")
        print(f"    ρ  = {rho:.4f}  (Pearson correlation)")
        print(f"    Σ  = [[{cov[0,0]:.4f}, {cov[0,1]:.4f}],")
        print(f"           [{cov[1,0]:.4f}, {cov[1,1]:.4f}]]")

    return results


def build_trip_pool(bvn_params, n, seed, lo=0.0, hi=20.0):
    """
    Pre-samples n trips using acceptance-rejection on the truncated BVN.

    For each draw:
      - Sample (px, py) from pickup  BVN
      - Sample (dx, dy) from dropoff BVN
      - Accept only if all four coordinates lie in [lo, hi]

    Oversamples by 25% in one batch to achieve n accepted draws without
    looping. If the batch yields fewer than n accepted draws (acceptance
    rate unexpectedly low), raises an informative error.

    Returns array shape (n, 4): columns are [px, py, dx, dy].
    """
    rng         = np.random.default_rng(seed)
    n_draw      = int(n * 1.25)   # 25% oversample — acceptance rate ~85%

    pu = multivariate_normal(mean=bvn_params["pickup"]["mean"],
                             cov=bvn_params["pickup"]["cov"])
    do = multivariate_normal(mean=bvn_params["dropoff"]["mean"],
                             cov=bvn_params["dropoff"]["cov"])

    pu_samples = pu.rvs(size=n_draw, random_state=rng)   # shape (n_draw, 2)
    do_samples = do.rvs(size=n_draw, random_state=rng)   # shape (n_draw, 2)

    # Acceptance mask: all four coordinates inside [lo, hi]
    accept = (
        (pu_samples[:, 0] >= lo) & (pu_samples[:, 0] <= hi) &
        (pu_samples[:, 1] >= lo) & (pu_samples[:, 1] <= hi) &
        (do_samples[:, 0] >= lo) & (do_samples[:, 0] <= hi) &
        (do_samples[:, 1] >= lo) & (do_samples[:, 1] <= hi)
    )

    accepted = np.hstack([pu_samples[accept], do_samples[accept]])  # shape (k, 4)

    acceptance_rate = accept.sum() / n_draw
    print(f"    Acceptance-rejection: {accept.sum():,}/{n_draw:,} accepted "
          f"({acceptance_rate*100:.1f}% acceptance rate)")

    if len(accepted) < n:
        raise RuntimeError(
            f"Acceptance-rejection yielded only {len(accepted)} samples "
            f"(needed {n}). Increase oversample factor or reduce n."
        )

    return accepted[:n]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUADRANT CENTROIDS
#
# Divides the 20×20 grid into four quadrants centred on the rider pickup
# hotspot (empirically at ~8.36, 12.32). Each quadrant's centroid is the
# density-weighted mean of all pickup points within that quadrant — so
# drivers reposition to where demand actually concentrates, not just the
# geometric centre of the quadrant.
#
# Quadrant layout (split at hotspot):
#   Q2 (NW) | Q1 (NE)
#   ─────────────────
#   Q3 (SW) | Q4 (SE)
# ══════════════════════════════════════════════════════════════════════════════

def compute_quadrant_centroids(riders_df, split_x=None, split_y=None):
    """
    Computes the density-weighted centroid of each of the four quadrants,
    split at (split_x, split_y). Defaults to the empirical pickup hotspot.

    Returns a list of four (cx, cy) tuples ordered [NE, NW, SW, SE].
    """
    px = riders_df["pickup_x"].values
    py = riders_df["pickup_y"].values

    if split_x is None: split_x = float(px.mean())
    if split_y is None: split_y = float(py.mean())

    quadrants = {
        "NE": (px >= split_x) & (py >= split_y),
        "NW": (px <  split_x) & (py >= split_y),
        "SW": (px <  split_x) & (py <  split_y),
        "SE": (px >= split_x) & (py <  split_y),
    }

    centroids = {}
    for label, mask in quadrants.items():
        if mask.sum() > 0:
            centroids[label] = (float(px[mask].mean()), float(py[mask].mean()))
        else:
            centroids[label] = (split_x, split_y)   # fallback: split point

    print("\n── Quadrant Centroids (split at pickup hotspot) ──────────────────")
    print(f"   Split point : ({split_x:.2f}, {split_y:.2f})")
    for label, (cx, cy) in centroids.items():
        n = quadrants[label].sum()
        print(f"   {label}: ({cx:.2f}, {cy:.2f})  [{n:,} pickup observations]")

    return [centroids["NE"], centroids["NW"], centroids["SW"], centroids["SE"]]



def compute_surge_multiplier(
    n_waiting,
    n_idle,
    surge_on   = True,
    k          = 0.8,
    r_min      = 2.0,
    r_cap      = 8.0,
    max_mult   = 3.0,
):
    """
    Exponential (Plinko-style) surge multiplier.

    ratio = n_waiting / max(n_idle, 1)
    - Below r_min (2.0): returns 1.0  (no surge)
    - At r_min + r_cap (10.0): returns max_mult (3.0x)
    - Exponentially interpolated between, k=0.8 gives balanced curve
    - Capped at max_mult for any ratio beyond r_min + r_cap
    - surge_on=False disables entirely (returns 1.0 always)
    """
    if not surge_on:
        return 1.0
    ratio  = n_waiting / max(n_idle, 1)
    if ratio < r_min:
        return 1.0
    excess = min(ratio - r_min, r_cap)
    scale  = max_mult - 1.0
    denom  = float(np.exp(k * r_cap) - 1.0)
    return 1.0 + scale * (float(np.exp(k * excess)) - 1.0) / denom


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DISCRETE EVENT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    T_end:                float = 200.0,
    seed:                 int   = 1,
    square_min:           float = 0.0,
    square_max:           float = 20.0,
    lam_driver_on:        float = 4.65,    # MLE-estimated driver arrival rate /hr
    lam_rider_arrival:    float = 34.92,   # MLE-estimated rider arrival rate /hr
    lam_patience:         float = 5.0,     # briefed patience rate /hr (censored)
    speed_mph:            float = 20.0,
    fare_base:            float = 3.0,     # £3 base charge
    fare_per_mile:        float = 2.0,     # £2 per trip mile
    cost_per_mile:        float = 0.20,    # £0.20/mile on ALL miles (deadhead + trip)
    burn_in:              float = 5.0,     # hours before metrics are recorded
    trip_pool:            np.ndarray | None = None,
    # ── Quadrant repositioning ─────────────────────────────────────────────
    quadrant_centroids:   list | None = None,  # list of 4 (x,y) tuples; None = disabled
    # ── Surge pricing ──────────────────────────────────────────────────────
    surge_on:             bool  = True,    # False disables surge entirely
    # ── Detour ──────────────────────────────────────────────────────
    detour_on:             bool  = False,    # False disables detour entirely
    # ── Busy-driver preassignment ─────────────────────────────────────
    busy_driver_on: bool = False,
    busy_driver_max_minutes: float = 10.0,
) -> dict:

    rng        = np.random.default_rng(seed)
    pool_index = [0]

    def sample_trip():
        """Returns (px, py, dx, dy) from KDE pool, or uniform fallback if pool exhausted."""
        if trip_pool is not None and pool_index[0] < len(trip_pool):
            row = trip_pool[pool_index[0]]
            pool_index[0] += 1
            return float(row[0]), float(row[1]), float(row[2]), float(row[3])
        px = float(rng.uniform(square_min, square_max))
        py = float(rng.uniform(square_min, square_max))
        dx = float(rng.uniform(square_min, square_max))
        dy = float(rng.uniform(square_min, square_max))
        return px, py, dx, dy

    def sample_driver_iat():     return float(rng.exponential(scale=1 / lam_driver_on))
    def sample_driver_dur():     return float(rng.uniform(6.0, 8.0))
    def sample_rider_iat():      return float(rng.exponential(scale=1 / lam_rider_arrival))
    def sample_patience():       return float(rng.exponential(scale=1 / lam_patience))
    def euclid(x1, y1, x2, y2): return float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    def mean_time(d):            return d / speed_mph
    def actual_time(mu):         return float(rng.uniform(0.8 * mu, 1.2 * mu))

    def nearest_quadrant_centroid(x, y):
        """Returns the (cx, cy) of the quadrant centroid nearest to (x, y)."""
        return min(quadrant_centroids,
                   key=lambda c: euclid(x, y, c[0], c[1]))
    
    def detour_multiplier(detour_on = False):
        if detour_on == False:
            return 1.0
        else: 
            # mean ~1.4, moderate variability; tune as needed
            m = float(rng.normal(loc=1.4, scale=0.15))
            return max(1.05, min(m, 2.0))

    driver_objects     = {}
    rider_objects      = {}
    idle_drivers       = []
    busy_drivers = []
    waiting_riders     = []
    next_driver_id     = 0
    next_rider_id      = 0
    completed_waits    = []
    abandoned_count    = 0
    completed_count    = 0
    surge_revenue      = 0.0   # total extra revenue collected during surge periods
    surge_ride_count   = 0     # number of rides completed at surge price
    driver_earnings    = {}
    driver_busy_time   = {}
    driver_online_time = {}

    heap = []
    def push(e): heapq.heappush(heap, e)
    def pop():   return heapq.heappop(heap)

    push(Event(time=sample_driver_iat(), kind="driver_arrives"))
    push(Event(time=sample_rider_iat(),  kind="rider_request"))

    def record_online_time(driver_id, current_time):
        d     = driver_objects[driver_id]
        start = max(d.arrival_time, burn_in)
        if current_time > start:
            driver_online_time[driver_id] = (
                driver_online_time.get(driver_id, 0.0) + current_time - start)

    def find_closest_driver(rx, ry):
        """Baseline if busy_driver_on=False. If True, also consider eligible busy drivers near dropoff."""

        # --- Baseline behaviour (unchanged) ---
        if not busy_driver_on:
            available = [did for did in idle_drivers
                         if not driver_objects[did].repositioning]
            if not available:
                available = idle_drivers
            if not available:
                return None
            return min(
                available,
                key=lambda did: euclid(driver_objects[did].x, driver_objects[did].y, rx, ry)
            )

        # --- Busy-driver mode ---
        # Prefer idle drivers first (same as busy file) :contentReference[oaicite:8]{index=8}
        if idle_drivers:
            best_id, best_dist = None, float("inf")
            for did in idle_drivers:
                d = driver_objects[did]
                dist = euclid(d.x, d.y, rx, ry)
                if dist < best_dist:
                    best_dist, best_id = dist, did
            return best_id

        # Otherwise consider busy drivers finishing soon (adapted from busy file) :contentReference[oaicite:9]{index=9}
        best_id, best_dist = None, float("inf")
        max_hr = busy_driver_max_minutes / 60.0

        for did in busy_drivers:
            d = driver_objects.get(did)
            if d is None:
                continue

            # must not be about to go offline
            if d.wants_offline:
                continue

            # if shift ends before/at current dropoff, don't preassign
            if d.offline_time is not None and d.pending_dropoff_time is not None and d.offline_time <= d.pending_dropoff_time:
                continue

            # must be actively in-ride and have a known dropoff time
            if d.status != "in_ride" or d.pending_dropoff_time is None:
                continue

            # must not already have a next ride queued
            if d.pending_pickup:
                continue

            time_to_dropoff = d.pending_dropoff_time - current_time
            if time_to_dropoff < 0 or time_to_dropoff > max_hr:
                continue

            # choose by closeness of future dropoff point to new pickup
            dist = euclid(d.pending_dropoff_x, d.pending_dropoff_y, rx, ry)
            if dist < best_dist:
                best_dist, best_id = dist, did

        return best_id
    
    def find_closest_rider(dx, dy):
        if not waiting_riders: return None
        return min(waiting_riders,
                   key=lambda rid: euclid(rider_objects[rid].pickup_x,
                                          rider_objects[rid].pickup_y, dx, dy))

    def match(driver_id, rider_id, t):
        d = driver_objects[driver_id]
        r = rider_objects[rider_id]

        d.busy = True
        d.repositioning = False
        d.busy_since = t

        r.status = "matched"
        r.driver_id = driver_id

        # ensure driver is tracked as busy in busy-driver mode
        if busy_driver_on and (driver_id not in busy_drivers):
            busy_drivers.append(driver_id)

        # remove from pools
        if driver_id in idle_drivers:
            idle_drivers.remove(driver_id)
        if rider_id in waiting_riders:
            waiting_riders.remove(rider_id)

        # lock in surge multiplier (keep your existing logic) :contentReference[oaicite:15]{index=15}
        r.surge_mult = compute_surge_multiplier(
            n_waiting=len(waiting_riders),
            n_idle=len(idle_drivers),
            surge_on=surge_on,
        )

        # --- distance & pickup scheduling ---
        if busy_driver_on and d.pending_dropoff_time is not None:
            # preassign: driver will go to pickup after finishing current ride
            d.pending_pickup = True
            start_x, start_y = d.pending_dropoff_x, d.pending_dropoff_y
            dist_to_pickup = euclid(start_x, start_y, r.pickup_x, r.pickup_y) * detour_multiplier()
            r.pickup_dist = dist_to_pickup

            travel_time = actual_time(mean_time(dist_to_pickup))
            pickup_time = d.pending_dropoff_time + travel_time
            push(Event(time=pickup_time, kind="pickup", rider_id=rider_id, driver_id=driver_id))
            return

        # baseline: driver is idle now
        d.status = "deadhead" if busy_driver_on else d.status
        dist_to_pickup = euclid(d.x, d.y, r.pickup_x, r.pickup_y) * detour_multiplier()
        r.pickup_dist = dist_to_pickup

        pickup_time = t + actual_time(mean_time(dist_to_pickup))
        push(Event(time=pickup_time, kind="pickup", rider_id=rider_id, driver_id=driver_id))

    def reposition(driver_id, current_time):
        """
        Sends an idle driver to the centroid of their nearest high-demand quadrant.
        Repositioning costs fuel but earns no fare. The driver remains in idle_drivers
        so they can still be matched to a rider — if matched, the reposition_arrive
        event is simply ignored (driver no longer exists in driver_objects or has
        repositioning=False).
        """
        d  = driver_objects[driver_id]
        cx, cy = nearest_quadrant_centroid(d.x, d.y)
        dist   = euclid(d.x, d.y, cx, cy)
        if dist < 0.01:
            return   # already at centroid — no need to move
        d.repositioning = True
        travel_t = actual_time(mean_time(dist))
        # Repositioning fuel cost charged at dropoff-equivalent point
        repo_cost = cost_per_mile * dist
        push(Event(time=current_time + travel_t,
                   kind="reposition_arrive",
                   driver_id=driver_id))
        # Store destination on driver object for fuel accounting
        d.repo_dest_x  = cx
        d.repo_dest_y  = cy
        d.repo_cost    = repo_cost

    current_time = 0.0

    while heap:
        event        = pop()
        current_time = event.time
        if current_time > T_end: break
        post_burnin = current_time >= burn_in

        # ── DRIVER ARRIVES ────────────────────────────────────────────────────
        if event.kind == "driver_arrives":
            did = next_driver_id; next_driver_id += 1
            x   = float(rng.uniform(square_min, square_max))
            y   = float(rng.uniform(square_min, square_max))
            driver_objects[did] = Driver(id=did, x=x, y=y, online=True,
                                         arrival_time=current_time,
                                         wants_offline=False,
                                         repositioning=False)
            idle_drivers.append(did)
            driver_earnings[did]    = 0.0
            driver_busy_time[did]   = 0.0
            driver_online_time[did] = 0.0
            off_t = current_time + sample_driver_dur()
            driver_objects[did].offline_time = off_t
            push(Event(time=off_t,                           kind="driver_offline", driver_id=did))
            push(Event(time=current_time + sample_driver_iat(), kind="driver_arrives"))
            closest_rider = find_closest_rider(x, y)
            if closest_rider is not None:
                match(did, closest_rider, current_time)
            elif quadrant_centroids is not None:
                reposition(did, current_time)

        # ── DRIVER OFFLINE ────────────────────────────────────────────────────
        elif event.kind == "driver_offline":
            did = event.driver_id
            if did not in driver_objects: continue
            d = driver_objects[did]
            if d.busy:
                d.wants_offline = True          # finish current ride first
            else:
                if did in idle_drivers: idle_drivers.remove(did)
                record_online_time(did, current_time)
                del driver_objects[did]

        # ── REPOSITION ARRIVE ─────────────────────────────────────────────────
        elif event.kind == "reposition_arrive":
            did = event.driver_id
            if did not in driver_objects: continue
            d = driver_objects[did]
            if not d.repositioning:
                continue   # was matched to a rider before arriving — nothing to do
            # Driver reaches quadrant centroid
            d.x             = d.repo_dest_x
            d.y             = d.repo_dest_y
            d.repositioning = False
            # Charge repositioning fuel cost to driver earnings
            if post_burnin:
                driver_earnings[did] = driver_earnings.get(did, 0.0) - d.repo_cost
            # Immediately check for waiting riders at new position
            closest_rider = find_closest_rider(d.x, d.y)
            if closest_rider is not None:
                match(did, closest_rider, current_time)

        # ── RIDER REQUEST ─────────────────────────────────────────────────────
        elif event.kind == "rider_request":
            rid = next_rider_id; next_rider_id += 1
            px, py, dx, dy = sample_trip()
            rider_objects[rid] = Rider(id=rid, request_time=current_time,
                                       pickup_x=px, pickup_y=py,
                                       dropoff_x=dx, dropoff_y=dy)
            push(Event(time=current_time + sample_rider_iat(), kind="rider_request"))
            closest_driver = find_closest_driver(px, py)
            if closest_driver is not None:
                match(closest_driver, rid, current_time)
            else:
                waiting_riders.append(rid)
                cancel_t = current_time + sample_patience()
                rider_objects[rid].cancel_time = cancel_t
                push(Event(time=cancel_t, kind="rider_cancel", rider_id=rid))

        # ── RIDER CANCEL ──────────────────────────────────────────────────────
        elif event.kind == "rider_cancel":
            rid = event.rider_id
            if rid not in rider_objects: continue
            r = rider_objects[rid]
            if r.status == "waiting":
                r.status = "cancelled"
                if rid in waiting_riders: waiting_riders.remove(rid)
                if post_burnin: abandoned_count += 1
                del rider_objects[rid]

        # ── PICKUP ────────────────────────────────────────────────────────────
        elif event.kind == "pickup":
            rid = event.rider_id; did = event.driver_id
            if rid not in rider_objects: continue
            r = rider_objects[rid]; d = driver_objects[did]
            r.status = "in_ride"
            if busy_driver_on:
                d.status = "in_ride"
                d.pending_pickup = False
            d.x = r.pickup_x; d.y = r.pickup_y
            if post_burnin:
                completed_waits.append(current_time - r.request_time)
            dist_trip = euclid(r.pickup_x, r.pickup_y, r.dropoff_x, r.dropoff_y)  * detour_multiplier()
            push(Event(time=current_time + actual_time(mean_time(dist_trip)),
                       kind="dropoff", rider_id=rid, driver_id=did))

        # ── DROPOFF ───────────────────────────────────────────────────────────
        elif event.kind == "dropoff":
            rider_id = event.rider_id
            driver_id = event.driver_id

            if driver_id not in driver_objects:
                continue
            if rider_id not in rider_objects:
                continue
            
            d = driver_objects[driver_id]
            r = rider_objects[rider_id]

            # Trip distance (keep your current distance model; if you use detour, apply it here too)
            dist_trip = euclid(r.pickup_x, r.pickup_y, r.dropoff_x, r.dropoff_y)
            total_miles = r.pickup_dist + dist_trip

            fare = fare_base + fare_per_mile * dist_trip
            cost = cost_per_mile * total_miles

            # move driver to dropoff location
            d.x = r.dropoff_x
            d.y = r.dropoff_y

            # earnings (after burn-in)
            if post_burnin:
                driver_earnings[driver_id] = driver_earnings.get(driver_id, 0.0) + (fare - cost)

            # busy time accounting: count from d.busy_since to now, if defined
            if d.busy_since is not None:
                driver_busy_time[driver_id] = driver_busy_time.get(driver_id, 0.0) + (current_time - d.busy_since)
            else:
                driver_busy_time[driver_id] = driver_busy_time.get(driver_id, 0.0)

            # completed ride count (after burn-in)
            if post_burnin:
                completed_count += 1

            # rider is done
            del rider_objects[rider_id]

            # --- BUSY DRIVER MODE: update "pending dropoff" state ---
            if busy_driver_on:
                # This dropoff is now complete, so the "pending dropoff" info is no longer valid
                d.pending_dropoff_x = None
                d.pending_dropoff_y = None
                d.pending_dropoff_time = None

            # Default: driver is no longer busy unless they already have a queued next pickup
            # (If you never set pending_pickup in your implementation, this behaves like baseline.)
            if busy_driver_on and getattr(d, "pending_pickup", False):
                # Driver stays busy, now deadheading to their already-scheduled next pickup.
                d.busy = True
                d.status = "deadhead"
                # Start a new busy interval for the deadhead leg to next pickup
                d.busy_since = current_time

                # Keep them in busy_drivers; ensure not in idle pool
                if driver_id in idle_drivers:
                    idle_drivers.remove(driver_id)
                if driver_id not in busy_drivers:
                    busy_drivers.append(driver_id)

            else:
                # Truly idle (baseline behaviour)
                d.busy = False
                d.busy_since = None
                if busy_driver_on:
                    d.status = "idle"
                    d.pending_pickup = False  # safe reset

                # remove from busy pool if you're tracking it
                if busy_driver_on and driver_id in busy_drivers:
                    busy_drivers.remove(driver_id)

                if d.wants_offline:
                    record_online_time(driver_id, current_time)
                    del driver_objects[driver_id]
                else:
                    next_rid = find_closest_rider(d.x, d.y)
                    if next_rid is not None:
                        match(driver_id, next_rid, current_time)
                    else:
                        if driver_id not in idle_drivers:
                            idle_drivers.append(driver_id)

    # Cleanup: record online time for drivers still active at T_end
    for did, d in list(driver_objects.items()):
        start = max(d.arrival_time, burn_in)
        if T_end > start:
            driver_online_time[did] = driver_online_time.get(did, 0.0) + (T_end - start)

    total_requests   = completed_count + abandoned_count
    avg_wait         = float(np.mean(completed_waits)) * 60 if completed_waits else 0.0
    abandonment_rate = abandoned_count / total_requests if total_requests > 0 else 0.0

    earnings_per_hr = []
    utilisation     = []
    for did in driver_earnings:
        online = driver_online_time.get(did, 0.0)
        busy   = driver_busy_time.get(did, 0.0)
        earned = driver_earnings[did]
        if online > 0:
            earnings_per_hr.append(earned / online)
            utilisation.append(busy / online)

    return {
        "completed_rides"    : completed_count,
        "abandoned_rides"    : abandoned_count,
        "total_requests"     : total_requests,
        "abandonment_rate"   : round(abandonment_rate, 4),
        "avg_wait_min"       : round(avg_wait, 2),
        "avg_earnings_per_hr": round(float(np.mean(earnings_per_hr)) if earnings_per_hr else 0.0, 2),
        "avg_utilisation"    : round(float(np.mean(utilisation))     if utilisation     else 0.0, 4),
        "earnings_std"       : round(float(np.std(earnings_per_hr))  if earnings_per_hr else 0.0, 2),
        "surge_revenue"      : round(surge_revenue, 2),
        "surge_ride_count"   : surge_ride_count,
        "surge_ride_pct"     : round(surge_ride_count / completed_count, 4) if completed_count else 0.0,
        "driver_earnings"    : driver_earnings,
        "driver_online_time" : driver_online_time,
        "driver_busy_time"   : driver_busy_time,
        "completed_waits"    : completed_waits,
        "earnings_per_hr"    : earnings_per_hr,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CONFIDENCE INTERVAL UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def confidence_interval(data):
    """Returns (mean, lower_95, upper_95) using Student's t-distribution."""
    n    = len(data)
    mean = float(np.mean(data))
    se   = float(np.std(data, ddof=1)) / np.sqrt(n)
    t_c  = t_dist.ppf(0.975, df=n - 1)
    return mean, mean - t_c * se, mean + t_c * se

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FIT TRUNCATED BVN AND QUADRANT CENTROIDS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FITTING TRUNCATED BIVARIATE NORMAL  (supervisor-recommended model)")
print("MLE = sample mean and covariance; boundary enforced by acceptance-rejection")
print("=" * 70)

bvn_params = fit_truncated_bvn(riders)
quad_cents = compute_quadrant_centroids(riders)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FOUR-SCENARIO COMPARISON
#
#  Scenario A — Baseline        : no quadrant repositioning, no surge
#  Scenario B — Quadrants only  : repositioning enabled, no surge
#  Scenario C — Surge only      : no repositioning, surge at queue >= 10, ×1.5
#  Scenario D — Both combined   : repositioning + surge
#
# Each scenario runs 20 independent replications (1000hr, 5hr burn-in).
# ══════════════════════════════════════════════════════════════════════════════

N_REPS = 20

scenarios = {
    "A — Baseline"      : dict(quadrant_centroids=None, surge_on=False, detour_on = False, busy_driver_on = False),
    "B — Quadrants"     : dict(quadrant_centroids=quad_cents, surge_on=False, detour_on = False, busy_driver_on = False),
    "C — Surge"         : dict(quadrant_centroids=None, surge_on=True, detour_on = False, busy_driver_on = False),
    "D — Detour"        : dict(quadrant_centroids=None,surge_on=False, detour_on = True, busy_driver_on = False),
    "E — Busy Driver"   : dict(quadrant_centroids=None,surge_on=False, detour_on = False, busy_driver_on = True, busy_driver_max_minutes = 10.0),
    "F — B, C, E combined": dict(quadrant_centroids=quad_cents, surge_on=True, detour_on = False, busy_driver_on = True),
}

all_results = {}

for scenario_name, kwargs in scenarios.items():
    print(f"\n{'═'*65}")
    print(f"  SCENARIO {scenario_name}")
    print(f"{'═'*65}")

    rep_ab, rep_wt, rep_er, rep_ut, rep_es = [], [], [], [], []
    rep_surge_rev, rep_surge_pct           = [], []

    for rep in range(N_REPS):
        pool = build_trip_pool(bvn_params, n=50000, seed=rep)
        r    = run_simulation(T_end=1000, seed=rep, burn_in=5.0,
                              trip_pool=pool, **kwargs)
        rep_ab.append(r["abandonment_rate"])
        rep_wt.append(r["avg_wait_min"])
        rep_er.append(r["avg_earnings_per_hr"])
        rep_ut.append(r["avg_utilisation"])
        rep_es.append(r["earnings_std"])
        rep_surge_rev.append(r["surge_revenue"])
        rep_surge_pct.append(r["surge_ride_pct"])
        print(f"  Rep {rep+1:2d}/20  abandon={r['abandonment_rate']*100:.1f}%  "
              f"wait={r['avg_wait_min']:.1f}min  £{r['avg_earnings_per_hr']:.2f}/hr  "
              f"surge_rev=£{r['surge_revenue']:.0f}")

    all_results[scenario_name] = {
        "abandonment" : confidence_interval(rep_ab),
        "wait"        : confidence_interval(rep_wt),
        "earnings"    : confidence_interval(rep_er),
        "utilisation" : confidence_interval(rep_ut),
        "earn_std"    : confidence_interval(rep_es),
        "surge_rev"   : confidence_interval(rep_surge_rev),
        "surge_pct"   : confidence_interval(rep_surge_pct),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RESULTS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "═"*95)
print(f"  SCENARIO COMPARISON  ({N_REPS} replications each, mean [95% CI])")
print("═"*95)
print(f"  {'Scenario':<22}  {'Abandon %':>10}  {'Wait (min)':>12}  "
      f"{'£/hr':>10}  {'Util %':>8}  {'Surge rev £':>12}  {'Surge %':>8}")
print("─"*95)

for name, res in all_results.items():
    ab  = res["abandonment"]
    wt  = res["wait"]
    er  = res["earnings"]
    ut  = res["utilisation"]
    sr  = res["surge_rev"]
    sp  = res["surge_pct"]
    print(f"  {name:<22}  "
          f"{ab[0]*100:>5.1f} [{ab[1]*100:.1f},{ab[2]*100:.1f}]  "
          f"{wt[0]:>6.2f} [{wt[1]:.2f},{wt[2]:.2f}]  "
          f"{er[0]:>6.2f} [{er[1]:.2f},{er[2]:.2f}]  "
          f"{ut[0]*100:>6.1f}  "
          f"{sr[0]:>8.0f} [{sr[1]:.0f},{sr[2]:.0f}]  "
          f"{sp[0]*100:>6.1f}%")

print("═"*95)
print("  Surge revenue = extra fare collected above baseline during surge periods.")
print("  Surge % = fraction of completed rides that were priced at surge rate.")