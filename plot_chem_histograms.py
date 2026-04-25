import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
import os

_erf = np.vectorize(math.erf)

def _normal_cdf(x, mu, sigma):
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))

def _mixture_pdf(xgrid, w, m, s):
    pdf = np.zeros_like(xgrid, float)
    for wi, mi, si in zip(w, m, s):
        pdf += wi * (1.0/(si*np.sqrt(2*np.pi))) * np.exp(-0.5*((xgrid-mi)/si)**2)
    return pdf

def _mixture_cdf(x, w, m, s):
    cdf = np.zeros_like(x, float)
    for wi, mi, si in zip(w, m, s):
        cdf += wi * _normal_cdf(x, mi, si)
    return cdf

def _fd_bins(x, max_bins=60):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2: return 5
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0: return min(max_bins, max(5, int(np.sqrt(n))))
    h = 2 * iqr / (n ** (1/3))
    if h <= 0: return min(max_bins, max(5, int(np.sqrt(n))))
    k = int(np.ceil((x.max() - x.min()) / h))
    return int(np.clip(k, 8, max_bins))

def _fit_gmm1d(x, k=2, rs=0):
    x = np.asarray(x).reshape(-1, 1)
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=rs, n_init=10, reg_covar=1e-6).fit(x)
    mus = gmm.means_.ravel()
    sig = np.sqrt(gmm.covariances_.reshape(-1))
    w = gmm.weights_.ravel()
    order = np.argsort(mus)
    return w[order], mus[order], sig[order]

def _clean(series, valid_range=None, clip_percentiles=None):
    x = pd.to_numeric(pd.Series(series), errors="coerce").dropna().values
    if x.size == 0: return x
    if valid_range is not None:
        lo, hi = valid_range
        x = x[(x >= lo) & (x <= hi)]
    if clip_percentiles is not None and x.size > 0:
        p1, p2 = np.percentile(x, clip_percentiles)
        x = x[(x >= p1) & (x <= p2)]
    return x

def plot_hist_gmm_clean(ax, data, xlabel, title_letter=None,
                        valid_range=None, clip_percentiles=None, bins=None):
    x = _clean(data, valid_range, clip_percentiles)
    if x.size == 0:
        ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); return

    nbins = _fd_bins(x) if bins is None else bins
    counts, edges = np.histogram(x, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n = x.size

    ax.scatter(centers, counts, marker="s", facecolors="none",
               edgecolors="#3A86FF", s=60, linewidths=1.5)

    try:
        if n < 5: raise RuntimeError("Too few points for GMM")
        w, m, s = _fit_gmm1d(x, k=2)
        grid = np.linspace(edges[0], edges[-1], 1000)

        y_mix = _mixture_pdf(grid, w, m, s) * n * width
        ax.plot(grid, y_mix, lw=3, color="#C2185B")

        for wi, mi, si in zip(w, m, s):
            y_comp = (wi * (1.0/(si*np.sqrt(2*np.pi))) *
                      np.exp(-0.5*((grid-mi)/si)**2)) * n * width
            ax.plot(grid, y_comp, lw=1, color="#888888", alpha=0.95)
            ax.text(mi, y_comp.max()*1.05, f"{mi:.1f}±{si:.1f}",
                    ha="center", va="bottom", fontsize=10)

        pred = n * (_mixture_cdf(edges[1:], w, m, s) -
                    _mixture_cdf(edges[:-1], w, m, s))
        ss_res = np.sum((counts - pred) ** 2)
        ss_tot = np.sum((counts - counts.mean()) ** 2) if counts.sum() > 0 else np.nan
        r2 = float("nan") if not np.isfinite(ss_tot) or ss_tot == 0 else 1 - ss_res/ss_tot
        ax.text(0.02, 0.92, f"$R^2$ = {r2:.4f}", transform=ax.transAxes,
                fontsize=11, ha="left", va="top")
    except Exception as e:
        ax.text(0.02, 0.92, f"GMM error: {e}", transform=ax.transAxes,
                fontsize=10, ha="left", va="top")

    ax.set_xlabel(xlabel); ax.set_ylabel("Frequency count")
    if title_letter: ax.set_title(title_letter, loc="left", fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(edges[0], edges[-1]); ax.margins(x=0.02)

# Create imgs directory if it doesn't exist
os.makedirs('imgs', exist_ok=True)

# Load the CSV file with the correct structure
csv_path = "chem_reference.csv"  # Update this path as needed
df = pd.read_csv(csv_path, skiprows=1)  # Skip the first row with the description

# The columns we need are:
# - "First discharge capacity, mAh/g" (column 1)
# - "Initial Coulombic efficiency, %" (column 3)

# Print column names to verify structure
print("Available columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Extract the data from the correct columns
first_discharge_capacity = df.iloc[:, 1]  # "First discharge capacity, mAh/g"
coulombic_efficiency = df.iloc[:, 3]  # "Initial Coulombic efficiency, %"

# Plot 1: First discharge capacity
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
plot_hist_gmm_clean(
    ax,
    first_discharge_capacity,
    xlabel="Capacity (mAh g$^{-1}$)",
    title_letter="a",
    valid_range=(160, 240),
    clip_percentiles=(1, 99),
)
plt.tight_layout()
plt.savefig('imgs/discharge_capacity_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Initial Coulombic efficiency
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
plot_hist_gmm_clean(
    ax,
    coulombic_efficiency,
    xlabel="Coulombic efficiency (%)",
    title_letter="b",
    valid_range=(70, 100),
    clip_percentiles=(1, 99),
)
plt.tight_layout()
plt.savefig('imgs/coulombic_efficiency_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

print("Histograms saved to imgs/ directory")

