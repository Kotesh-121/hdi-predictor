"""
generate_data.py
----------------
Generates a synthetic but realistic country-level dataset for Human
Development Index (HDI) modeling, following the actual UNDP HDI formula:

    Life Expectancy Index (LEI) = (LE - 20) / (85 - 20)
    Mean Years of Schooling Index (MYSI) = MYS / 15
    Expected Years of Schooling Index (EYSI) = EYS / 18
    Education Index (EI) = (MYSI + EYSI) / 2
    Income Index (II) = (ln(GNIpc) - ln(100)) / (ln(75000) - ln(100))
    HDI = (LEI * EI * II) ** (1/3)              [geometric mean]

Countries are then bucketed into the four official UNDP tiers:
    Very High : HDI >= 0.800
    High      : 0.700 <= HDI < 0.800
    Medium    : 0.550 <= HDI < 0.700
    Low       : HDI < 0.550

The dataset mixes three "development archetypes" (developed, emerging,
developing) so the feature distributions look like real-world regional
clusters rather than pure random noise, then adds Gaussian noise for
realism.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_GROUP = 220  # ~660 total synthetic "countries"


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def make_group(n, le_mu, le_sigma, mys_mu, mys_sigma,
               eys_mu, eys_sigma, gni_mu, gni_sigma):
    life_expectancy = clip(np.random.normal(le_mu, le_sigma, n), 45, 85)
    mean_years_schooling = clip(np.random.normal(mys_mu, mys_sigma, n), 0.5, 15)
    expected_years_schooling = clip(np.random.normal(eys_mu, eys_sigma, n), 4, 20)
    # GNI is log-normally distributed in the real world
    gni_per_capita = clip(np.random.lognormal(np.log(gni_mu), gni_sigma, n), 400, 120000)
    return life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita


# --- Three broad archetypes to give the data realistic regional clustering ---
# 1. Advanced economies (e.g. Western Europe, North America, East Asia)
le1, mys1, eys1, gni1 = make_group(
    N_PER_GROUP, le_mu=80, le_sigma=2.5, mys_mu=12, mys_sigma=1.5,
    eys_mu=16.5, eys_sigma=1.2, gni_mu=42000, gni_sigma=0.35)

# 2. Emerging / middle-income economies (e.g. Latin America, SE Asia, E. Europe)
le2, mys2, eys2, gni2 = make_group(
    N_PER_GROUP, le_mu=72, le_sigma=3.5, mys_mu=8.5, mys_sigma=1.8,
    eys_mu=13, eys_sigma=1.8, gni_mu=12000, gni_sigma=0.45)

# 3. Developing / low-income economies (e.g. parts of Sub-Saharan Africa, S. Asia)
le3, mys3, eys3, gni3 = make_group(
    N_PER_GROUP, le_mu=59, le_sigma=5.0, mys_mu=4.5, mys_sigma=1.8,
    eys_mu=8.5, eys_sigma=2.2, gni_mu=2200, gni_sigma=0.55)

life_expectancy = np.concatenate([le1, le2, le3])
mean_years_schooling = np.concatenate([mys1, mys2, mys3])
expected_years_schooling = np.concatenate([eys1, eys2, eys3])
gni_per_capita = np.concatenate([gni1, gni2, gni3])

# Expected years of schooling must be >= mean years of schooling logically
expected_years_schooling = np.maximum(expected_years_schooling, mean_years_schooling + 1)
expected_years_schooling = clip(expected_years_schooling, 4, 20)

# --- Compute the official HDI sub-indices ---
LEI = (life_expectancy - 20) / (85 - 20)
MYSI = mean_years_schooling / 15
EYSI = expected_years_schooling / 18
EI = (MYSI + EYSI) / 2
II = (np.log(gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100))
II = clip(II, 0, 1)

HDI = (clip(LEI, 0, 1) * clip(EI, 0, 1) * II) ** (1 / 3)

# small measurement noise to avoid a perfectly deterministic target
HDI = clip(HDI + np.random.normal(0, 0.01, len(HDI)), 0, 1)


def classify(hdi):
    if hdi >= 0.800:
        return "Very High"
    elif hdi >= 0.700:
        return "High"
    elif hdi >= 0.550:
        return "Medium"
    else:
        return "Low"


hdi_category = np.array([classify(h) for h in HDI])

df = pd.DataFrame({
    "life_expectancy": np.round(life_expectancy, 1),
    "mean_years_schooling": np.round(mean_years_schooling, 2),
    "expected_years_schooling": np.round(expected_years_schooling, 2),
    "gni_per_capita": np.round(gni_per_capita, 0),
    "hdi_score": np.round(HDI, 4),
    "hdi_category": hdi_category,
})

# shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path = "/home/claude/hdi_project/data/hdi_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(df["hdi_category"].value_counts())
print(df.head())
