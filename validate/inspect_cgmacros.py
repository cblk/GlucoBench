import pandas as pd
from pathlib import Path

p = Path("output/cgmacros_subset/subjects")
files = sorted(p.glob("*.csv"))

print("Total files:", len(files))
for f in files[:8]:
    df = pd.read_csv(f)
    ts = pd.to_datetime(df["Timestamp"])
    gaps = ts.diff().dt.total_seconds() / 60.0
    dex_nulls = df["Dexcom GL"].isna().sum()
    lib_nulls = df["Libre GL"].isna().sum()
    span = (ts.max() - ts.min()).total_seconds() / 86400.0
    print(f"{f.name}: rows={len(df)}, span={span:.1f}d, gap_mode={gaps.mode()[0]}m, Dexcom nulls={dex_nulls}, Libre nulls={lib_nulls}")
