"""Project 5 - Smart Factory Machine Maintenance Analysis (compact: all 15 tasks, NumPy + pandas only). Associations, not causation.
Run from the repo root:  python src/machine_maintenance_analysis_compact.py   (reads a CSV; edit PATH to use another file; every result is saved as compact_*.csv)"""
import numpy as np
import pandas as pd

PATH = "05_machine_maintenance_1500 (1).csv"  # dataset location, relative to the folder you run from
SENS = ["Operating_Hours", "Temperature_C", "Vibration_mm_s", "Rotational_Speed_RPM", "Tool_Wear_Minutes"]
HIGH_TEMP = 90.0  # task 12 cut-off (about the 95th percentile)
RULES = {"Temperature_C": 85, "Vibration_mm_s": 6.0, "Tool_Wear_Minutes": 270,  # task 13: about the top 10 % of each sensor,
         "Operating_Hours": 8800, "Rotational_Speed_RPM": 1750}                  # set from its own distribution
sheets = {}
def show(name, table, note=""):  # print a result and keep it to save as a CSV (one file per task)
    table = table.to_frame() if isinstance(table, pd.Series) else table
    print(f"\n=== {name} {note}\n{table.to_string()}")
    sheets[name] = table
def yes_pct(s):
    return round((s == "Yes").mean() * 100, 2)
rng = np.random.default_rng(42)
def perm_p(values, groups, n=2000):  # NumPy-only permutation test: is the spread of group means bigger than chance?
    codes, v = pd.factorize(groups)[0], np.asarray(values, float)
    def spread(x):
        m = np.bincount(codes, x) / np.bincount(codes)
        return m.max() - m.min()
    return (1 + sum(spread(rng.permutation(v)) >= spread(v) for _ in range(n))) / (n + 1)
# 1. Load and inspect (keep_default_na=False so the text "None" in Failure_Type is not silently turned into NaN)
raw = pd.read_csv(PATH, keep_default_na=False, na_values=[""])
show("01_Inspection", raw.dtypes.astype(str).to_frame("dtype").assign(non_null=raw.notna().sum(), unique=raw.nunique()),
     f"shape={raw.shape}")
# 2. Missing readings: blank vibration cells -> median (robust, ignores the label); "None" failure type is a real category
df = raw.copy()
df[SENS] = df[SENS].apply(pd.to_numeric, errors="coerce")
df["Failure_Type"] = df["Failure_Type"].replace({"None": "No_Failure"})
df["Source_Row"] = df.index + 2  # file row number, header = row 1 (the file has no record ID)
obs = df["Vibration_mm_s"].copy()
df["Vibration_Imputed"] = obs.isna()
df["Vibration_mm_s"] = obs.fillna(obs.median())
show("02_Missing", raw.isna().sum().to_frame("blank_cells").assign(text_None=raw.astype(str).eq("None").sum()),
     f"-> {obs.isna().sum()} vibration blanks filled with median {obs.median():.2f}")
# 3. Duplicates and machine identifiers
ids = df["Machine_ID"]
show("03_Duplicates_IDs", pd.Series({"duplicate rows": raw.duplicated().sum(), "duplicates on 5 sensors": df.duplicated(SENS).sum(),
     "distinct Machine_IDs": ids.nunique(), "IDs not matching M##": (~ids.str.match(r"^M\d{2}$")).sum(),
     "IDs with stray spaces": (ids != ids.str.strip()).sum()}))
# 4-5. Sensor summary and maintenance counts
show("04_Sensor_Summary", df[SENS].agg(["mean", "median", "std"]).T.round(3))
mc = df["Maintenance_Required"].value_counts(); show("05_Maintenance_Counts", pd.DataFrame({"count": mc, "percent": (mc / len(df) * 100).round(2)}))
# 6-7. Temperature and vibration by maintenance status
g = df.groupby("Maintenance_Required")
flag = (df.Maintenance_Required == "Yes").astype(int)  # 1 = maintenance required
show("06_Temp_by_Status", g.mean(numeric_only=True).drop(columns=["Source_Row", "Vibration_Imputed"], errors="ignore").round(3))
yes, no = (df.loc[df.Maintenance_Required == k] for k in ("Yes", "No"))
sd = np.sqrt(((len(yes) - 1) * yes.Vibration_mm_s.var() + (len(no) - 1) * no.Vibration_mm_s.var()) / (len(df) - 2))
d = (yes.Vibration_mm_s.mean() - no.Vibration_mm_s.mean()) / sd  # Cohen's d (below 0.2 = negligible)
p_temp, p_vib = perm_p(df.Temperature_C, df.Maintenance_Required), perm_p(df.Vibration_mm_s, df.Maintenance_Required)
show("07_Vibration_Compare", g["Vibration_mm_s"].agg(["count", "mean", "median", "std"]).round(3),
     f"Cohen d={d:.2f}, permutation p={p_vib:.3f} (temperature p={p_temp:.3f})")
# 8. Machines ranked by average operating hours (rank 1 = highest)
rank = df.groupby("Machine_ID").agg(avg_hours=("Operating_Hours", "mean"), records=("Machine_ID", "size"),
                                    maint_pct=("Maintenance_Required", yes_pct)).round(1)
rank = rank.assign(rank=rank["avg_hours"].rank(ascending=False, method="min").astype(int)).sort_values("rank")
show("08_Machine_Ranking", rank)
# 9. Unusually high vibration by the IQR rule (quartiles from observed, non-imputed readings)
q1, q3 = obs.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
cols = ["Source_Row", "Machine_ID"] + SENS + ["Maintenance_Required", "Failure_Type"]
flagged = df.loc[df.Vibration_mm_s > upper, cols].sort_values("Vibration_mm_s", ascending=False)
show("09_IQR_High_Vibration", flagged, f"(fences {q1 - 1.5 * (q3 - q1):.2f} / {upper:.2f} mm/s)")
# 10-11. Failure_Type groups and average tool wear
ft = df.groupby("Failure_Type").agg(records=("Machine_ID", "size"), maint_pct=("Maintenance_Required", yes_pct),
                                    avg_temp=("Temperature_C", "mean"), avg_tool_wear=("Tool_Wear_Minutes", "mean")).round(2)
ft["percent"] = (ft["records"] / len(df) * 100).round(2)
ft = ft.sort_values("records", ascending=False)
show("10_FailureType_Groups", ft, f"(permutation p vs maintenance={perm_p(flag, df.Failure_Type):.2f})")
show("11_ToolWear_by_Failure", ft[["records", "avg_tool_wear"]].sort_values("avg_tool_wear", ascending=False), f"(permutation p={perm_p(df.Tool_Wear_Minutes, df.Failure_Type):.2f})")
# 12. NumPy Boolean indexing: high-temperature records
temp = df["Temperature_C"].to_numpy()
mask = temp >= HIGH_TEMP  # Boolean array
hot = df.loc[mask, cols].sort_values("Temperature_C", ascending=False)
show("12_HighTemp_Records", hot, f"{mask.sum()} records ({mask.mean() * 100:.1f} %), mean {temp[mask].mean():.1f} C, "
     f"maintenance {yes_pct(hot.Maintenance_Required)} % vs {yes_pct(df.Maintenance_Required)} % overall")
# 13. Risk band from documented thresholds (score = thresholds met: 0 Low, 1 Medium, 2+ High)
df["Risk_Score"] = sum(np.where(df[c] >= t, 1, 0) for c, t in RULES.items())
df["Risk_Band"] = np.where(df.Risk_Score >= 2, "High", np.where(df.Risk_Score == 1, "Medium", "Low"))
band = df.groupby("Risk_Band").agg(records=("Machine_ID", "size"), maint_pct=("Maintenance_Required", yes_pct)).reindex(["Low", "Medium", "High"])
show("13_Risk_Band", band, f"{RULES}; permutation p={perm_p(flag, df.Risk_Band):.2f}")
# 14. Correlations among sensors, and each sensor against the maintenance flag
corr = df[SENS].corr().round(3)
show("14_Correlations", corr, f"strongest pair |r|={np.abs(corr.values[np.triu_indices(5, 1)]).max():.3f}; "
     f"each sensor vs maintenance flag: max |r|={df[SENS].corrwith(flag).abs().max():.3f}")
# 15. Recommendations (all numbers come from the results above)
p_mach = perm_p(flag, df.Machine_ID)
has_ft = df.Failure_Type != "No_Failure"
watch = df.groupby("Machine_ID")["Risk_Band"].apply(lambda s: (s == "High").mean() * 100).nlargest(3).round(1)
recs = [
    f"1. Keep schedule-based inspections; use thresholds as prompts, not triggers: temperature {yes.Temperature_C.mean():.1f} vs "
    f"{no.Temperature_C.mean():.1f} C, vibration d={d:.2f}, High band {band.maint_pct['High']}% vs Low {band.maint_pct['Low']}% maintenance.",
    f"2. Verify extreme records in person: {len(flagged)} above {upper:.2f} mm/s ({(flagged.Maintenance_Required == 'Yes').sum()} already Yes), {mask.sum()} at >= {HIGH_TEMP:g} C.",
    f"3. Standardise labels: {(has_ft & (df.Maintenance_Required == 'No')).sum()} of {has_ft.sum()} failure-typed records are 'No' maintenance; "
    f"{(~has_ft & (df.Maintenance_Required == 'Yes')).sum()} of {len(yes)} maintenance records have no failure type.",
    f"4. Fix logging gaps ({obs.isna().sum()} blank vibration readings on {df.loc[df.Vibration_Imputed, 'Machine_ID'].nunique()} machines); "
    "add timestamp, record ID, hours since service, load, maintenance type.",
    f"5. Do not re-allocate by operating-hours rank ({rank.avg_hours.min():,.0f}-{rank.avg_hours.max():,.0f} h; machine maintenance rates differ at p={p_mach:.2f}); "
    "watch-list by High-risk share: " + ", ".join(f"{m} ({v}%)" for m, v in watch.items()) + ".",
]
print("\n=== 15_Recommendations\n" + "\n".join(recs)); sheets["15_Recommendations"] = pd.DataFrame({"Recommendation": recs})
# Outputs: one CSV per task table, plus the cleaned data
for name, table in sheets.items():
    table.to_csv(f"compact_{name}.csv")
df.to_csv("compact_cleaned_machine_data.csv", index=False)
print("\nDone. Files saved as compact_* in the current folder.")
