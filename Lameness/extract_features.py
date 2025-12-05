# src/extract_features.py
import pandas as pd
import numpy as np
import glob
import os

# === CONFIG ===
DLC_CSV_FOLDER = r"C:\Users\kulso\OneDrive\Desktop\Project\DLC_PROJECT\Classificationnn\csvfolder"   
LABELS_CSV = r"C:\Users\kulso\OneDrive\Desktop\Project\DLC_PROJECT\Classificationnn\labels.csv"     
OUTPUT_FEATURES = r"C:\Users\kulso\OneDrive\Desktop\Project\DLC_PROJECT\Classificationnn\features\all_video_features.csv"

LIKELIHOOD_THRESHOLD = 0.6

def load_dlc_csv(path):
    # DLC CSV often has 3-row header. If not, tweak header arg.
    df = pd.read_csv(path, header=[0,1,2])
    # flatten columns
    df.columns = [f"{bp}_{coord}" for scorer, bp, coord in df.columns]
    # drop unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def get_bodyparts(df):
    # detect bodyparts by x columns (col endswith '_x')
    x_cols = [c for c in df.columns if c.endswith('_x')]
    bps = [c[:-2] for c in x_cols]  # drop trailing '_x'
    return bps

def clean_and_smooth(df, threshold=0.6):
    bps = get_bodyparts(df)
    coord_cols = []
    for bp in bps:
        xcol, ycol, lcol = f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"
        coord_cols += [xcol, ycol]
        if lcol in df.columns:
            df.loc[df[lcol] < threshold, [xcol, ycol]] = np.nan
    # linear interpolate then forward/back fill
    df[coord_cols] = df[coord_cols].interpolate(method='linear', axis=0, limit_direction='both')
    df[coord_cols] = df[coord_cols].fillna(method='bfill').fillna(method='ffill')
    # smooth with rolling mean
    df[coord_cols] = df[coord_cols].rolling(window=5, min_periods=1, center=True).mean()
    return df

def pairwise_distance(df, bp1, bp2):
    x1, y1 = df[f"{bp1}_x"].values, df[f"{bp1}_y"].values
    x2, y2 = df[f"{bp2}_x"].values, df[f"{bp2}_y"].values
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def angle_at_b(df, a, b, c):
    Ax, Ay = df[f"{a}_x"].values, df[f"{a}_y"].values
    Bx, By = df[f"{b}_x"].values, df[f"{b}_y"].values
    Cx, Cy = df[f"{c}_x"].values, df[f"{c}_y"].values
    BAx, BAy = Ax - Bx, Ay - By
    BCx, BCy = Cx - Bx, Cy - By
    dot = BAx*BCx + BAy*BCy
    normBA = np.sqrt(BAx**2 + BAy**2)
    normBC = np.sqrt(BCx**2 + BCy**2)
    cos = dot / (normBA*normBC + 1e-8)
    ang = np.arccos(np.clip(cos, -1, 1))
    return ang  # radians

def speed(df, bp):
    x = df[f"{bp}_x"].values
    y = df[f"{bp}_y"].values
    vx = np.gradient(x)    # pixel per frame
    vy = np.gradient(y)
    speed = np.sqrt(vx**2 + vy**2)
    return speed

def summarize(ts, prefix):
    return {
        f"{prefix}_mean": np.nanmean(ts),
        f"{prefix}_std":  np.nanstd(ts),
        f"{prefix}_min":  np.nanmin(ts),
        f"{prefix}_max":  np.nanmax(ts),
        f"{prefix}_median": np.nanmedian(ts),
        f"{prefix}_p25":  np.percentile(ts,25),
        f"{prefix}_p75":  np.percentile(ts,75),
    }

def extract_features_for_video(csv_path):
    df = load_dlc_csv(csv_path)
    df = clean_and_smooth(df)

    bps = get_bodyparts(df)
    feats = {}

    # Spine & neck distances
    if {"mouth", "head"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "mouth", "head"), "mouth_head_dist"))

    if {"head", "withers"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "head", "withers"), "head_withers_dist"))

    if {"withers", "midback"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "withers", "midback"), "withers_midback_dist"))

    if {"midback", "tailbone"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "midback", "tailbone"), "midback_tailbone_dist"))

    # Front legs
    if {"withers", "F_knee1"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "withers", "F_knee1"), "withers_fknee1_dist"))

    if {"F_knee1", "F_hoof1"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "F_knee1", "F_hoof1"), "fknee1_fhoof1_dist"))

    if {"withers", "F_knee2"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "withers", "F_knee2"), "withers_fknee2_dist"))

    if {"F_knee2", "F_hoof2"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "F_knee2", "F_hoof2"), "fknee2_fhoof2_dist"))

    # Back legs
    if {"tailbone", "B_knee1"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "tailbone", "B_knee1"), "tailbone_bknee1_dist"))

    if {"B_knee1", "B_hoof1"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "B_knee1", "B_hoof1"), "bknee1_bhoof1_dist"))

    if {"tailbone", "B_knee2"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "tailbone", "B_knee2"), "tailbone_bknee2_dist"))

    if {"B_knee2", "B_hoof2"}.issubset(bps):
        feats.update(summarize(pairwise_distance(df, "B_knee2", "B_hoof2"), "bknee2_bhoof2_dist"))

    # Angles
    feats.update(summarize(angle_at_b(df, "mouth", "head", "withers"), "neck_angle"))
    feats.update(summarize(angle_at_b(df, "head", "withers", "midback"), "shoulder_angle"))
    feats.update(summarize(angle_at_b(df, "withers", "midback", "tailbone"), "spine_angle"))

    feats.update(summarize(angle_at_b(df, "withers", "F_knee1", "F_hoof1"), "front_left_knee_angle"))
    feats.update(summarize(angle_at_b(df, "withers", "F_knee2", "F_hoof2"), "front_right_knee_angle"))

    feats.update(summarize(angle_at_b(df, "tailbone", "B_knee1", "B_hoof1"), "back_left_knee_angle"))
    feats.update(summarize(angle_at_b(df, "tailbone", "B_knee2", "B_hoof2"), "back_right_knee_angle"))

    # Speeds
    for bp in ["mouth", "head", "withers", "midback", "tailbone", 
               "F_hoof1", "F_hoof2", "B_hoof1", "B_hoof2"]:
        if bp in bps:
            feats.update(summarize(speed(df, bp), f"{bp}_speed"))

    return feats


def main():
    csv_files = glob.glob(os.path.join(DLC_CSV_FOLDER, "*.csv"))
    labels_df = pd.read_csv(LABELS_CSV)  # video,label

    all_features = []
    for csv_path in csv_files:
        video_name = os.path.basename(csv_path).replace(".csv", "")
        feats = extract_features_for_video(csv_path)
        feats["video"] = video_name

        # Attach label
        label_row = labels_df[labels_df["video"] == video_name]
        if len(label_row) == 0:
            print(f"⚠ No label for {video_name}, skipping.")
            continue
        feats["label"] = label_row["label"].values[0]

        all_features.append(feats)

    out_df = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    out_df.to_csv(OUTPUT_FEATURES, index=False)
    print(f" Features saved to {OUTPUT_FEATURES}")

if __name__ == "__main__":
    main()