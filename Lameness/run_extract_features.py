import pandas as pd
import os
from extract_features import extract_features_for_video

# Paths
base_dir = r"C:\Users\kulso\OneDrive\Desktop\Project\DLC_PROJECT\Classificationnn"
csv_dir = os.path.join(base_dir, "csvfolder")
labels_path = os.path.join(base_dir, "labels.csv")
output_path = os.path.join(base_dir, "features.csv")

# Read labels CSV
labels_df = pd.read_csv(labels_path)
all_features = []

# Loop through each video CSV
for _, row in labels_df.iterrows():
    file_path = os.path.join(csv_dir, row['video'])
    feats = extract_features_for_video(file_path)
    feats['label'] = row['label']
    feats['video'] = row['video']
    all_features.append(feats)

# Save all extracted features
features_df = pd.DataFrame(all_features)
features_df.to_csv(output_path, index=False)

print(f" Features saved to {output_path}")
