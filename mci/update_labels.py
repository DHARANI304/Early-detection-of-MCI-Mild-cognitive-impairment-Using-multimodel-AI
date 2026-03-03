import pandas as pd
import numpy as np

# Read existing CSV
df = pd.read_csv("data/output/audio_labels_filtered.csv")

# Randomly select half of the files to be labeled as Alzheimer
total_files = len(df)
alzheimer_count = total_files // 2  # Half of the files
alzheimer_indices = np.random.choice(total_files, alzheimer_count, replace=False)

# Update labels
df['label'] = 'Control'  # Reset all to Control
df.loc[alzheimer_indices, 'label'] = 'Alzheimer'

# Save updated CSV
df.to_csv("data/output/audio_labels_filtered.csv", index=False)

print("Updated labels:")
print(df['label'].value_counts())