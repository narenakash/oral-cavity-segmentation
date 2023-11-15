import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

data_csv = pd.read_csv("data/opmd_dataset.csv")

train_df, test_df = train_test_split(data_csv, test_size=0.2, random_state=42, stratify=data_csv["mask_label"])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["mask_label"])

train_df.to_csv("data/opmd_train.csv", index=False)
val_df.to_csv("data/opmd_val.csv", index=False)
test_df.to_csv("data/opmd_test.csv", index=False)