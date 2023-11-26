import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


data_csv = pd.read_csv("../data/opmd_dataset.csv")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for index, (train_index, test_index) in enumerate(skf.split(data_csv, data_csv["mask_label"])):

    train_df = data_csv.iloc[train_index]
    test_df = data_csv.iloc[test_index]

    #split train into train and val
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["mask_label"])

    train_df.to_csv(f"../data/opmd_train_fold{index+1}.csv", index=False)
    val_df.to_csv(f"../data/opmd_val_fold{index+1}.csv", index=False)
    test_df.to_csv(f"../data/opmd_test_fold{index+1}.csv", index=False)