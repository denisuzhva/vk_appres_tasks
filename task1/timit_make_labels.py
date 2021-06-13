# timit_make_labels.py
# Merge TRAIN and TEST partitions in the TIMIT dataset 

import os
import pandas as pd



DATASET_PATH = "datasets/TIMIT/"
tt = ["TRAIN", "TEST"]
sentence_paths = []
labels = []
lab = 0

for t in tt:
    t_path = DATASET_PATH + t + '/'
    for reg in os.listdir(t_path):
        reg_path = t_path + reg + '/'
        for sid in os.listdir(reg_path):
            sid_path = reg_path + sid + '/'
            for file in os.listdir(sid_path):
                if file.endswith(".WAV"):
                    sentence_paths.append(sid_path + file)
                    labels.append(lab)
            lab += 1

df = pd.DataFrame({"path": sentence_paths, "label": labels})
#df.index.name = "idx"
df.to_csv(DATASET_PATH + "items_labeled.csv")