# timit_make_labels_multi.py
# Prepare two sets of labels: one for each speaker id and the other for each speaker gender (male or female)

import os
import pandas as pd



DATASET_PATH = "datasets/TIMIT/" # Assuming TRAIN and TEST directories are in DATASET_PATH
TRAIN_ONLY = False
if TRAIN_ONLY:
    tt = ["TRAIN"]
else:
    tt = ["TRAIN", "TEST"]
sentence_paths = []
labels_multi = [] # for each speaker id
labels_binary = [] # for each speaker gender (male or female)
label_multi = 0

for t in tt:
    t_path = DATASET_PATH + t + '/'
    for reg in os.listdir(t_path):
        reg_path = t_path + reg + '/'
        for sid in os.listdir(reg_path):
            sid_path = reg_path + sid + '/'
            for file in os.listdir(sid_path):
                if file.endswith(".WAV"):
                    sentence_paths.append(sid_path + file)
                    labels_multi.append(label_multi)
                    if sid[0] == 'M':
                        labels_binary.append(0)
                    elif sid[0] == 'F':
                        labels_binary.append(1)
            label_multi += 1

df_multi = pd.DataFrame({"path": sentence_paths, "label": labels_multi})
df_binary = pd.DataFrame({"path": sentence_paths, "label": labels_binary})
df_multi.to_csv(DATASET_PATH + f"items_labeled_{label_multi}.csv")
df_binary.to_csv(DATASET_PATH + "items_labeled_2.csv")