# settings.py

CONF_N = 1

# Dataset path
N_CLASSES = 630
TRAIN_DATASET_NAME = "TIMIT"
DF_PATH = f"datasets/{TRAIN_DATASET_NAME}/items_labeled_{N_CLASSES}.csv"

# Trainer parameters
BATCH_SIZE = 256
VALID_SPLIT = .2
SHUFFLE_DATASET = True
N_EPOCH = 300
LEARNING_RATE = 0.001
VALIDATE_EACH_N_EPOCH = 5

# NN parameters
WAVEFORM_FS = 16000
SAMPLE_LEN = 2048
SAMPLE_CH = 16
CONV_N_FILTERS = [64, 32, 32]
CONV_KERNEL_SIZES = [251, 5, 5] 
POOL_KERNEL_SIZES = [3, 3, 3]
DROPOUT = [0, 0, 0]
FC_SIZES = [2048, 2048, 2048]
LRELU_SLOPE = 0.01
DO_SINCCONV = False

# Log and weight dump paths
if DO_SINCCONV:
    prefix = "SincNet"
else:
    prefix = "CNN"
LOG_DF_PATH = f"results/{CONF_N}_{prefix}_{TRAIN_DATASET_NAME}_{N_CLASSES}_log.csv"
MODEL_DUMP_PATH = f"models/{CONF_N}_{prefix}_{TRAIN_DATASET_NAME}_{N_CLASSES}_model.pth"

