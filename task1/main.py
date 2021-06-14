# main.py

import os
import torch
from model import SincNet
from trainer import train_model
from cfg.settings import *



if __name__=="__main__":

    # Get CUDA device if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set default float type
    torch.set_default_dtype(torch.float32)

    # Make log and dump directories
    for path in [LOG_DF_PATH, MODEL_DUMP_PATH]:
        os.makedirs(os.path.split(path)[0], exist_ok=True)

    # Define the model
    model = SincNet(SAMPLE_CH, SAMPLE_LEN, CONV_N_FILTERS,
                    CONV_KERNEL_SIZES, POOL_KERNEL_SIZES,
                    FC_SIZES, N_CLASSES,
                    DROPOUT, LRELU_SLOPE, DO_SINCCONV)
    model.to(device)

    # Train the model
    train_model(DF_PATH, model, 
                N_CLASSES, SAMPLE_LEN, SAMPLE_CH, 
                BATCH_SIZE, N_EPOCH, LEARNING_RATE, 
                device, LOG_DF_PATH, MODEL_DUMP_PATH,
                VALIDATE_EACH_N_EPOCH)

    