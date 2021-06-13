# main.py

import torch
from model import SincNet
from trainer import train_model
from cfg.settings import *



if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    model = SincNet(SAMPLE_CH, SAMPLE_LEN, CONV_CHANNELS,
                    CONV_KERNEL_SIZES, POOL_KERNEL_SIZES,
                    FC_SIZES, N_CLASSES,
                    DROPOUT)

    model.to(device)

    train_model(DF_PATH, model, N_CLASSES, SAMPLE_LEN, SAMPLE_CH, BATCH_SIZE, N_EPOCH, LEARNING_RATE, device)

    