import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import ViTBinaryClassifier
from torch_utils import set_seed, get_device, numpy_to_data_loader

from torch_utils import model_fit, classification_acc

# Define function to sample 10% of the data
def sample_data(x, y, fraction):
    # Determine sample size
    sample_size = int(len(x) * fraction)
    # Generate random indices
    indices = np.random.choice(len(x), sample_size, replace=False)
    # Sample the data
    x_sampled = x[indices]
    y_sampled = y[indices]
    return x_sampled, y_sampled

def main():
    # random_seed = 42
    # val_size = 0.2
    norm_factors = np.array([92, 49, 288])
    sub_len = 100
    adam_lr=0.0001
    EPOCHS = 2
    train_batch_size = 16
    val_batch_size = 128
    fraction = 0.5
    # model_size = "base"
    model_size = "large"

    device = get_device()

    # Load pre-processed data
    x_train, y_train = pickle.load(open('whole_day_with_status/train_siamese_whole_day_with_status_halfyear_100_20000.pkl', 'rb'))
    x_val, y_val = pickle.load(open('whole_day_with_status/val_siamese_whole_day_with_status_halfyear_2000_2100_10000.pkl', 'rb'))

    # Sample 10% of the training and validation data
    x_train, y_train = sample_data(x_train, y_train, fraction)
    x_val, y_val = sample_data(x_val, y_val, fraction)

    # normalize the data
    x_train[:, :, :, :3] /= norm_factors
    x_val[:, :, :, :3] /= norm_factors

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    
    sub_len = 100
    whole_len = x_train.shape[2]
    num_sub = int(whole_len / sub_len)

    # Number of pairs for driver 1: num_subtrajectories*(num_subtrajectories-1)/2
    num_pairs_d1 = int((num_sub * (num_sub - 1)) / 2)
    # Number of pairs for driver 1 and driver 2: num_subtrajectories * num_subtrajectories
    num_pairs_d2 = num_sub * num_sub
    # number of pairs for driver 2: num_subtrajectories * (num_subtrajectories-1)
    num_pairs_d3 = int((num_sub * (num_sub - 1)) / 2)

    # Create labels for pairs from driver 1
    y_train_d1 = np.ones((y_train.shape[0], num_pairs_d1))
    y_val_d1 = np.ones((y_val.shape[0], num_pairs_d1))
    print(y_train_d1.shape, y_val_d1.shape)

    # Create labels for pairs from different drivers (driver 1 and driver 2)
    y_train_d2 = np.zeros((y_train.shape[0], num_pairs_d2))
    y_val_d2 = np.zeros((y_val.shape[0], num_pairs_d2))
    print(y_train_d2.shape, y_val_d2.shape)

    # Create labels for pairs from driver 2
    y_train_d3 = np.ones((y_train.shape[0], num_pairs_d3))
    y_val_d3 = np.ones((y_val.shape[0], num_pairs_d3))
    print(y_train_d3.shape, y_val_d3.shape)

    # Concatenate the labels for driver 1 and driver 2 pairs
    y_train = np.concatenate((y_train_d1, y_train_d2, y_train_d3), axis=1)
    y_val = np.concatenate((y_val_d1, y_val_d2, y_val_d3), axis=1)
    print(y_train.shape, y_val.shape)

    # prepare data
    train_loader = numpy_to_data_loader(
        x_train, y_train, y_dtype=torch.float32, batch_size=train_batch_size, shuffle=True
    )
    val_loader = numpy_to_data_loader(
        x_val, y_val, y_dtype=torch.float32, batch_size=val_batch_size, shuffle=False
    )

    model = ViTBinaryClassifier(input_size=x_train.shape[-1])

    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs: ", num_gpus)

    if num_gpus > 1:
        # Wrap the model with nn.DataParallel
        model = nn.DataParallel(model)
    
    # Train model
    model.to(device)
    loss_fn = nn.BCELoss()
    acc_fn = classification_acc("binary")
    optimizer = optim.Adam(model.parameters(), lr=adam_lr)

    history = model_fit(
        model,
        loss_fn,
        acc_fn,
        optimizer,
        train_loader,
        epochs=EPOCHS,
        val_loader=val_loader,
        save_best_only=True,
        early_stopping=2,
        save_every_epoch=True,
        save_path=f'pretrainmodels/{model_size}_model_ViT_Siamese_with_status_halfyear_{int(20000*fraction)}_{EPOCHS}_epochs.pt',
        device=device,
    )

    # train_loss = history['train_loss']
    # val_loss = history['val_loss']
    # train_acc = history['train_acc']
    # val_acc = history['val_acc']

    # caluculate total parameters in model
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    # EPOCHS = len(train_loss)


    # # Draw figure to plot the loss and accuracy
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(EPOCHS), train_loss, label='train')
    # plt.plot(range(EPOCHS), val_loss, label='val')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(range(EPOCHS), train_acc, label='train')
    # plt.plot(range(EPOCHS), val_acc, label='val')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.savefig(f'{model_size}_model_ViT_Siamese_with_status_halfyear_{20000*fraction}_{EPOCHS}epochs.png')

if __name__ == '__main__':
    # set seed
    set_seed(0)

    # Run main
    main()

