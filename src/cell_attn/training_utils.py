""" Train and test functions for attention networks """

import os
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import pandas as pd
from datetime import datetime

def train_loop(model, training_dataloader, test_dataloader, num_epochs, burn_in, patience, loss_fn, learning_rate, min_learning_rate,
               stloc, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    """ Training loop. Called on by train_model.train_deep_attention_network().

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Deep attention network model.
        training_dataloader: torch.utils.data.DataLoader
            DataLoader containing data to train on.
        validation_dataloader: torch.utils.data.DataLoader
            DataLoader containing data to validate on.
        num_epochs: int
            Maximum number of epochs to train for. 
        burn_in: int
            Number of epochs to wait before beginning counting for early stopping.
        patience: int
            Number of epochs to allow a lack of improvement for before terminating training. 
            To disable early stopping set this equal to num_epochs.
        loss_fn: 
            Loss function to use for training.
        learning_rate: float
            Initial learning rate.
        min_learning_rate: float
            Minimum learning rate to anneal to.
        stloc: str
            Directory to store trained model and training metrics in.
        device: torch.device
            Device that model and DataLoader live on.

        """
    
    train_losses = []
    test_losses = []
    test_losses_per_epoch = []

    def add_to_log(text):
        with open(os.path.join(stloc, 'training_log.txt'), 'a') as f:
            f.write(text + '\n')

    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimiser, T_max=np.max([60,num_epochs]), eta_min=min_learning_rate, last_epoch=-1)
    model.train()

    num_epochs_without_improvement = 0
    start = datetime.now()
    for epoch in range(num_epochs):
        print(f'---------- Epoch {epoch} ----------')
        add_to_log(f'---------- Epoch {epoch} ----------')
            
        for batch, (X, y) in enumerate(training_dataloader):
            predictions = model(X).to(device)
            loss = loss_fn(predictions, y)
            
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            if batch % 100 == 0:
                train_losses.append(loss.item())
                
                X_test, y_test = next(iter(test_dataloader))
                with torch.no_grad():
                    predictions = model(X_test).to(device)
                    loss = loss_fn(predictions, y_test).item()
                    test_losses.append(loss)

        scheduler.step()

        pd.DataFrame({'train_loss':train_losses, 'test_loss':test_losses}).to_csv(os.path.join(stloc, 'losses.csv'), index=False)

        with torch.no_grad():
            loss = test_loop(model, test_dataloader, loss_fn)
        test_losses_per_epoch.append(loss)
        print(f'Test loss: {loss}')
        add_to_log(f'Test loss: {loss}')

        pd.DataFrame({'test_loss':test_losses_per_epoch}).to_csv(os.path.join(stloc, 'metrics_per_epoch.csv'), index=False)

        # Early stopping
        # Note that if condition will always activate at epoch 0
        if (epoch==0) | (loss == min(test_losses_per_epoch)):
            print('Saving model')
            add_to_log('Saving model')
            num_epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(stloc, 'model.pth'))
        elif epoch > burn_in:
            num_epochs_without_improvement += 1
            
        if num_epochs_without_improvement >= patience:
            print(f'Stopping due to lack of improvement over {patience} epochs')
            add_to_log(f'Stopping due to lack of improvement over {patience} epochs')
            break

    finish = datetime.now()
    print('Training finished :)')
    print('Training time:', finish - start)
    add_to_log('Training finshed :)')
    add_to_log('Training time: ' + str(finish - start))

# -------------------------------------------------------------

def test_loop(model, dataloader, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """ Test loop. Called on by train_model.train_deep_attention_network().

        Parameters
        --------------
        model: model_definition_utils.velocity_predictor
            Deep attention network model.
        dataloader: torch.utils.data.DataLoader
            DataLoader containing data to validate on.
        loss_fn: 
            Loss function to use for training.
        device: torch.device
            Device that model and DataLoader live on.

        """
    
    num_batches = len(dataloader)
    num_datapoints = len(dataloader.dataset)
    
    loss = 0
    num_correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            predictions = model(X).to(device)
            loss += loss_fn(predictions, y).item()
    
        loss /= num_batches

    return loss