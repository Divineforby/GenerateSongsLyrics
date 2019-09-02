from models import GenLSTM
from dataloader import SongData, Pad, onehot, data_split
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import os
import pickle as pkl
from configs import cfg

# Make new folder to keep sessions checkpoints
def makeSessionDir():

    # Get all the indices of the sessions
    sessionIndices = [int(name[-1]) for name in os.listdir(path='model_checkpoints/') if 'training_session' in name]

    # If none exists we make the index 0
    if ( not sessionIndices ):
        newIndex = 0
    else:
        newIndex = max(sessionIndices) + 1

    # Make the new folder and return the path
    path = './model_checkpoints/training_session' + str(newIndex)
    os.mkdir(path)

    return path

# Calculate Validation Loss
def validation(model, val_set):
    # No need to calculate gradients
    with torch.no_grad():
        LossFcn = torch.nn.CrossEntropyLoss()
        # Config
        batchSize = cfg['validation_batch_size']

        # Validation set data loader
        validation_loader = DataLoader(val_set, batch_size=batchSize, collate_fn=Pad)

        # Validation loss
        val_loss = 0

        # Go into evaluation mode
        model.eval()

        # Calculate loss w.r.t the entire validation set
        for idx, (data, labels) in enumerate(validation_loader):

            # One_hot
            data = onehot(data, len(SongData.vocab))

            # Push data and labels onto device
            data, labels = data.to(device), labels.to(device)

            # Run the data through the model
            Output, hc = model(data)

            # Reshape output and loss to interpret timesteps as another sample
            Output, labels = Output.view(Output.shape[0]*Output.shape[1], -1), labels.view(labels.shape[0]*labels.shape[1])

            # Compute Loss and add to average
            Loss = LossFcn(Output, labels)
            val_loss += float(Loss.item())

        # Average loss over entire set
        val_loss /= idx

    return val_loss

# Takes a fresh model and data set
# Train the model and save checkpoints
def train(model, train_set, val_set):

    # Training options
    maxEpochs = np.arange(cfg['max_epochs'])
    lr = cfg['lr']
    l2_decay = cfg['l2_penalty']
    batchSize = cfg['batch_size']

    # Send model to chose device
    model = model.to(device)

    # Create training set data loader, loss, and optimizer
    train_loader = DataLoader(train_set, batch_size=batchSize, collate_fn=Pad, shuffle=True)
    LossFcn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr = lr, weight_decay=l2_decay)

    # Loss collection and model chkpt every Nth batch
    # Checkpoint every Mth batch
    N = 50
    M = len(train_loader)/4

    # Training mode
    model.train()

    # Make new dir to keep this training session
    checkpointPath = makeSessionDir()

    # Losses every Nth batch for the epoch
    epoch_losses = []
    epoch_val_losses = []

    # Early stopping boolean
    # If the values in the early_stopping_range is monotonically increasing
    early_stopping = False
    early_stopping_range = 4

    # For each epoch train the model on the data
    for e in maxEpochs:

        print("In epoch %d..." % e)

        batch_loss = 0

        # For each batch in the loader send all
        for idx, (data, labels) in enumerate(train_loader):

            # One_hot the inputs using the number of possible encoding values
            data = onehot(data, len(SongData.vocab))

            # Push data and labels onto device
            data, labels = data.to(device), labels.to(device)

            # Reset gradients
            optim.zero_grad()

            # Run the data through the model
            Output, hc = model(data)

            # Reshape output and loss to interpret timesteps as another sample
            Output, labels = Output.view(Output.shape[0]*Output.shape[1], -1), labels.view(labels.shape[0]*labels.shape[1])

            # Compute Loss and add to average
            Loss = LossFcn(Output, labels)
            batch_loss += float(Loss.item())

            # Take gradient and update
            Loss.backward()
            optim.step()
            
            # Checkpoint
            if ( idx % M == 0 and idx > 0 ):

                print("Checkpoint saving as epoch %d batch %d" % (e, idx), flush=True)
                # Checkpoint, save model and optimizer
                modelPath = os.path.join( checkpointPath, 'LSTMmodel_E_{0}_B_{1}.mdl'.format(e,idx) )
                optimPath = os.path.join( checkpointPath, 'LSTMoptim_E_{0}_B_{1}.mdl'.format(e,idx) )

                torch.save(model.state_dict(), modelPath )
                torch.save(optim.state_dict(), optimPath )


            # Calculate average batch losses every nth batch
            if ( idx % N == 0 and idx > 0  ):

                # Calculate average training loss of N batches
                avg_loss = batch_loss/idx
                batch_loss = 0
                epoch_losses.append(avg_loss)

                # Try to clear some unused variable and run validation
                # reset to training mode
                del Loss, Output, data, labels, hc
                val_loss = validation(model, val_set)
                model.train()

                epoch_val_losses.append(val_loss)

                print("Loss at batch %d of epoch %d is %f" % (idx, e, avg_loss), flush=True)
                print("Validation Loss at batch %d of epoch %d is %f" % (idx, e, val_loss), flush=True)

                # Write to log
                with open(os.path.join (checkpointPath, 'output.log' ), "w+") as logfile:
                    logfile.write("Epoch: {0} Batch: {1}\nTraining Loss: {2}\nValidation Loss: {3}\n\n"
                                  .format(e, idx, avg_loss, val_loss))
                                  
                # Check for early stopping with monotonicity
                # Only check when we have at least early_stopping_range values to check
                # Give at least one epoch before we check for early stopping
                if (e > 0 and 
                    len(epoch_val_losses[(early_stopping_range-1)::-1]) == early_stopping_range and 
                    monotonicIncr(epoch_val_losses[(early_stopping_range-1)::-1])):
                    
                    early_stopping = True            

            # If early stopping we stop going through the data set
            if (early_stopping):
                break

        # Break out of epoch loop
        if (early_stopping):
            break

    # Write out the loss arrays
    with open(os.path.join(checkpointPath, 'train_losses.pkl'), "wb+") as tlossfile:
        pkl.dump(epoch_losses, tlossfile, pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(checkpointPath, 'val_losses.pkl'), "wb+") as vlossfile:
        pkl.dump(epoch_losses, vlossfile, pkl.HIGHEST_PROTOCOL)

    modelPath = os.path.join( checkpointPath, 'LSTMmodel_Final.mdl'.format(e,idx) )
    torch.save(model.state_dict(), modelPath)

    print("Finished training for %d epochs..." % e, flush=True)

# Check whether the list is monotonically increasing
def monotonicIncr(lst):

    # Monotonic increasing means subsequent values are always larger
    # Differences are then always >= 0
    diff = np.diff(lst[::-1])    
    diff = (diff >= 0)
   
    # If all the differences are > 0 we are monotonically decreasing
    return diff.all()

# Entry point
if __name__ == "__main__":

    print("Reading data from CSV...", flush=True)

    # Read in data and construct our vocabulary so we can use song data
    data = pd.read_csv('songdata.csv')
    SongData.init_vocab(data)

    print("Dividing whole data into training and validation sets...", flush=True)

    # Split data to training and validation set
    train_set, val_set = data_split(data)

    # Make fresh model to train
    # Both input and output are size of possible characters
    # We are inputting characters at t and asking to predict character t+1
    model = GenLSTM(input_size = len(SongData.vocab), output_size = len(SongData.vocab) )

    print("Beginning model training...", flush=True)
    
    useCuda = cfg['use_cuda']
    # Check for cuda and set default compute device
    if ( torch.cuda.is_available() and useCuda ):
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("Using %s for compute..." % device.type, flush=True)

    # Train the model
    train(model, train_set, val_set)
