from models import GenLSTM
from dataloader import SongData, Pad, onehot
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
def validation(model, val_set, device, LossFcn):
    # No need to calculate gradients
    with torch.no_grad():
        
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
            data = onehot(data, len(val_set.encode))

            # Push data and labels onto device
            data, labels = data.to(device), labels.to(device)

            # Run the data through the model
            Output = model(data)

            # Reshape output and loss to interpret timesteps as another sample
            Output, labels = Output.view(Output.shape[0]*Output.shape[1], -1), labels.view(labels.shape[0]*labels.shape[1])

            # Compute Loss and add to average 
            Loss = LossFcn(Output, labels) 
            val_loss += float(Loss.item())

        # Average loss over entire set
        val_loss /= len(validation_loader)
    
    return val_loss

# Takes a fresh model and data set
# Train the model and save checkpoints 
def train(model, train_set, val_set):
    
    # Training options
    maxEpochs = np.arange(cfg['max_epochs'])
    lr = cfg['lr']
    l2_decay = cfg['l2_penalty']
    batchSize = cfg['batch_size']
    useCuda = cfg['use_cuda']
    
    # Check for cuda and set default compute device
    if ( torch.cuda.is_available() and useCuda ):
        device = torch.device("cuda")
    
    else:
        device = torch.device("cpu")
    
    print("Using %s for compute..." % device.type)
  
        
    # Send model to chose device
    model = model.to(device)

    # Create training set data loader, loss, and optimizer
    train_loader = DataLoader(train_set, batch_size=batchSize, collate_fn=Pad, shuffle=True)
    LossFcn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr = lr, weight_decay=l2_decay)
    
    # Loss collection and model chkpt every Nth batch
    N = 50
  
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
    early_stopping_range = 3
    
    # For each epoch train the model on the data
    for e in maxEpochs:
        
        print("In epoch %d..." % e)
        
        batch_loss = 0
        
        # For each batch in the loader send all 
        for idx, (data, labels) in enumerate(train_loader):
             
            # One_hot the inputs using the number of possible encoding values
            data = onehot(data, len(train_set.encode))
            
            # Push data and labels onto device
            data, labels = data.to(device), labels.to(device)
            
            # Reset gradients
            optim.zero_grad()

            # Run the data through the model
            Output = model(data)
            
            # Reshape output and loss to interpret timesteps as another sample
            Output, labels = Output.view(Output.shape[0]*Output.shape[1], -1), labels.view(labels.shape[0]*labels.shape[1])
                          
            # Compute Loss and add to average 
            Loss = LossFcn(Output, labels) 
            batch_loss += float(Loss.item())        
                          
            # Take gradient and update
            Loss.backward()
            optim.step()
                          
            # Calculate average batch losses every nth batch
            if ( idx % N == 0 and idx > 0  ):
                
                # Checkpoint, save model and optimizer 
                modelPath = os.path.join( checkpointPath, 'LSTMmodel_E_{0}_B_{1}'.format(e,idx) )
                optimPath = os.path.join( checkpointPath, 'LSTMoptim_E_{0}_B_{1}'.format(e,idx) )
                
                torch.save(model.state_dict(), modelPath )
                torch.save(optim.state_dict(), optimPath )
                # Calculate average training loss of N batches
                avg_loss = batch_loss/N
                batch_loss = 0
                epoch_losses.append(avg_loss)
                          
                print("Loss at %d batch of epoch %d is %f" % (idx, e, avg_loss) )
                
                # Try to clear some unused variable and run validation 
                del Loss, Output, data, labels
                val_loss = validation(model, val_set, device, LossFcn)
                
                # reset to training mode
                model.train()
         
                epoch_val_losses.append(val_loss)
                
                print("Validation Loss at %d batch of epoch %d is %f" % (idx, e, val_loss))
            
            # Check for early stopping with monotonicity 
            # Only check when we have at least 3 values to check
            if (len(epoch_val_losses[early_stopping_range::-1]) > early_stopping_range and monotonicIncr(epoch_val_losses[3::-1])):
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
                          
    print("Finished training for %d epochs..." % e)
        
# Check whether the list is monotonically increasing
def monotonicIncr(lst):

    # Monotonic increasing means subsequent values are always larger
    # Differences are then always <= 0 
    diff = np.array(lst[::-1])
    diff = (diff < 0)
    
    # If all the differences are > 0 we are monotonically decreasing
    return diff.all()
    
# Splits up full dataset to train and validation
# Returns pytorch Dataset of splits
def data_split(data):
    
    # Sample 10% from the dataset for validation
    val = data.sample(frac = .01)
    
    # Remove the sampled rows from the original
    train = data.drop(val.index)
    
    # Reset both indices for uniformity
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    
    # Make song datasets from both dataframes
    val_set = SongData(val)
    train_set = SongData(train)
    
    return train_set, val_set
    
    
    
# Entry point
if __name__ == "__main__":
    
    print("Reading data from CSV...")
    
    # Read in data and create train_val split
    data = pd.read_csv('songdata.csv')
    
    print("Dividing whole data into training and validation sets...")
    
    # Split data to training and validation set
    train_set, val_set = data_split(data)
    
    # Make fresh model to train
    # Both input and output are size of possible characters
    # We are inputting characters at t and asking to predict character t+1 
    model = GenLSTM(input_size = len(train_set.encode), output_size = len(train_set.encode) )
    
    print("Beginning model training...")
    
    # Train the model
    train(model, train_set, val_set)
    
    
    