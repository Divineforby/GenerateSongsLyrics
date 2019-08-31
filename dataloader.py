import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

# Takes a numpy array of indices and create one-hot array for all of these indices
# Return the tensor
def onehot(indices, numFeature):
    
    # Indices are shaped (batch, seqlen)
    batchSize = indices.shape[0]
    seqLen = indices.shape[1]
    
    # 2d Array of zeros for easier indexing
    onehotted = np.zeros((batchSize * seqLen, numFeature))

    # Index the correct location for each character
    onehotted[np.arange(batchSize*seqLen), indices.reshape(1,-1)] = 1

    return torch.Tensor(onehotted.reshape(batchSize, seqLen, -1))

# Custom collate function for Dataloader to pack variable-length sequences for batch
def Pad(batch):
    # Sort batch by length of sequence
    batch.sort(key=lambda x:len(x[1]), reverse=True)
    
    # Take in batch of ordinals and split between data and target
    data = [x[0] for x in batch]
    target = [x[1] for x in batch]
    
    # Pad both data and target with EOS
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=1)
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=1)
    
    return data, target

# Pytorch dataset of for song lyrics
class SongData(Dataset):
    
    # Constructs the dataset by loading in the csv file for song lyrics
    def __init__(self, dataframe):
        
        # Read and preprocess data
        self.data = dataframe

        # Used for indexing when one-hotting encoding characters
        self.encode = self.ChartoOrd(self.data)
        # Create the reverse mapping to decode
        self.decode = self.OrdtoChar(self.encode)
        
        # Preprocess data
        self.preprocessData(self.data)

    # Attach SOS and EOS, uniform text lengths, convert all characters to one-hot indices
    def preprocessData(self, data):
        
        # Attach SOS, EOS to all text
        data['text'] = '\2' + data['text'] + '\3'
        
        # Translate each character into its ordinals once for easy one-hot encoding
        data['to_onehot'] = data['text'].apply(lambda x: [self.encode[c] for c in x])
        
    # Defines length
    def __len__(self):
        return(len(self.data))
    
    # Defines how to a single training sample
    def __getitem__(self, idx):
    
        # Get list of ordinal values to transform for this text
        indices = self.data.iloc[idx].to_onehot
        ordinals = torch.tensor(indices[:-1])
        labels = torch.tensor(indices[1:])
        
        # One hots the indices and return the tensor
        return ordinals, labels
    
    
    # Finds all unique characters and assign them an ordinal value 
    def ChartoOrd(self, data):
        
        # Dictionary of unique characters
        uniqueChars = {}
        
        # Assign \2 as SOS and \3 as EOS mapping the characters to first ordinals
        uniqueChars['\2'] = 0
        uniqueChars['\3'] = 1
        
        # Ord counter, start at 2 to account for our SOS and EOS
        ordCounter = 2
        
        # Check each character to see if it exists in the dict
        # If not give it an ordinal number and increment the ord 
        # for the next unique character
        for lyrics in data.text:
            for c in lyrics:
                if c not in uniqueChars:
                    uniqueChars[c] = ordCounter
                    ordCounter += 1
            
        return uniqueChars
    
    # Reverse the encoder dict 
    def OrdtoChar(self, encoder):
        
        # Make a decoder mapping
        decoded = {}
        
        # Reverse
        for k,v in encoder.items():
            decoded[v] = k
        
        
        return decoded 
 
