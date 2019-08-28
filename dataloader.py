import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

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

    # Attach SOS and EOS, uniform text lengths, one-hot all characters
    def preprocessData(self, data):
        
        # Attach SOS, EOS to all text
        data['text'] = '\2' + data['text'] + '\3'
        
        # Find maximum length of characters 
        maxLen = max(data['text'].apply(len))
        
        # Append EOS to each text until the maxLen is reached
        data['text'] = data['text'].apply(lambda x: x + ('\3' * (maxLen - len(x))))
        
        # Translate each character into its ordinals once for easy one-hot encoding
        data['to_onehot'] = data['text'].apply(lambda x: [self.encode[c] for c in x])
        
    # Defines length
    def __len__(self):
        return(len(self.data))
    
    # Defines how to a single training sample
    def __getitem__(self, idx):
    
        # Get list of ordinal values to transform for this text
        indices = self.data.iloc[idx].to_onehot
        onehotted = self.onehotted(indices[:-1])
        labels = torch.Tensor(indices[1:])
        
        # One hots the indices and return the tensor
        return onehotted, labels.int()
    
    # Takes a list of indices and create one-hot array for all of these indices
    # Return the tensor
    def onehotted(self, indices):
        
        # Dummy array to be indexed
        # Row = Character, Column = Possible character
        # Size of one-hot is 1xC where C is possible characters
        # Size of matrix is NxC where N is the number of characters in this text
        dummy = np.zeros((len(indices), len(self.encode)))
        
        # Index the correct location for each character
        dummy[np.arange(len(indices)), indices] = 1
        
        return torch.Tensor(dummy)
    
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
        