import torch
import pandas as pd
import numpy as np
import os
from dataloader import data_split,onehot
from models import GenLSTM
from configs import cfg

# Load the state_dict of a model given path
def load_model(path):
    
    # Make a fresh model
    model = GenLSTM(input_size = len(SongData.vocab), output_size = len(SongData.vocab) )
    
    # Load in saved model params at checkpoint path
    model.load_state_dict(torch.load(path))
    
    return model

# Takes a pre-trained model and generate new text
def generateText(data, model):
    
    # Number of texts to generate
    # Max length of text
    numText = 10
    maxLen = 500
    
    # Start the sentence with the SOS ordinal, ordinal value is 0
    # One-hot the values for generation
    text = torch.tensor(np.zeros((numText, 1))).long()
    
    EOSreached = False
    currSeqIndex = 0
    hc = None
    Temp = cfg['temperature']
    
    model.to(device)
    
    # Put into eval mode
    model.eval()
    
    # No need for gradient
    with torch.no_grad():
        
        # While we haven't reached the EOS run latest character through model with the previous hidden and cell state
        # Softmax and sample output and use that as input
        while (not EOSreached and currSeqIndex < maxLen):
            
            # Onehot the batch at the current sequence index
            oh_input = onehot(text[:,currSeqIndex:currSeqIndex+1], len(data.encode))
            oh_input = oh_input.to(device)
            
            # Pass characters at current sequence index through model
            # First character we don't have hc
            if not hc:
                output, hc = model(oh_input)
            # Every subsequent character we pass in the previous hc
            else:
                output, hc = model(oh_input, hc)
                
            # Convert outputs to probabilities using softmax with temperature
            probs = torch.nn.functional.softmax(output/Temp, dim=2)
            
            # Create distribution and sample for the next indices
            sampled = torch.distributions.Categorical(probs).sample()
            
            # Join sample with the current body of characters
            text = torch.cat((text, sampled.cpu()), dim=1)
            
            currSeqIndex += 1
            
            # Check whether each rows contain any 1s(EOS) and that must be true for all rows
            if (np.array(text) == 1).any(axis=1).all():
                EOSreached = True
                
    # Decode the ordinals to characters
    generated = [''.join([data.decode[c] for c in gen]) for gen in np.array(text)]
    
    # For each string remove everything after the first EOS(\3)
    generated = [text[0:text.find('\3')+1] for text in generated]
    
    print("Finished text generation...", flush=True)
    
    return generated

if __name__ == "__main__":
    
    print("Loading necessary files to generate...", flush=True)
    # Get dataset to know meta-data
    data = pd.read_csv('songdata.csv')
    SongData.init_vocab(data)
    train_set, val_set = data_split(data)
    
    # Get path
    checkpointPath = 'model_checkpoints/training_session0/'
    savedModePath = 'LSTMmodel_E_1_B_1338.mdl'
    
    # Load in trained model
    model = load_model(data, os.path.join(checkpointPath, savedModePath))
    
    
    useCuda = cfg['use_cuda']
    # Check for cuda and set default compute device
    if ( torch.cuda.is_available() and useCuda ):
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("Using %s for compute..." % device.type, flush=True)
    
    
    print("Beginning text generation...", flush=True)
    
    # Generate text
    gen = generateText(train_set, model)
    

    