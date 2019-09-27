import torch
import pandas as pd
import numpy as np
import os
import sys
from dataloader import data_split,onehot, SongData
from models import *
from configs import cfg

# Load the state_dict of a model given path
def load_model(path):
    
    # Make a fresh model
    model = GenGRU(input_size = len(SongData.vocab), output_size = len(SongData.vocab))
    
    # Load in saved model params at checkpoint path
    model.load_state_dict(torch.load(path))
    
    return model

# Takes a pre-trained model and generate new text
def generateText(data, model, temp, num_samples, max_len):
    
    # Number of texts to generate
    # Max length of text
    numText = num_samples
    maxLen = max_len
    
    # Start the sentence with the SOS ordinal, ordinal value is 0
    # One-hot the values for generation
    text = torch.tensor(np.zeros((numText, 1))).long()
    
    EOSreached = False
    currSeqIndex = 0
    hc = None
    Temp = temp
    
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
           
            output, hc = model(oh_input, hc)
                
            # Convert outputs to probabilities using softmax with temperature
            probs = torch.nn.functional.softmax(output/Temp, dim=2)
            
            # Create distribution and sample for the next indices
            sampled = torch.distributions.categorical.Categorical(probs).sample()
            
            # Join sample with the current body of characters
            text = torch.cat((text, sampled.cpu()), dim=1)
            
            currSeqIndex += 1
            
            # Check whether each rows contain any 1s(EOS) and that must be true for all rows
            if (np.array(text) == 1).any(axis=1).all():
                EOSreached = True
                
    # Decode the ordinals to characters
    generated = [''.join([data.decode[c] for c in gen]) for gen in np.array(text)]
    
    # For each string remove everything after the first EOS(\3) and remove all SOS(\2)
    generated = [text[0:text.find('\3')+1].replace('\2','') if text.find('\3') > 0 else text.replace('\2','') 
                 for text in generated]
    
    print("Finished text generation...", flush=True)
    
    return generated

if __name__ == "__main__":
    
    # Check for arguments
    if (len(sys.argv) < 4):
      print("Usage: python generate.py <Temperature> <Number of samples> <Max length of each sample>", flush=True)
      sys.exit()
    
    # Parse arguments
    temp = float(sys.argv[1])
    num_samples = int(sys.argv[2])
    max_len = int(sys.argv[3])
    
    
    print("Loading necessary files to generate...", flush=True)
    # Get dataset to know meta-data
    data = pd.read_csv('songdata.csv')
    SongData.init_vocab(data)
    train_set, val_set = data_split(data)
   

    train_sess = 0

    # Get path
    checkpointPath = 'model_checkpoints/training_session_{0}/'.format(train_sess)
    savedModePath = 'LSTMmodel_E_49_B_2649.mdl'
    
    # Load in trained model
    model = load_model(os.path.join(checkpointPath, savedModePath))
    
    useCuda = cfg['use_cuda']
    # Check for cuda and set default compute device
    if ( torch.cuda.is_available() and useCuda ):
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print("Using %s for compute..." % device.type, flush=True)
    
    
    print("Beginning text generation...", flush=True)
    
    # Generate text
    gen = generateText(train_set, model, temp, num_samples, max_len)
    
    # Write out generated texts into result file
    resultPath = "./results/"
    resultFile = "generatedsamples_temp_{0}".format(temp)
    
    with open(os.path.join(resultPath, resultFile), "w+") as resFile:
        resFile.write("Generated sample using these options: Temperature - {0} Number of Samples - {1} Max Length Per Sample - {2}\n".format(temp, num_samples, max_len)) 
        for idx, text in enumerate(gen):
            resFile.write("SAMPLE NUMBER {0} \n\n".format(idx))
            resFile.write(text+"\n")
            resFile.write("---------------------------\n\n")
    
