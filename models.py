import torch
from configs import mdl_cfg

# Define the LSTM 
class GenLSTM(torch.nn.Module):
    
    # Initialize the LSTM 
    def __init__(self, input_size, output_size ):
        super(GenLSTM, self).__init__()
        
        # LSTM Layer(s) that map the input to hidden layers 
        self.LSTM = torch.nn.LSTM(input_size = input_size, hidden_size = mdl_cfg['hidden'], 
                                  num_layers = mdl_cfg['layers'], dropout = mdl_cfg['dropout'],
                                  batch_first=True)
        
        # Final Fully connected layer to map the output of LSTM to the number of classes
        self.FC = torch.nn.Linear(in_features = mdl_cfg['hidden'], out_features = output_size)
        
        # For each layer in the LSTM initialize the weights 
        for param in self.LSTM.named_parameters():
            if ( 'weight' in param[0] ):
                torch.nn.init.xavier_normal_(param[1])
                
        # Init fully connected layer
        torch.nn.init.xavier_normal_(self.FC.weight)
        
    # Define the forward pass    
    def forward(self, x):
        
        # Run through LSTM units
        out, hc = self.LSTM(x)
        
        # Run through FC for final layer
        out = self.FC(out)
        
        # Want raw outputs and not softmax to apply 
        # temperature during generation
        return out
        