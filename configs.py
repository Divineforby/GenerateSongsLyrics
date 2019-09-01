mdl_cfg = {} 

mdl_cfg['hidden'] = 512 # Hidden layer of LSTM size
mdl_cfg['layers'] = 2 # Number of LSTM layers
mdl_cfg['dropout'] = 0 # Drop out probability of each cell


cfg = {}
cfg['use_cuda'] = True
cfg['lr'] = 10e-5
cfg['max_epochs'] = 50
cfg['batch_size'] = 16
cfg['l2_penalty'] = 1
cfg['validation_batch_size'] = 64