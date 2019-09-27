mdl_cfg = {} 

mdl_cfg['hidden'] = 128 # Hidden layer of LSTM size
mdl_cfg['layers'] = 2 # Number of LSTM layers
mdl_cfg['dropout'] = 0.2 # Drop out probability of each cell
mdl_cfg['bd'] = True


cfg = {}
cfg['use_cuda'] = True
cfg['lr'] = 1e-4
cfg['max_epochs'] = 200
cfg['batch_size'] = 16
cfg['l2_penalty'] = 0
cfg['validation_batch_size'] = 64
