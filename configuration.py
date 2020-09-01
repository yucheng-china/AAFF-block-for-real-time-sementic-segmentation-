from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

train_config = {
    
  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD, RMSProp and MOMENTUM are supported
                       'momentum': 0.9,
                       'use_nesterov': False,
                       'decay': 0.9, },          # Discounting factor for history gradient(useful in RMSProp Mode)

  # Learning rate configs
  'lr_config': {'policy': 'polynomial',         # piecewise_constant, exponential, polynomial and cosine
                'initial_lr': 0.01,
                'power': 0.9,                   # Only useful in polynomial
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685113737513527,
                'staircase': True, },

}

