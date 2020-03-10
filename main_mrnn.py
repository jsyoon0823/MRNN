"""Main function for MRNN

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
--------------------------------------------------
(1) Load the data
(2) Train MRNN model
(3) Impute missing data
(4) Evaluate the imputation performance
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import shutil
import os

from data_loader import data_loader
from mrnn import mrnn
from utils import imputation_performance


def main (args):
  """MRNN main function.
  
  Args:
    - file_name: dataset file name
    - seq_len: sequence length of time-series data
    - missing_rate: the rate of introduced missingness
    - h_dim: hidden state dimensions
    - batch_size: the number of samples in mini batch
    - iteration: the number of iteration
    - learning_rate: learning rate of model training
    - metric_name: imputation performance metric (mse, mae, rmse)
    
  Returns:
    - output:
      - x: original data with missing
      - ori_x: original data without missing
      - m: mask matrix
      - t: time matrix
      - imputed_x: imputed data
      - performance: imputation performance
  """  
  
  ## Load data
  x, m, t, ori_x = data_loader(args.file_name, 
                               args.seq_len, 
                               args.missing_rate)
      
  ## Train M-RNN
  # Remove 'tmp/mrnn_imputation' directory if exist
  if os.path.exists('tmp/mrnn_imputation'):
    shutil.rmtree('tmp/mrnn_imputation')
  
  # mrnn model parameters
  model_parameters = {'h_dim': args.h_dim,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration, 
                      'learning_rate': args.learning_rate}  
  # Fit mrnn_model
  mrnn_model = mrnn(x, model_parameters)
  mrnn_model.fit(x, m, t)
  
  # Impute missing data
  imputed_x = mrnn_model.transform(x, m, t)
  
  # Evaluate the imputation performance
  performance = imputation_performance (ori_x, imputed_x, m, args.metric_name)
  
  # Report the result
  print(args.metric_name + ': ' + str(np.round(performance, 4)))
  
  # Return the output
  output = {'x': x, 'ori_x': ori_x, 'm': m, 't': t, 'imputed_x': imputed_x,
            'performance': performance}
   
  if os.path.exists('tmp/mrnn_imputation'): 
    shutil.rmtree('tmp/mrnn_imputation')
  
  return output


##
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file_name',
      default='data/google.csv',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length of time-series data',
      default=7,
      type=int)
  parser.add_argument(
      '--missing_rate',
      help='the rate of introduced missingness',
      default=0.2,
      type=float)
  parser.add_argument(
      '--h_dim',
      help='hidden state dimensions',
      default=10,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini batch',
      default=128,
      type=int)
  parser.add_argument(
      '--iteration',
      help='the number of iteration',
      default=2000,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning rate of model training',
      default=0.01,
      type=float)
  parser.add_argument(
      '--metric_name',
      help='imputation performance metric',
      default='rmse',
      type=str)
  
  args = parser.parse_args() 
  
  # Call main function  
  output = main(args)