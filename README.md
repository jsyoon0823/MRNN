# Codebase for "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networkss (MRNN)"

Authors: Jinsung Yoon, William R. Zame, Mihaela van der Schaar

Paper: Jinsung Yoon, William R. Zame, Mihaela van der Schaar, 
       "Estimating Missing Data in Temporal Data Streams Using 
       Multi-Directional Recurrent Neural Networks," 
       IEEE Transactions on Biomedical Engineering, 2019.
 
Paper Link: https://ieeexplore.ieee.org/document/8485748

Contact: jsyoon0823@gmail.com

This directory contains implementations of MRNN framework for imputation
in time-series data using GOOGLE stocks dataset.

To run the pipeline for training and evaluation on MRNN framwork, simply run 
python3 -m main_mrnn.py.

### Command inputs:

-   file_name: data file name
-   seq_len: sequence length of time-series data
-   miss_rate: probability of missing components (to be introduced)
-   h_dim: hidden state dimensions
-   batch_size: the number of samples in mini batch
-   iteration: the number of iteration
-   learning_rate: learning rate of model training
-   metric_name: imputation performance metric


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
      help='imputation metric',
      default='rmse',
      type=str)

### Example command

```shell
$ python3 main_mrnn.py --file_name data/google.csv --seq_len 7 
--missing_rate: 0.2 --h_dim 10 --batch_size 128 --iteration 2000
--learning_rate 0.01 --metric_name rmse
```

### Outputs

-   x: original data with missing
-   ori_x: original data without missing
-   m: mask matrix
-   t: time matrix
-   imputed_x: imputed data
-   performance: imputation performance