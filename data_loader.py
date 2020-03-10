"""Functions for data loading.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
           "Estimating Missing Data in Temporal Data Streams Using 
           Multi-Directional Recurrent Neural Networks," 
           in IEEE Transactions on Biomedical Engineering, 
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
import numpy as np
from utils import MinMaxScaler


def data_loader (file_name = 'data/google.csv', seq_len = 7, 
                 missing_rate = 0.2):
  """Load complete data and introduce missingness.
  
  Args:
    - file_name: the location of file to be loaded
    - seq_len: sequence length
    - missing_rate: rate of missing data to be introduced
    
  Returns:
    - x: data with missing values
    - m: observation indicator (m=1: observe, m=0: missing)
    - t: time information (time difference between two measurments)
    - ori_x: original data without missing values (for evaluation)
  """
  
  # Load the dataset
  data = np.loadtxt(file_name, delimiter = ",", skiprows = 1)
  # Reverse time order
  data = data[::-1]
  # Normalize the data
  data, norm_parameters = MinMaxScaler(data)
  
  # Parameters
  no, dim = data.shape
  no = no - seq_len
  
  # Define original data
  ori_x = list()  
  for i in range(no):
    temp_ori_x = data[i:(i+seq_len)]
    ori_x = ori_x + [temp_ori_x]
    
  # Introduce missingness
  m = list()
  x = list()
  t = list()
  
  for i in range(no):
    # m
    temp_m = 1*(np.random.uniform(0, 1, [seq_len, dim]) > missing_rate)
    m = m + [temp_m]
    # x
    temp_x = ori_x[i].copy()
    temp_x[np.where(temp_m == 0)] = np.nan
    x = x + [temp_x]
    # t
    temp_t = np.ones([seq_len, dim])
    for j in range(dim):
      for k in range(1, seq_len):
        if temp_m[k, j] == 0:
          temp_t[k, j] = temp_t[k-1, j] + 1
    t = t + [temp_t]
    
  # Convert into 3d numpy array
  x = np.asarray(x)
  m = np.asarray(m)
  t = np.asarray(t)
  ori_x = np.asarray(ori_x)  
  
  # Fill 0 to the missing values
  x = np.nan_to_num(x, 0)

  return x, m, t, ori_x       
   