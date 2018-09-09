'''
Jinsung Yoon (09/11/2018)
M-RNN Main
'''
#%% Packages
import numpy as np

#%% Functions
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/MRNN/MRNN_New_Revision')

# 1. Data Preprocessing
'''
If the original data is complete, use Complete, 
If the original data is incomplete, use Incomplete
'''
from Data_Loader import Data_Loader_Complete, Data_Loader_Incomplete

# 2. Imputation Block
from M_RNN import M_RNN

#%% Parameters
# train Parameters
train_rate = 0.8
missing_rate = 0.2

# Mode
'''
If the original data is complete, use Complete, 
If the original data is incomplete, use Incomplete
'''
mode_sets = ['Complete', 'Incomplete']
mode = mode_sets[0] 

#%% Main


#%% Complete data
if mode == 'Complete':
    # 1. Data Preprocessing (Add missing values)
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    T: Time Gap
    '''
    _, trainZ, trainM, trainT, testX, testZ, testM, testT = Data_Loader_Complete(train_rate, missing_rate)
    
    # 2. M_RNN_Imputation (Recovery)
    _, Recover_testX = M_RNN(trainZ, trainM, trainT, testZ, testM, testT)
    
    # 3. Imputation Performance Evaluation
    MSE = np.sum ( np.square ( testX * (1-testM) - Recover_testX * (1-testM) ) )  / np.sum(1-testM) 

    print(MSE)    
    
#%% Incomplete data
if mode == 'Incomplete':
    # 1. Data Preprocessing (Not add missing values because the original data has missing values)
    _, trainZ, trainM, trainT, testX, testZ, testM, testT = Data_Loader_Incomplete(train_rate)
    
    # 2. M_RNN_Imputation (Recovery on testX)
    _, Recover_testX = M_RNN(trainZ, trainM, trainT, testZ, testM, testT)
    
    '''
    We do not compute MSE as performance metrics because there is no ground truth. (NAN in trainX)    
    '''    