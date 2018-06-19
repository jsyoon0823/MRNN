'''
Jinsung Yoon (06/19/2018)
M-RNN Main
'''

#%% Packages
import numpy as np
from sklearn.metrics import roc_auc_score

#%% Functions
import sys
sys.path.append('/home/jinsung/Documents/Jinsung/MRNN/M_RNN_Github')

# 1. Data Preprocessing
import Data_Loader

# 2. Imputation Block
import M_RNN

# 3. Prediction Block
import RNN_Basic

#%% Parameters
# train Parameters
train_rate = 0.8
missing_rate = 0.2
missing_setting = 'MAR'

#%% Main

# 1. Data Preprocessing
'''
X: Original Feature
Z: Feature with Missing
M: Missing Matrix
Y: Label
T: Time Gap
'''
trainX, trainZ, trainM, trainY, trainT, testX, testZ, testM, testY, testT = Data_Loader.Data_Loader(train_rate, missing_rate, missing_setting)

# 2. M_RNN_Imputation
Recover_trainX, Recover_testX = M_RNN.M_RNN(trainX, trainZ, trainM, trainT, testX, testZ, testM, testT)

# 3. Recovery
New_trainX = (trainM) * trainX  + (1-trainM) * Recover_trainX
New_testX  = (testM)  * testX   + (1-testM)  * Recover_testX

# 4. Imputation Performance Evaluation
RMSE = np.sqrt( np.sum ( np.square ( testX * (1-testM) - Recover_testX * (1-testM) ) )  / np.sum(1-testM) )

# 5. Prediction
Prediction = RNN_Basic.RNN_Basic(Recover_trainX, trainY, Recover_testX)

# 6. Prediction Performance Evaluation
Test_No = len(testY)
Seq_No = len(testY[0])

Eval_Y = np.reshape(testY, [Test_No*Seq_No,])
New_Y = np.reshape(Prediction, [Test_No*Seq_No,])

AUC = roc_auc_score(Eval_Y, New_Y)

