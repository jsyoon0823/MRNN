'''
Jinsung Yoon (09/06/2018)
Data Loading
'''

#%% Necessary Packages
import numpy as np

#%% Google data loading

'''
1. train_rate: training / testing set ratio
2. missing_rate: the amount of introducing missingness
'''

def Data_Loader_Complete(train_rate = 0.8, missing_rate = 0.2):
    
    #%% Normalization
    def MinMaxScaler(data):        
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-8)

    #%% Data Preprocessing
    xy = np.loadtxt('/home/jinsung/Documents/Jinsung/MRNN/MRNN_New_Revision/Data/GOOGLE.csv', delimiter = ",",skiprows = 1)
    xy = xy[::-1]
    xy = MinMaxScaler(xy)
    x = xy
    
    #%% Parameters
    seq_length = 7
    col_no = len(x[0,:])
    row_no = len(x[:,0]) - seq_length
    
    # Dataset build
    dataX = []
    for i in range(0, len(xy[:,0]) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    #%% Introduce Missingness (MCAR)
    
    dataZ = []
    dataM = []
    dataT = []
    
    for i in range(row_no):
        
        #%% Missing matrix construct
        temp_m = np.random.uniform(0,1,[seq_length, col_no]) 
        m = np.zeros([seq_length, col_no])
        m[np.where(temp_m >= missing_rate)] = 1
        
        dataM.append(m)
        
        #%% Introduce missingness to the original data
        z = np.copy(dataX[i])    
        z[np.where(m==0)] = 0
        
        dataZ.append(z)
        
        #%% Time gap generation
        t = np.ones([seq_length, col_no])
        for j in range(col_no):
            for k in range(seq_length):
                if (k > 0):
                    if (m[k,j] == 0):
                        t[k,j] = t[k-1,j] + 1
                        
        dataT.append(t)
        
    #%% Building the dataset
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    T: Time Gap
    '''
                
    #%% Train / Test Division   
    train_size = int(len(dataX) * train_rate)
    
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainZ, testZ = np.array(dataZ[0:train_size]), np.array(dataZ[train_size:len(dataX)])
    trainM, testM = np.array(dataM[0:train_size]), np.array(dataM[train_size:len(dataX)])
    trainT, testT = np.array(dataT[0:train_size]), np.array(dataT[train_size:len(dataX)])
    
    return [trainX, trainZ, trainM, trainT, testX, testZ, testM, testT]


#%% Loading the data with missing values (represnted in nan)


def Data_Loader_Incomplete(train_rate):
    
    #%% Normalization
    def MinMaxScaler(data):
        
        numerator = data - np.nanmin(data, 0)
        denominator = np.nanmax(data, 0) - np.nanmin(data, 0)
        return numerator / (denominator + 1e-8)

    #%% Data Preprocessing
    xy = np.loadtxt('/home/jinsung/Documents/Jinsung/MRNN/MRNN_New_Revision/Data/GOOGLE_Missing.csv', delimiter = ",",skiprows = 1)
    xy = xy[::-1]
    xy = MinMaxScaler(xy)
    x = xy
    
    #%% Parameters
    seq_length = 7
    col_no = len(x[0,:])
    row_no = len(x[:,0]) - seq_length
    
    # Dataset build
    dataX = []
    for i in range(0, len(x[:,0]) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    #%% Introduce Missingness (MCAR)
    
    dataZ = []
    dataM = []
    dataT = []
    
    for i in range(row_no):
        
        #%% Missing matrix construct
        m = np.ones([seq_length, col_no])
        m[np.where(np.isnan(dataX[i])==1)] = 0
        
        dataM.append(m)
        
        #%% Introduce missingness to the original data
        z = np.copy(dataX[i])    
        z[np.where(m==0)] = 0
        
        dataZ.append(z)
        
        #%% Time gap generation
        t = np.ones([seq_length, col_no])
        for j in range(col_no):
            for k in range(seq_length):
                if (k > 0):
                    if (m[k,j] == 0):
                        t[k,j] = t[k-1,j] + 1
                        
        dataT.append(t)
        
    #%% Building the dataset
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    T: Time Gap
    '''
                
    #%% Train / Test Division   
    train_size = int(len(dataX) * train_rate)
    
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainZ, testZ = np.array(dataZ[0:train_size]), np.array(dataZ[train_size:len(dataX)])
    trainM, testM = np.array(dataM[0:train_size]), np.array(dataM[train_size:len(dataX)])
    trainT, testT = np.array(dataT[0:train_size]), np.array(dataT[train_size:len(dataX)])
    
    return [trainX, trainZ, trainM, trainT, testX, testZ, testM, testT]
    

'''    
#%% Data with missingness
    
xy = np.loadtxt('/home/jinsung/Documents/Jinsung/MRNN/MRNN_New_Revision/Data/GOOGLE_Missing.csv', delimiter = ",",skiprows = 1)
row_no = len(xy[:,0])
col_no = len(xy[0,:])

temp_m = np.random.uniform(0,1,[row_no, col_no]) 
m = np.zeros([row_no, col_no])
m[np.where(temp_m >= 0.2)] = 1
        
xy[np.where(m==0)] = np.nan

np.savetxt('/home/jinsung/Documents/Jinsung/MRNN/MRNN_New_Revision/Data/GOOGLE_Missing.csv',xy)
'''