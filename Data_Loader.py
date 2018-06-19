'''
Jinsung Yoon (06/19/2018)
Data Loading
'''

#%% Packages
import numpy as np

#%% Main Function
'''
1. train_rate: training / testing set ratio
2. missing_rate: the amount of introducing missingness
3. missing_setting:
   - MAR: Missing At Random
   - MCAR: Missing Completely At Random 
'''

def Data_Loader(train_rate, missing_rate, missing_setting):
    
    #%% Normalization Function
    def MinMaxScaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    #%% 1. Data Preprocessing (Feature, Time, Label)
    # Data Input
    xy = np.loadtxt("/home/jinsung/Documents/Jinsung/MRNN/data/Example.csv", delimiter=",",skiprows=1)
    
    # Label: Diabetes
    y = xy[:,19]    
    y[np.where(y<0)] = 0

    # Time
    t = xy[:,-1]   
    # Yearly Based
    t = t/365
    
    # Feature (Time, Label, ID Delete)
    xy = np.delete(xy, [0,19,114], axis = 1)
    # Normalization
    xy = MinMaxScaler(xy)
    x = xy
    
    
    #%% 2. Introduce Missingness
    
    #%% (1) MCAR Setting
    if missing_setting == 'MCAR':
        
        # Missing matrix construct
        temp_m = np.random.uniform(0,1,[len(x), len(x[0])]) 
        m = np.zeros([len(x), len(x[0])])
        m[np.where(temp_m < missing_rate)] = 1
    
        # Introduce missingness
        new_x = np.copy(x)    
        new_x[np.where(m==1)] = 0       
    
    #%% (2) MAR Setting
    if missing_setting == "MAR":    
        
        # Weight initialization for introduce missingness
        w = np.random.uniform(0., 1., size = [len(x[0]),len(x[0])])
    
        # Missing matrix initialization
        m = np.zeros((len(x),len(x[0])))
    
        # For each feature
        for i in range(len(x[0])):
            if i == 0:
                A = np.random.uniform(0., 1., size = [len(x),])
                B = A > missing_rate
                m[:,i] = 1.*B
            if i > 0:
                New1 = np.matmul(x[:,:(i)]*m[:,:(i)],w[i,:i])
                New2 = np.matmul(m[:,:i],w[i,:i])
                New = New1 + New2      
                New = np.exp(-New)
            
                Q = np.percentile(New, 100*missing_rate)
                B = New > Q
                m[:,i] = 1.*B
        
        # Introduce missingness
        new_x = np.copy(x)    
        new_x[np.where(m==1)] = 0       

    #%% Time Gap Computation
    
    td = np.zeros([len(x[:,0]),len(x[0,:])])
    
    for i in range(int(len(y)/3)):
        for j in range(len(x[0,:])):
            if (i % 3 == 0):
                td[i,j] = 0
            elif (m[i-1,j] == 0):
                td[i,j] = td[i-1,j] + t[i] - t[i-1]
                
        
    #%% Building the dataset
    '''
    X: Original Feature
    Z: Feature with Missing
    M: Missing Matrix
    Y: Label
    T: Time Gap
    '''
    dataX = []    
    dataZ = []
    dataY = []
    dataM = []
    dataT = []

    # For each patient (total 3902 patients), 
    for i in range(int(len(y)/3)):
        _x = x[(3*i):(3*(i+1)),:]
        _z = new_x[(3*i):(3*(i+1)),:]
        _m = m[(3*i):(3*(i+1)),:]
        _y = y[(3*i):(3*(i+1))]    
        _td = td[(3*i):(3*(i+1))]    
        
        dataX.append(_x)
        dataZ.append(_z)
        dataY.append(_y)
        dataM.append(_m)
        dataT.append(_td)
        
    #%% Train / Test Dividing

    # Number of training set
    train_size = int(len(dataY) * train_rate)
    
    # Random index
    idx = np.random.permutation(len(dataY))    
    
    # Subset of original dataX, dataM, dataZ, dataY
    trainX, testX = np.array([dataX[i] for i in idx[0:train_size]]), np.array([dataX[i] for i in idx[train_size:len(dataY)]])
    trainZ, testZ = np.array([dataZ[i] for i in idx[0:train_size]]), np.array([dataZ[i] for i in idx[train_size:len(dataY)]])
    trainM, testM = np.array([dataM[i] for i in idx[0:train_size]]), np.array([dataM[i] for i in idx[train_size:len(dataY)]])
    trainY, testY = np.array([dataY[i] for i in idx[0:train_size]]), np.array([dataY[i] for i in idx[train_size:len(dataY)]])
    trainT, testT = np.array([dataT[i] for i in idx[0:train_size]]), np.array([dataT[i] for i in idx[train_size:len(dataY)]])
    
    
    return [trainX, trainZ, trainM, trainY, trainT, testX, testZ, testM, testY, testT]

