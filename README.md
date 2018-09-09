# MRNN
Multi-directional Recurrent Neural Networks

1. Datasets (GOOGLE.csv, GOOGLE_Missing.csv)
- These datasets are the example time-series datasets that can be used for testing M-RNN Architecture with Data_Loader.py

2. Data_Loader.py
- Using the Raw datasets, it extracts the features and time information.
- It also divides training and testing sets for further experiments
- It has two input parameters
(1). train_rate: training / testing set ratio
(2). missing_rate: the amount of introducing missingness

- It has 4 outputs for each training and testing set
(1). X: Original Feature
(2). Z: Feature with Missing
(3). M: Missing Matrix
(4). T: Time Gap 

3. M_RNN.py
- Using the outputs of the Data_Loader.py, it imputes the missing features using M-RNN architecture
- It consists of Bi-directional GRU and MLP.
- The details of the M-RNN architecture can be found in the following link.
- https://arxiv.org/pdf/1711.08742.pdf 

4. M_RNN_Main.py
- Combine the above three components with MSE performance metric.
