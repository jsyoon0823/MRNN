# MRNN
Multi-directional Recurrent Neural Networks

1. Dataset (Example.csv)
- This dataset is the example time-series dataset that can be used for testing M-RNN Architecture with Data_Loader.py
- 20th Column is used as the label \

2. Data_Loader.py
- Using the Raw dataset (Example.csv), it extracts the features, labels, and time information.
- It also divides training and testing sets for further experiments
- It has three input parameters
(1). train_rate: training / testing set ratio
(2). missing_rate: the amount of introducing missingness
(3). missing_setting:
   - MAR: Missing At Random
   - MCAR: Missing Completely At Random 
- It has 5 outputs for each training and testing set
(1). X: Original Feature
(2). Z: Feature with Missing
(3). M: Missing Matrix
(4). Y: Label
(5). T: Time Gap \

3. M_RNN.py
- Using the outputs of the Data_Loader.py, it imputes the missing features using M-RNN architecture
- It consists of Bi-directional GRU and MLP.
- The details of the M-RNN architecture can be found in the following link.
- https://arxiv.org/pdf/1711.08742.pdf \

4. RNN_Basic.py
- Using the imputed time-series data, it predicts time-series binary labels using simple RNN architecture
- It uses LSTM architecture \

5. M_RNN_Main.py
- Combine the above four components with various performance metrics
- It consists of RMSE and AUC metrics. \
