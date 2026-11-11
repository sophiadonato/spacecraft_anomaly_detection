# spacecraft_anomaly_detection
Build a LSTM anomaly detector, with dynamic thresholding on NASA SMAP and Curiosity rover telemetry.

## Dataset 
Data from NASA Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL) rover, Curiosity. 
- The data from SMAP has 55 channels and the one from MSL has 27 channels A channel corresponds to a .npy file, and is named after its type (P = power, T = temperature, ..). The shape of each channel is (n_timesteps, n_inputs). The inputs correspond to telemetry value (column 0) and commands sent to the spacecraft.
- The data is already split into training and testing sets. Each channel has a measure (ie a .npy file) in the training set without anomalies and a measure in the testing set with anomalies labeled thanks to a .csv file
- The labeled_anomalies.csv file has different columns : chan_id (channel name, eg: P-1), spacecraft (SMAP or MSL), anomaly sequences (start and end indices of true anomalies in the test set, eg : [[5000, 5030], [5610, 6086]]), class (type of each anomaly sequence), num_values (number of telemetry values)
- An anomaly can be of type : point (values far outside normal range), or contextual (values that seem normal but are wrong for that moment)

## Anomaly detection using LSTM
The solution presented in the paper (see references) and used here is the following one : one LSTM is build per channel and uses prior telemetry values and encoded commands. At each time step, prediction are computed and then compared to the actual ones. Errors on the predictions correspond to potential anomaly. Then, a nonparametric, unsupervised approach for thresholding these errors is used to identifying anomalous sequences of errors. </br>
The followed pipeline is : 
- Training : train the LSTMs to learn normal behavior on the training set (assumed without anomalies)
- Anomaly detection : used the trained LSTMs on the test files to output the error between the prediction (normal behavior) and the actual value. You will therefore have an error curve for all timesteps (except first 250)
- Smooth the error curve, apply non parametric threshold and prune false alarms to obtain prediction of anomaly seqences
- Compare predicted anomaly to actual ones using the .csv file


## References
- Dataset on Kaggle : https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data
- Github repo : https://github.com/khundman/telemanom
- Paper : https://arxiv.org/pdf/1802.04431
- Instagram account with explanations : @byswergiu