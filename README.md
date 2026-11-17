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

## Dynamic threshold explanation
The abnormal values are those for which the error between the actual value and the predicted one (by the LSTM) is above a certain threshold. For each error curve, a threshold is determined dynamically in the following way : 
- All thresholds looking like : $$\epsilon = \mu(err) + z * \sigma(err)$$ with z in (2.5,10.5) are tested
- For each z, a score is computed as 
$$s = \frac{\frac{\mu(err) - \mu(under)}{\mu(err)} + \frac{\sigma(err)-\sigma(under)}{\sigma(err)} }{|above| + |seq above|^2}$$
With 
- under : all values of the error under the proposed threshold 
- above : all values of the error above the proposed threshold. </br>
This score is build to have a tradeoff : 
- numerator : if we remove the points above the thresholds, are the mean and std decreasing ? if yes, it means the above points were really far, so probably true anomalies. if the threshold is too big, nothing is above it so the mean and std don't decrease : the denominator is small and score low.
- denominator : it counts the nb of anomaly points and the number of sequences. if the threshold is too low, everything is an anomaly, which is bad : the denominator is big and score low.  

## Pruning explanation 
Once having the detected anomaly sequences, pruning is applied to sort out the false positives. For that, the sequences are ordered by maximal error and compared in that order to one an other. If the drop of error between one sequence and the next one is less than 13%, it means that the sequence with the smaller error is actually a false positive and is ruled out. 

## Results 
The final results are a less good than the ones presented in the paper (75% precision and 39% recall), especially for the recall, which is much lower, meaning that there are a lot of missed anomalies. This is probably due to the LSTM initialisation, which is not the same in PyTorch (here) than in Keras (paper), and therefore maybe outputs less good predicitions. The rest of the code is exactly the same (parameters and steps). 

## References
- Dataset on Kaggle : https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data
- Github repo : https://github.com/khundman/telemanom
- Paper : https://arxiv.org/pdf/1802.04431
- Instagram account with explanations : @byswergiu