# NeVRo

<b>Intro</b>

We used virtual reality (VR) to investigate arousal under ecologically valid conditions. 45 subjects experienced virtual roller coaster rides while their neural (EEG) and peripheral physiological (ECG) responses were recorded. Afterwards, they rated their arousal during the rides on a continuous scale while viewing a recording of their experience.

<b>Methods</b>

We tackled the data with three model approaches.

*SPoC Model* [upload coming soon] <br> 
Source Power Comodulation (SPoC) decomposes the EEG signal such that it maximes the covariance between the power-band of the frequency of interest (here alpha, 8-12Hz) and the target variable (ratings).

*CSP Model* <br>
Common Spatial Pattern (CSP) algorithm decomposes the EEG such that it maximes the difference in variance between two dichotomic classes (here low and high arousal) in the EEG signal.

*LSTM Model* <br>
Long Short-Term Memory (LSTM) recurrent neural networks (RNNs) were trained on alpha-frequency components of the recorded EEG signal to predict subjective reports of arousal (ratings) in a binary (low and high arousal) and a continous prediction task. The fed EEG components were generated via Spatio Spectral Decomposition (SSD) or SPoC. The SSD emphasizes the frequency of interest (here alpha) while attenuating the adjecent frequency bins. Performances of SPoC-trained models served as benchmark-proxies for models that were trained only on neural alpha information.

Furthermore, we tested whether peripheral physiological responses, here the cardiac information (ECG), increases the performance of the model, and therefore encodes additional information about the subjective experience of arousal.
