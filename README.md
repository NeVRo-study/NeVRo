# NeVRo

<b>Intro</b>

We used virtual reality (VR) to investigate arousal under ecologically valid conditions. 45 subjects experienced virtual roller coaster rides while their neural (EEG) and peripheral physiological (ECG) responses were recorded. Afterwards, they rated their arousal during the rides on a continuous scale while viewing a recording of their experience.

<b>Methods</b>

We tackled the data with three model approaches.

*SPoC Model*
Source Power Comodulation (SPoC) algorithm finds the maximal covariance between the power-band of the frequency of interest (here alpha, 8-12Hz) and the target variable (ratings).

*CSP Model*
The Common Spatial Pattern (CSP) algorithm maximes the difference in variance between two dichotomic classes (here low and high arousal) in the EEG signal.

*LSTM Model*
Long Short-Term Memory (LSTM) recurrent neural networks (RNNs) were trained on alpha-frequency band-passed neural components of the recorded EEG signal to predict subjective reports of arousal (ratings). The EEG components were generated via Spatio Spectral Decomposition (SSD) or SPoC. The SSD emphasizes the frequency of interest (here alpha) while attenuating the adjecent frequency bins. 

Furthermore, we tested whether peripheral physiological responses, here the cardiac information (ECG), increases the performance of the model, and therefore encodes additional information about the subjective experience of arousal.
