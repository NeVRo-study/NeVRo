<h1>NeVRo</h1> 

<h2>Intro</h2> 

We used virtual reality (VR) to investigate arousal under ecologically valid conditions. 45 subjects experienced virtual roller coaster rides while their neural (EEG) and peripheral physiological (ECG) responses were recorded. Afterwards, they rated their arousal during the rides on a continuous scale while viewing a recording of their experience.

<h2>Methods</h2> 
We tackled the data with three model approaches.

<h4>SPoC Model</h4> 
<em>upload coming soon</em><br> 
<a href="https://doi-org.browser.cbs.mpg.de/10.1016/j.neuroimage.2013.07.079">Source Power Comodulation (SPoC)</a> decomposes the EEG signal such that it maximes the covariance between the power-band of the frequency of interest (here alpha, 8-12Hz) and the target variable (ratings).

<h4>CSP Model</h4>
<a href="https://ieeexplore.ieee.org/document/4408441/">Common Spatial Pattern (CSP)</a> algorithm decomposes the EEG such that it maximes the difference in variance between two dichotomic classes (here low and high arousal) in the EEG signal.

<h4>LSTM Model</h4>
<a href="https://doi.org/10.1162/neco.1997.9.8.1735">Long Short-Term Memory (LSTM)</a> recurrent neural networks (RNNs) were trained on alpha-frequency components of the recorded EEG signal to predict subjective reports of arousal (ratings) in a binary (low and high arousal) and a continous prediction task. The fed EEG components were generated via <a href="https://doi.org/10.1016/j.neuroimage.2011.01.057">Spatio Spectral Decomposition (SSD)</a> or SPoC. The SSD emphasizes the frequency of interest (here alpha) while attenuating the adjecent frequency bins. Performances of SPoC-trained models served as benchmark-proxies for models that were trained only on neural alpha information.

Furthermore, we tested whether peripheral physiological responses, here the cardiac information (ECG), increases the performance of the model, and therefore encodes additional information about the subjective experience of arousal.
