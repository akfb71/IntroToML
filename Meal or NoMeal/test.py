#!/usr/bin/env python
# coding: utf-8

# In[47]:


#libraries
import pandas as pd
import warnings
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV,KFold,RandomizedSearchCV,StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# Ignore all warnings
warnings.filterwarnings('ignore')


# In[48]:


test=pd.read_csv('test.csv')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('pca.pkl','rb') as file:
    pca = pickle.load(file)


# In[49]:


# Feature Extractor #
features=['mealCGM','CGM spike duration (mins)','Normalized CGM','F1 (freq1 of CGM wave Fourier Transform)','P1 (power response for F1)'
          ,'F2 (freq2 of CGM wave Fourier Transform)','P2 (power response for F2)','d/dt (CGM wave)','d2/dt2 (CGM wave)']
X=pd.DataFrame(columns=features)

#mealCGM
X['mealCGM']=test.iloc[:,:3].mean(axis=1)

#CGM spike duration (mins)#
spike_col = pd.Series(np.argmax(test.values, axis=1))
spike_col_nums=spike_col.apply(lambda x: x*5)
X['CGM spike duration (mins)']=spike_col_nums

#Normalized CGM#
X['Normalized CGM']=X.apply(
    lambda row:(
        (test.iloc[:,row["CGM spike duration (mins)"]//5]-row['mealCGM'])/row['mealCGM']
    ).iloc[row.name]
,axis=1)
#Freqs and power responses for the cgm wave#

def compute_freq_power(sin_wave):
    fft_wave=np.fft.fft(sin_wave)
    fft_magnitude=np.abs(fft_wave)
    power_spectrum=fft_magnitude**2
    fft_freq=np.fft.fftfreq(len(sin_wave),d=5)
    positive_indices=fft_freq>0

    fft_magnitude=fft_magnitude[positive_indices]
    power_spectrum=power_spectrum[positive_indices]
    fft_freq=fft_freq[positive_indices]

    two_indices=np.argsort(power_spectrum)[-2:]
    fft_two_freq=fft_freq[two_indices]
    power_two_freq=power_spectrum[two_indices]
    freq1=fft_two_freq[1]
    freq2=fft_two_freq[0]
    power_response_1=power_two_freq[1]
    power_response_2=power_two_freq[0]
    return pd.Series({'Freq1': freq1,'Freq2':freq2, 'Power Response1': power_response_1,'Power Response2':power_response_2})

points=test.apply(lambda row: row.tolist(),axis=1)
sin_wave=points.apply(lambda row: np.sin(row))
#resample sine wave
def resample_wave(wave, num_points):
    original_len = len(wave)
    new_x = np.linspace(0, original_len - 1, num_points)
    original_x = np.arange(original_len)
    return np.interp(new_x, original_x, wave)

resampled_sin_wave = sin_wave.apply(lambda row: resample_wave(row, 24))
#normalize sine wave
def normalize_wave(wave):
    return (wave - np.min(wave)) / (np.max(wave) - np.min(wave))
normalized_sin_wave = resampled_sin_wave.apply(lambda row: normalize_wave(row))

freq_power=normalized_sin_wave.apply(lambda row: compute_freq_power(row))
X['F1 (freq1 of CGM wave Fourier Transform)']=freq_power['Freq1']
X['P1 (power response for F1)']=freq_power['Power Response1']
X['F2 (freq2 of CGM wave Fourier Transform)']=freq_power['Freq2']
X['P2 (power response for F2)']=freq_power['Power Response2']


#Derivatives of the cgm wave#
def compute_derivatives(row):
    points = row.values
    first_derivative = np.gradient(points)
    second_derivative = np.gradient(first_derivative)
    return pd.Series({'First Derivative': first_derivative, 'Second Derivative': second_derivative})
derivatives=test.apply(lambda row: compute_derivatives(row),axis=1)
X['d/dt (CGM wave)']=derivatives['First Derivative'].apply(lambda x: np.mean(x[0:2]))
X['d2/dt2 (CGM wave)']=derivatives['Second Derivative'].apply(lambda x: np.mean(x[0:2]))


# In[50]:


X_test_scaled=scaler.transform(X)
X_pca=pca.transform(X_test_scaled)
predictions=model.predict(X_pca)
results=pd.DataFrame(predictions)
results.to_csv('Result.csv',index=False)


# In[ ]:
