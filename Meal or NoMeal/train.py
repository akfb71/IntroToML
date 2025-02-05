#!/usr/bin/env python
# coding: utf-8

# In[308]:


#libraries
import pandas as pd
import warnings
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV,KFold,RandomizedSearchCV,StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
# Ignore all warnings
warnings.filterwarnings('ignore')


# In[340]:


###Data Preprocessing###
df1=pd.read_csv('InsulinData.csv')
df2=pd.read_csv('Insulin_patient2.csv').drop(columns=['Unnamed: 0'])
df2['Date']=df2['Date'].str.split(' ').str[0]
df2['Date']=pd.to_datetime(df2['Date'],format='%Y-%m-%d')
df2['Date']=df2['Date'].dt.strftime('%m/%d/%Y')
insulin=pd.concat([df1,df2],axis=0,ignore_index=True)
insulin['Timestamp']=pd.to_datetime(insulin['Date']+" "+insulin['Time'])
insulin=insulin.sort_values(by='Timestamp')
df1=pd.read_csv('CGMData.csv')
df2=pd.read_csv('CGM_patient2.csv').drop(columns=['Unnamed: 0'])
df2['Date']=df2['Date'].str.split(' ').str[0]
df2['Date']=pd.to_datetime(df2['Date'],format='%Y-%m-%d')
df2['Date']=df2['Date'].dt.strftime('%m/%d/%Y')
cgm=pd.concat([df1,df2],axis=0,ignore_index=True)
cgm['Timestamp']=pd.to_datetime(cgm['Date']+' '+cgm['Time'])
cgm=cgm[['Timestamp','Sensor Glucose (mg/dL)']]
cgm=cgm.sort_values(by='Timestamp')
## Meal Data Extraction ##
meal_times=(insulin[insulin['BWZ Carb Input (grams)'].notna()&(insulin['BWZ Carb Input (grams)']!=0)])['Timestamp']
valid_meals_ins=[]
#select only valid meal timings
for tm in range(0,len(meal_times)-1):
    if meal_times.iloc[tm+1]-meal_times.iloc[tm] > pd.Timedelta('02:00:00'):
        valid_meals_ins.append(meal_times.iloc[tm])
columns_30mins=[f'Glucose_offset-{i}'for i in range(30,0,-5)]
columns_2hrs=[f'Glucose_offset+{i}' for i in range(0,120,5)]
columns=columns_30mins+columns_2hrs
## Meal Data Creation ##
meal_data=pd.DataFrame(columns=columns)
#time intervals which'll be cols of new df meal_data
offsets1=[-pd.Timedelta(f'00:{i}:00') for i in range(30,0,-5)]
offsets2=[pd.Timedelta(f'00:{i}:00') for i in range(0,121,5)]
offsets=offsets1+offsets2
for tm in valid_meals_ins:
    filtered_data=[]
    for i in range(len(offsets)-1):
        filtered=cgm[(cgm['Timestamp']>=tm+offsets[i]) & (cgm['Timestamp']<tm+offsets[i+1])]
        if not filtered.empty:
            filtered_data.append(filtered['Sensor Glucose (mg/dL)'].mean(axis=0))
        else:
            filtered_data.append(None)
    meal_data.loc[len(meal_data)]=filtered_data

# meal_data.head()


# In[341]:


## No Meal Data Extraction ##
valid_nomeals=[]
filtered_insulin=insulin
while not filtered_insulin.empty:
    current_time=filtered_insulin['Timestamp'].iloc[0]
    end_time=current_time+pd.Timedelta('2 hours')
    carbs_in_window=filtered_insulin[
        (filtered_insulin['Timestamp'] >= current_time)
        & (filtered_insulin['Timestamp'] < end_time)
        & (filtered_insulin['BWZ Carb Input (grams)'].notna()|(filtered_insulin['BWZ Carb Input (grams)']==0))
    ]
    if carbs_in_window.empty:
        valid_nomeals.append(current_time)
    filtered_insulin=filtered_insulin[
        ~((filtered_insulin['Timestamp'] >= current_time) & (filtered_insulin['Timestamp'] < end_time))
    ]
columns=[f'Glucose_offset+{i}' for i in range(0,120,5)]
nomeal_data=pd.DataFrame(columns=columns)
offsets=offsets2
for tm in valid_nomeals:
    filtered_data=[]
    for i in range(len(offsets)-1):
        filtered=cgm[(cgm['Timestamp']>=tm+offsets[i]) & (cgm['Timestamp']<tm+offsets[i+1])]
        if not filtered.empty:
            filtered_data.append(filtered['Sensor Glucose (mg/dL)'].mean(axis=0))
        else:
            filtered_data.append(None)
    nomeal_data.loc[len(nomeal_data)]=filtered_data

# nomeal_data.head()


# In[342]:


## Handling missing data ##

def fill_missing(df):
    for col in df.columns:
        next_values = df[col].shift(-1)
        prev_values = df[col].shift(1)
        temp_df = pd.DataFrame({'prev': prev_values, 'next': next_values})
        df[col] = df[col].fillna(temp_df.mean(axis=1))
    return df

threshold=.3

mealna_counts = meal_data.isna().sum(axis=1)
meal_data=meal_data[mealna_counts/30<=threshold]
meal_df=(fill_missing(meal_data)).dropna()
# meal_df=meal_data.dropna()
meal_df=meal_df.reset_index(drop=True)
class_size=len(meal_df)

nomealna_counts=nomeal_data.isna().sum(axis=1)
nomeal_data=nomeal_data[nomealna_counts/24<=threshold]
nomeal_df=(fill_missing(nomeal_data)).dropna()
# nomeal_df=nomeal_data.dropna()
nomeal_df = nomeal_df.sample(n=class_size, random_state=42)
nomeal_df=nomeal_df.reset_index(drop=True)


# print(len(meal_data))
# print(len(meal_df))
# print(len(nomeal_data))
# print(len(nomeal_df))


# In[343]:


##Feature Extraction##
features=['mealCGM','CGM spike duration (mins)','Normalized CGM','F1 (freq1 of CGM wave Fourier Transform)','P1 (power response for F1)'
          ,'F2 (freq2 of CGM wave Fourier Transform)','P2 (power response for F2)','d/dt (CGM wave)','d2/dt2 (CGM wave)']
meal=pd.DataFrame(columns=features)
nomeal=pd.DataFrame(columns=features)
#mealCGM
meal['mealCGM']=meal_df[['Glucose_offset-10','Glucose_offset-5','Glucose_offset+0','Glucose_offset+5','Glucose_offset+10']].mean(axis=1)
nomeal['mealCGM']=nomeal_df[['Glucose_offset+0','Glucose_offset+5','Glucose_offset+10']].mean(axis=1)

#CGM spike duration (mins)#
spike_col=meal_df.iloc[:,6:].idxmax(axis=1)
spike_col_nums=spike_col.apply(lambda x: pd.to_numeric(x.split('+')[-1]))
meal['CGM spike duration (mins)']=spike_col_nums
spike_col=nomeal_df.iloc[:].idxmax(axis=1)
spike_col_nums=spike_col.apply(lambda x: pd.to_numeric(x.split('+')[-1]))
nomeal['CGM spike duration (mins)']=spike_col_nums
#Normalized CGM#
meal['Normalized CGM']=meal.apply(
    lambda row:(
            (meal_df[f'Glucose_offset+{row["CGM spike duration (mins)"]}']-row['mealCGM'])/row['mealCGM']
    ).iloc[row.name]
,axis=1)
nomeal['Normalized CGM']=nomeal.apply(
    lambda row:(
            (nomeal_df[f'Glucose_offset+{row["CGM spike duration (mins)"]}']-row['mealCGM'])/row['mealCGM']
    ).iloc[row.name]
,axis=1)
# nomeal['Normalized CGM']=nomeal.apply(
#     lambda row:(
#             (nomeal_df[f'Glucose_offset+{row["CGM spike duration (mins)"]}']-nomeal_df['Glucose_offset+0'])/nomeal_df['Glucose_offset+0']
#     ).iloc[row.name]
# ,axis=1)
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

points=meal_df.apply(lambda row: row.tolist(),axis=1)
sin_wave_meal=points.apply(lambda row: np.sin(row))
points=nomeal_df.apply(lambda row: row.tolist(),axis=1)
sin_wave_nomeal=points.apply(lambda row: np.sin(row))
#resample sine wave
def resample_wave(wave, num_points):
    original_len = len(wave)
    new_x = np.linspace(0, original_len - 1, num_points)
    original_x = np.arange(original_len)
    return np.interp(new_x, original_x, wave)

resampled_sin_wave_meal = sin_wave_meal.apply(lambda row: resample_wave(row, 24))
resampled_sin_wave_nomeal = sin_wave_nomeal.apply(lambda row: resample_wave(row, 24))
#normalize sine wave
def normalize_wave(wave):
    return (wave - np.min(wave)) / (np.max(wave) - np.min(wave))
normalized_sin_wave_meal = resampled_sin_wave_meal.apply(lambda row: normalize_wave(row))
normalized_sin_wave_nomeal = resampled_sin_wave_nomeal.apply(lambda row: normalize_wave(row))

# print(normalized_sin_wave_meal)
# print(normalized_sin_wave_nomeal)
freq_power_meal=normalized_sin_wave_meal.apply(lambda row: compute_freq_power(row))
meal['F1 (freq1 of CGM wave Fourier Transform)']=freq_power_meal['Freq1']
meal['P1 (power response for F1)']=freq_power_meal['Power Response1']
meal['F2 (freq2 of CGM wave Fourier Transform)']=freq_power_meal['Freq2']
meal['P2 (power response for F2)']=freq_power_meal['Power Response2']


freq_power_nomeal=sin_wave_nomeal.apply(lambda row: compute_freq_power(row))
nomeal['F1 (freq1 of CGM wave Fourier Transform)']=freq_power_nomeal['Freq1']
nomeal['P1 (power response for F1)']=freq_power_nomeal['Power Response1']
nomeal['F2 (freq2 of CGM wave Fourier Transform)']=freq_power_nomeal['Freq2']
nomeal['P2 (power response for F2)']=freq_power_nomeal['Power Response2']

#Derivatives of the cgm wave#
def compute_derivatives(row):
    points = row.values
    first_derivative = np.gradient(points)
    second_derivative = np.gradient(first_derivative)
    return pd.Series({'First Derivative': first_derivative, 'Second Derivative': second_derivative})
derivatives_meal=meal_df.apply(lambda row: compute_derivatives(row),axis=1)
meal['d/dt (CGM wave)']=derivatives_meal['First Derivative'].apply(lambda x: np.mean(x[6:8]))
meal['d2/dt2 (CGM wave)']=derivatives_meal['Second Derivative'].apply(lambda x: np.mean(x[6:8]))
derivatives_nomeal=nomeal_df.apply(lambda row: compute_derivatives(row),axis=1)
nomeal['d/dt (CGM wave)']=derivatives_nomeal['First Derivative'].apply(lambda x: np.mean(x[0:2]))
nomeal['d2/dt2 (CGM wave)']=derivatives_nomeal['Second Derivative'].apply(lambda x: np.mean(x[0:2]))



#New column for meal/no meal#
meal['Meal Taken']=1
nomeal['Meal Taken']=0


# In[368]:





# In[386]:


## Training Model ##

## Splitting into training and testing data ##
train_meal,test_meal=train_test_split(meal,test_size=0.2,random_state=33)
train_nomeal,test_nomeal=train_test_split(nomeal,test_size=0.2,random_state=55)
train=pd.concat([train_meal,train_nomeal],axis=0).reset_index(drop=True)
test=pd.concat([test_meal,test_nomeal],axis=0).reset_index(drop=True)

X_train=train.drop(columns=['Meal Taken'])
y_train=train['Meal Taken']
X_test=test.drop(columns=['Meal Taken'])
y_test=test['Meal Taken']

#Decision Tree Classifier#
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_train)
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X_scaled)
model=DecisionTreeClassifier(random_state=78)
skf = StratifiedKFold(n_splits=5)
cv_scores=cross_val_score(model, X_pca, y_train, cv=skf)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')

param_grid={
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}
random_search=RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_search.fit(X_pca,y_train)
best_params = random_search.best_params_
best_score = random_search.best_score_
print("Best parameters found: ", best_params)
print("Best cross-validation score: {:.2f}".format(best_score))
best_model = random_search.best_estimator_
best_model.fit(X_pca,y_train)

# #SVM Classifier#
# #Linear -relatively few support vectors hence, not the best#
# scaler = StandardScaler()
# X=X_train
# X = scaler.fit_transform(X)
# pca = PCA(n_components=6)  # Retain 10 principal components
# X = pca.fit_transform(X)
# model=SVC(random_state=42)
# skf = StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model, X, y_train, cv=skf)
# print(f'Cross-validation scores: {cv_scores}')
# print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
# print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')
# param_grid = {
#     'C': [1,10,100],
#     'kernel': ['linear']
# }
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X, y_train)
# print(f'Best parameters: {grid_search.best_params_}')
# print(f'Best score: {grid_search.best_score_:.2f}')
# best_model=grid_search.best_estimator_
# best_model.fit(X,y_train)
# print(f'Number of support vectors: {len(best_model.support_vectors_)}')

#RBF - best model#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_train)
# pca = PCA(n_components=6)
# X = pca.fit_transform(X_scaled)
# model=SVC(random_state=42)
# skf = StratifiedKFold(n_splits=5)
# cv_scores=cross_val_score(model, X, y_train, cv=skf)
# print(f'Cross-validation scores: {cv_scores}')
# print(f'Mean cross-validation score: {cv_scores.mean():.2f}')
# print(f'Standard deviation of cross-validation scores: {cv_scores.std():.2f}')
# param_grid = {
#     'C': [10,20,100],
#     'gamma': [0.0005,0.001,0.1],
# #     'degree':[2,3,4],
#     'kernel': ['rbf']
# }
# random_search=RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
# random_search.fit(X,y_train)
# print(f'Best parameters: {random_search.best_params_}')
# print(f'Best score: {random_search.best_score_:.2f}')
# best_model=random_search.best_estimator_
# best_model.fit(X,y_train)
# print(f'Number of support vectors: {len(best_model.support_vectors_)}')


# In[387]:


## Evaluate Model ##
X_test_scaled=scaler.transform(X_test)
X_test_pca=pca.transform(X_test_scaled)
y_pred=best_model.predict(X_test_pca)
accuracy=accuracy_score(y_test,y_pred)
print(classification_report(y_test, y_pred))


# In[388]:


## Save the model to a pickle file ##
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('pca.pkl','wb') as file:
    pickle.dump(pca,file)


# In[133]:





# In[348]:


meal


# In[349]:


nomeal


# In[375]:



# Assuming X is your dataset (without the target variable)
# Standardizing the data (PCA assumes the data is centered/scaled)
X=X_train.iloc[:,1:]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=5)  # Adjust n_components based on how many PCs you want to keep
pca.fit(X_scaled)

# Explained variance ratio (importance of each PC)
explained_variance = pca.explained_variance_ratio_

# PCA Components (loadings)
pca_components = pca.components_

# Get feature names (if using a DataFrame)
features = X.columns if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]

# Create a DataFrame with PCA loadings
pca_df = pd.DataFrame(pca_components.T, index=features, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

print("Explained Variance Ratio for each PC:")
print(explained_variance)

print("\nPCA Components (Loadings):")
print(pca_df)

# To see the most important features for each principal component
# Sort by absolute value for each principal component
most_important_features = pd.DataFrame()
for i in range(pca.n_components_):
    most_important_features[f"PC{i+1}"] = pca_df[f"PC{i+1}"].abs().sort_values(ascending=False).index

print("\nMost Important Features for each PC:")
print(most_important_features)

# Visualize explained variance (optional)
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, align='center')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.show()


# In[ ]:
