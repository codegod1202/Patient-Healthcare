#Importing Libraries
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
       
import numpy as np # linear algebra
import pandas as pd # data processing

import seaborn as sns # checking data balancing(heat map)
import plotly.express as px # Data Visualization
import matplotlib.pyplot as plt # heat map figure size

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split # for splitting
from sklearn.metrics import f1_score # evaluation metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # for visualising confusion matrix
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

#Loading Train Dataset
train_data = pd.read_csv('/kaggle/input/patient-healthcare/train.csv')
train_data.head(10)

train_data.shape

#Data Processing
train_data.isnull().sum()
train_data.info()
numerical_cols = train_data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
print("List of Numerical features: \n" , numerical_cols)
print("\n\nList of Categorical features: \n" , categorical_cols)

#Filled null values of numerical data with mean and categorical data with mode
for col in numerical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mean())
for col in categorical_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    
train_data.isnull().sum() 
num = LabelEncoder()

label_col = ['ethnicity', 'icu_admit_source', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'icu_type', 'icu_stay_type']

for x in label_col:
    train_data[x] = num.fit_transform(train_data[x].astype(str))

train_data.head(10)
train_data = pd.get_dummies(train_data, columns=['gender'], prefix=['gender'])
train_data.head()
train_data.shape

train_data = train_data.drop(columns=['encounter_id', 'patient_id', 'hospital_id', 'height', 'weight'])
x_train = train_data.drop('has_died',axis = 1)

# Target variable column
y_train = train_data['has_died']

#Exploratory Data Analysis
plt.figure(figsize=(20,20))
_=sns.heatmap(train_data.corr())

for i in train_data.columns[3::]:
    plt.figure(figsize=(12,8))
    sns.distplot(train_data[i][train_data['has_died']==0],color='y', label='Survive ',hist_kws={'edgecolor':'black'})
    sns.distplot(train_data[i][train_data['has_died']==1],color='r',label='Death',hist_kws={'edgecolor':'black'})
    plt.legend()
    plt.show()

sscaler = StandardScaler()
x_train = sscaler.fit_transform(x_train)
model = Sequential()
model.add(Dense(64,input_dim=x_train.shape[1],activation='relu'))
model.add(Dense(32,activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(16,activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(8,activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid',kernel_regularizer=l2(0.001)))    

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy",Recall(),Precision()])
model.fit(x_train,y_train,epochs=100,callbacks=[early_stopping],validation_batch_size=0.1,verbose=2)

test_data = pd.read_csv('/kaggle/input/patient-healthcare/test.csv')
test_data.head(10)

test_data.shape
test_data.isnull().sum()

numerical_cols = test_data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = test_data.select_dtypes(include=['object']).columns.tolist()

for col in numerical_cols:
    test_data[col] = test_data[col].fillna(test_data[col].mean())
for col in categorical_cols:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
num = LabelEncoder()

label_col = ['ethnicity', 'icu_admit_source', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'icu_type', 'icu_stay_type']

for x in label_col:
    test_data[x] = num.fit_transform(test_data[x].astype(str))

test_data.head(10)

test_data = pd.get_dummies(test_data, columns=['gender'], prefix=['gender'])
test_data.head()

patient_id = test_data['patient_id']
test_data = test_data.drop(columns=['encounter_id', 'patient_id', 'hospital_id', 'height', 'weight'])
sscaler = StandardScaler()
test_data = sscaler.fit_transform(test_data)

predictions = []

for i in model.predict(test_data):
    if i < 0.5:
        predictions.append(0)
    else:
        predictions.append(1)
len(predictions)

data = pd.DataFrame()
data['patient_id'] = patient_id
data['has_died'] = predictions
data.head()

data.to_csv('submission.csv',index=False)
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = data

# create a link to download the dataframe
create_download_link(df)
