import pandas as pd
import numpy as np

data = pd.read_csv('../input/patient-healthcare/train.csv')
test_data = pd.read_csv('../input/patient-healthcare/test.csv')

data.head()

data.info()

patient_id = data['patient_id']
data = data.drop('patient_id', axis=1)

test_patient_id = test_data['patient_id']
test_data = test_data.drop('patient_id', axis=1)

y = data['has_died']
data = data.drop('has_died', axis=1)

float_cols = data.select_dtypes('float64')
int_cols = data.select_dtypes('int64')
obj_cols = data.select_dtypes('object')

for col in float_cols:
    if data[col].isnull().sum() > 0:
        mean = data[col].describe()['mean']
        data[col] = data[col].fillna(mean)
        test_data[col] = test_data[col].fillna(mean)
        
for col in int_cols:
    if data[col].isnull().sum() > 0:
        mean = data[col].describe()['mean']
        data[col] = data[col].fillna(mean)
        test_data[col] = test_data[col].fillna(mean)
        
for col in obj_cols:
    if data[col].isnull().sum() > 0:
        mostFrequent = data[col].describe()['top']
        data[col] = data[col].fillna(mostFrequent)
        test_data[col] = test_data[col].fillna(mostFrequent)

data.isnull().sum()
test_data.isnull().sum()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[float_cols.columns] = scaler.fit_transform(data[float_cols.columns])
test_data[float_cols.columns] = scaler.transform(test_data[float_cols.columns])
data[int_cols.columns] = scaler.fit_transform(data[int_cols.columns])
test_data[int_cols.columns] = scaler.transform(test_data[int_cols.columns])

data.head()
test_data.head()

import catboost as cb
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(data, y, test_size=0.1, random_state=42)
train_dataset = cb.Pool(X_train, y_train, cat_features=list(obj_cols.columns))
valid_dataset = cb.Pool(X_valid, y_valid, cat_features=list(obj_cols.columns))

model = cb.CatBoostClassifier(verbose=False)
model.fit(train_dataset)

test_preds = model.predict(test_data)
test_patient_id = np.array(test_patient_id)

submission = pd.DataFrame(data=[test_patient_id, test_preds]).T
submission.columns = ['patient_id', 'has_died']
submission.set_index('patient_id', inplace=True)
submission.head(100)

submission.to_csv('submission.csv')
