# ML-homework
## Credit Risk Resampling Techniques
### import 
```
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

### Read the CSV into DataFrame
```
file_path = Path('Resources/lending_data.csv')
df = pd.read_csv(file_path)
df.head()
```
pic

### Split the Data into Training and Testing
```
X = df.drop(columns="loan_status")
X = pd.get_dummies(X, columns=["homeowner"])
X.head()
# Create our target
y = df['loan_status']
```
pic2

### Data Pre-Processing
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

### Simple Logistic Regression
pic

## Oversampling
### Naive Random Oversampling

```
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```
Counter({'low_risk': 56271, 'high_risk': 56271})

```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```
LogisticRegression(random_state=1)

```
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
```
0.9520479254722232

```
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
```
array([[  615,     4],
       [  116, 18649]], dtype=int64)
       
```
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
```
pic

### SMOTE Oversampling
```
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
    X_train, y_train
)
from collections import Counter

Counter(y_resampled)
```
Counter({'low_risk': 56271, 'high_risk': 56271})
```
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```
LogisticRegression(random_state=1)
```
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```
0.9936781215845847
```
confusion_matrix(y_test, y_pred)
```
array([[  615,     4],
       [  116, 18649]], dtype=int64)
```
print(classification_report_imbalanced(y_test, y_pred))
```
pic

### Undersampling

```
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

Counter(y_resampled)
```
Counter({'high_risk': 1881, 'low_risk': 1881})
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```
LogisticRegression(random_state=1)
```
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
```
0.9936781215845847
```
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
```
array([[  606,    13],
       [  112, 18653]], dtype=int64)
```
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```
pic

### Combination (Over and Under) Sampling
```
from imblearn.combine import SMOTEENN

sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
Counter(y_resampled)
```
Counter({'high_risk': 55377, 'low_risk': 55937})
```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```
LogisticRegression(random_state=1)
```
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
```
0.9865149130022852

```
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)
```
array([[  615,     4],
       [  122, 18643]], dtype=int64)
```
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))
```
pic

### Final Questions

1. Which model had the best balanced accuracy score?<b\ >

Answer: SMOTE Oversampling model has the highest accuracy rate

2. Which model had the best recall score?

Answer: all the models have 0.99 recall rate.

3. Which model had the best geometric mean score?

Answer: most of them has 0.99 beside simple logistic regression



