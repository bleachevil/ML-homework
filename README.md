# ML-homework
## Credit Risk Resampling Techniques
### initial import 
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
![](https://github.com/bleachevil/ML-homework/blob/main/pic/readcsvdatasampling.png?raw=true)

### Split the Data into Training and Testing
```
X = df.drop(columns="loan_status")
X = pd.get_dummies(X, columns=["homeowner"])
X.head()
# Create our target
y = df['loan_status']
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/splitdatatestingandtraining.png?raw=true)
![](https://github.com/bleachevil/ML-homework/blob/main/pic/splitdatatestingandtraining2.png?raw=true)

### Data Pre-Processing
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

### Simple Logistic Regression
![](https://github.com/bleachevil/ML-homework/blob/main/pic/simplelogisticRegression.png?raw=true)

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
![](https://github.com/bleachevil/ML-homework/blob/main/pic/nativeramdonoversampling.png?raw=true)

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
![](https://github.com/bleachevil/ML-homework/blob/main/pic/smoteoversampling.png?raw=true)

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
![](https://github.com/bleachevil/ML-homework/blob/main/pic/undersampling.png?raw=true)

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
![](https://github.com/bleachevil/ML-homework/blob/main/pic/combinationundersampling.png?raw=true)

### Final Questions

1. Which model had the best balanced accuracy score?<br />

Answer: SMOTE Oversampling model has the highest accuracy rate. <br />

2. Which model had the best recall score?<br />

Answer: all the models have 0.99 recall rate.<br />

3. Which model had the best geometric mean score?<br />

Answer: most of them has 0.99 beside simple logistic regression. <br />


## Ensemble Learning

### inital imports
```
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
```
### Read the CSV and Perform Basic Data Cleaning
```
file_path = Path('Resources/LoanStats_2019Q1.csv')
df = pd.read_csv(file_path)
df.head()
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/readcsvdatalearning.png?raw=true)

### Split the Data into Training and Testing
```
X = df.drop(columns="loan_status")
X = pd.get_dummies(X, columns=["home_ownership","verification_status","issue_d","pymnt_plan","initial_list_status","hardship_flag","debt_settlement_flag","next_pymnt_d","application_type"])
# Create our target
y = df['loan_status']
X.describe()
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/Xdescribe.png?raw=true)

### Data Pre-Processing
```
from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

## Ensemble Learners
### Balanced Random Forest Classifier
```
from imblearn.ensemble import BalancedRandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=78)
rf_model = rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
```
0.6781593559262381
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```
array([[   36,    65],
       [    2, 17102]], dtype=int64)
```
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/balancedrandmforest.png?raw=true)
```
importances = rf_model.feature_importances_
importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
importances_sorted[:10]
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/top10.png?raw=true)

### Easy Ensemble Classifier
```
from imblearn.ensemble import EasyEnsembleClassifier
rf_model = EasyEnsembleClassifier(random_state=42)
rf_model = rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test, y_pred)
```
0.9201411400494586
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```
array([[   90,    11],
       [  869, 16235]], dtype=int64)
```
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```
![](https://github.com/bleachevil/ML-homework/blob/main/pic/Easyensembleclassifier.png?raw=true)

### Final Questions

1.Which model had the best balanced accuracy score?<br />

Answer: Easy ensemble model has best balanced accuacy score.<br />

2.Which model had the best recall score?<br />

Answer: Balanced Random Forest Classifier has best recall score.<br />

3. Which model had the best geometric mean score?<br />

Answer: Easy ensemble model has the best geometric mean score.<br />

4.What are the top three features?

Answer: last_pymnt_amnt, total_rec_int,total_pymnt_inv are the top three

