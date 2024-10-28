import pandas as pd  
import numpy as np  
from scipy import stats  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
    
df = pd.read_csv('fraudulent.csv')  
   
mode_dict = df.mode().iloc[0].to_dict()  
  
def fill_missing_with_mode(column):  
    mode_value = mode_dict[column.name]  
    return column.fillna(mode_value)  
  
df = df.apply(fill_missing_with_mode)  

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1_weighted = f1_score(y_test, y_pred, average='weighted')  
print(f'Weighted F1 Score: {f1_weighted:.2f}')