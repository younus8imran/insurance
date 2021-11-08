import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/insurance.csv')

X = data.drop(['charges'], axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

num_cols = make_column_selector(dtype_include='number')
cat_cols = make_column_selector(dtype_exclude='number')

preprocessor = make_column_transformer(
    (make_pipeline(scaler), num_cols),
    (make_pipeline(ohe), cat_cols)
)

lr = LinearRegression()

pipe = make_pipeline(preprocessor, lr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

#mean_squared_error(y_test, y_pred, squared=False)

with open('insurance_lr_pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)
