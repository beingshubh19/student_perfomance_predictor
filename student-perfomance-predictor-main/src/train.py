import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

current_dir = os.path.dirname(__file__)

data_path = os.path.join(current_dir, '..', 'dataset', 'student_data.csv')
data = pd.read_csv(data_path)

X = data[['hours_studied', 'attendance', 'sleep_hours', 'previous_score']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Model trained!")
print("MSE:", mse)

model_dir = os.path.join(current_dir, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("Model saved at:", model_path)