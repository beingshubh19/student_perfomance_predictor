import pickle
import numpy as np


with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)


def get_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if value < min_val or value > max_val:
                print(f" Enter value between {min_val} and {max_val}")
            else:
                return value
        except:
            print(" Invalid input, try again")


hours = get_input("Enter study hours (0–12): ", 0, 12)
attendance = get_input("Enter attendance (0–100): ", 0, 100)
sleep = get_input("Enter sleep hours (0–12): ", 0, 12)
prev = get_input("Enter previous score (0–100): ", 0, 100)

input_data = np.array([[hours, attendance, sleep, prev]])
prediction = model.predict(input_data)

print("Predicted Score:", prediction[0])
