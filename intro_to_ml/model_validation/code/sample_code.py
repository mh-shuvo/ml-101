import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
melbourne_data_path = "../../data/melb_data.csv"

melbourne_data = pd.read_csv(melbourne_data_path)
melbourne_data = melbourne_data.dropna(axis=0)

prediction_target = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
features = melbourne_data[melbourne_features]

model = DecisionTreeRegressor(random_state=1)
model.fit(features,prediction_target)
print("Making predictions for the following 5 houses:")
print(features.head())
print("The predictions are")
predicted_result = model.predict(features.head())
print(predicted_result)
error = mean_absolute_error(prediction_target.head(), predicted_result)
print("Absolute mean error is: ",error)
