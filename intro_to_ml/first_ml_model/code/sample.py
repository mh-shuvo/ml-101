import pandas as pd
from sklearn.tree import DecisionTreeRegressor
melbourne_data_path = "../../data/melb_data.csv"

melbourne_data = pd.read_csv(melbourne_data_path)

prediction_target = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
features = melbourne_data[melbourne_features]

model = DecisionTreeRegressor(random_state=1)
model.fit(features,prediction_target)
print("Making predictions for the following 5 houses:")
print(features.head())
print("The predictions are")
print(model.predict(features.head()))