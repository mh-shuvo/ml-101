import joblib
import pandas as pd
# Load the model
model = joblib.load('result.pkl')

# Prepare input as a DataFrame
features = pd.DataFrame([{
    'Rooms': 5,
    'Bedroom2': 3,
    'Landsize': 180,
    'YearBuilt': 2023,
    'Bathroom': 2,
    'Distance': 10.5,
    'Postcode': 2500,
    'BuildingArea': 100
}])

# Predict
result = model.predict(features)
print(result)
