
# 🏡 House Price Prediction – Mini ML Project

This is a simple machine learning project that predicts house prices based on features like number of rooms, bathroom count, land size, and more. It uses the **Random Forest Regressor** model from `scikit-learn`.

---

## 📁 Project Structure

```
house_price_project/
│
├── data/
│   └── melb_data.csv          # Input dataset (Melbourne housing data)
│
├── main.py                    # Script to make prediction from trained model
├── train_model.py             # Trains the ML model and saves it as a .pkl file
├── result.pkl                 # Saved model after training
├── README.md                  # You're reading this!
```

---

## 📦 Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib

Install the required packages using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn’t exist yet, here's a quick setup:

```bash
pip install pandas scikit-learn joblib
```

---

## 🚀 How It Works

### Step 1: Training the Model (`train_model.py`)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load and clean the data
houses = pd.read_csv("./data/melb_data.csv")
houses.dropna(inplace=True)

# Select features and target
features = ['Rooms','Bedroom2','Landsize','YearBuilt','Bathroom','Distance','Postcode','BuildingArea']
X = houses[features]
y = houses['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the trained model
joblib.dump(model, "result.pkl")
```

### Step 2: Predict Using Saved Model (`main.py`)

```python
import pandas as pd
import joblib

# Load the model
model = joblib.load('result.pkl')

# Example input (change as needed)
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

# Predict the price
result = model.predict(features)
print("Predicted Price:", result)
```

---

## 📊 Model Performance

- **Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors.
- **R² Score:** Indicates how well the model explains the variance in the target. Closer to `1.0` means better.

Example output:

```
MAE: 18027.89
R2 Score: 0.734
```

---

## 🧠 What You Learn from This Project

- Data loading and cleaning with pandas
- Feature selection
- Model training using Random Forest
- Evaluating regression performance
- Saving/loading models using `joblib`
- Making predictions on real data

---

## 🛠️ Next Steps

You can extend this project by:
- Using more features (like suburb, type, region)
- Applying feature scaling or encoding categorical features
- Comparing multiple regression models
- Building a web UI with Streamlit

---

## 📚 Dataset Source

Melbourne housing dataset from Kaggle or open government portals.

---

## ✅ Author

Mohammad Mehedi Hasan  
> 6+ years of professional software development experience  
> Now exploring Machine Learning step by step 🚀
