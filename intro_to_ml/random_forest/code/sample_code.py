from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
# import kagglehub

# path = kagglehub.dataset_download("aagambshah/lung-cancer-dataset")

data_path = "../../data/lung-cancer.csv"
lung_cancer_data = pd.read_csv(data_path)
lung_cancer_data = lung_cancer_data.dropna(axis=0)
# print(lung_cancer_data.columns)
features = [
    'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    'CHRONIC DISEASE', 'WHEEZING', 'ALCOHOL CONSUMING',
    ]
X = lung_cancer_data[features]

lung_cancer_data['LUNG_CANCER'] = lung_cancer_data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

y = lung_cancer_data['LUNG_CANCER']

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))