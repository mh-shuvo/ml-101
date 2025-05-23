from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
y = lung_cancer_data['LUNG_CANCER']

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

model = DecisionTreeClassifier(random_state=1)
model.fit(train_X,train_y)
prediction_result = model.predict(val_X)
mae_value = accuracy_score(val_y,prediction_result)
print(mae_value)