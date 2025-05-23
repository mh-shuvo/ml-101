import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data_path = "../../data/melb_data.csv"
melb_data = pd.read_csv(data_path)
filterd_melb_data = melb_data.dropna(axis=0)

y = filterd_melb_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X=filterd_melb_data[features]

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)

# model = DecisionTreeRegressor(random_state=1)
# model.fit(train_X,train_y)
# prediction_result = model.predict(val_X)

# validation = mean_absolute_error(val_y,prediction_result)
# print(validation)


'''
Now we are going to figure the best leaf nodes from some user defined leaf nods. We want to see
how much will be the mean_absolute_error for different number of tree depth
here we will use sample leaf_nodes array and call get_mae() to get the mean_absolute_error for each of the item
------
 tree's depth is a measure of how many splits it makes before coming to a prediction. This is a relatively shallow tree
'''

def get_mae(max_leaf_nodes,train_X,train_y,val_X,val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1)
    model.fit(train_X,train_y)
    prediction_result = model.predict(val_X)

    mae_value = mean_absolute_error(val_y,prediction_result)
    return mae_value

leaf_nodes = [5,10,50,100,500,1000,10000]

for item in leaf_nodes:
    mae = get_mae(item,train_X,train_y,val_X,val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(item, mae))