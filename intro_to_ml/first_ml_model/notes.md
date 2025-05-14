# First Machine Learning Models

### Selecting Data for Modeling
Your dataset had too many variables to wrap your head around, or even to print out nicely. How can you pare down this overwhelming amount of data to something you can understand?

We'll start by picking a few variables using our intuition. Later courses will show you statistical techniques to automatically prioritize variables.
To choose variables/columns, we'll need to see a list of all columns in the dataset. That is done with the columns property of the DataFrame (the bottom line of code below).

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```

When drop the missing data it means we are creating a subset of the original dataset. There are many way to create subset but right now we will focus on two things
* Dot notation, which we use to select the **Prediction Target**
* Selecting with a column list, which we use to select the **features**

#### Prediction Target
We'll use the dot notation to select the column we want to predict, which is called the **prediction target**. By convention, the **prediction target** is called y. So the code we need to save the house prices in the Melbourne data is

```python
y = melbourne_data.Price
```
#### Selecting features
The columns that are inputted into our model (and later used to make predictions) are called "**features**." In our case, those would be the columns used to determine the home price. Sometimes, you will use all columns except the target as **features**. Other times you'll be better off with fewer **features**.

```python
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
```

## Building Model
You will use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.

The steps to building and using a model are:

* **Define**: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
* **Fit**: Capture patterns from provided data. This is the heart of modeling.
* **Predict**: Just what it sounds like
* **Evaluate**: Determine how accurate the model's predictions are.

```python

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

```