#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello Shuvo")


# In[16]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[17]:


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LogisticRegression model
LR = LogisticRegression(max_iter=1000)

# Train the model
LR.fit(X_train, y_train)

# Make predictions
y_predict = LR.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:",accuracy)


# In[ ]:




