import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV file
data = pd.read_csv('iphone_price.csv')

# Plot the data
plt.scatter(data['version'], data['price'])

# Fit the Linear Regression model
model = LinearRegression()
model.fit(data[['version']], data['price'])

# Predict the price for a given version
version = 15
version_to_predict = [[version]]
price = model.predict(version_to_predict)[0]

# Print the predicted price
print(f'Iphone {version} price can be {price:.2f} USD')

# Display the plot
plt.show()
