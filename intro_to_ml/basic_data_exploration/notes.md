# Basic Data Exploration
### Pandas
First of all we have to use  Pandas to Get Familiar With Your Data. Pandash is the primary tool for **data scientists** to explore and manipulate data.
### DataFrame
The most important part of the Pandas library is the DataFrame. A DataFrame holds the type of data you might think of as a table. This is similar to a sheet in Excel, or a table in a SQL database.
### Interpreting Data Description

if you run the following code then the results shown 8 numbers for each column in your original datasets. I will explain these 8 numbers

**count**:  how many rows have non-missing values.

**mean** :  average

**std**  :  standard deviation, which measures how numerically spread out the values are.

**min**  :  The smallest value in the dataset. Think of it as the lowest score, price, or measurement recorded. 

**25%**  :  - This is the value where 25% of the data falls below it. If the dataset were sorted from smallest to largest, this number would mark the point where the first quarter of the values end.

**50%**  :   The middle value of the dataset. Half the values are smaller, and half are larger. It gives a better idea of the "typical" value than the mean, especially if there are extreme values in the dataset.

**75%**  :   (Third Quartile): The point where 75% of the values fall below it. It's like saying, "Most of the data is below this number."

**max**  :   The highest value in the dataset—the peak number observed.
These statistics help understand how the data is spread out and where most values lie. If you'd like a practical example to make it even clearer, let me know!


### How to access columns for specific group?

```python
import pandas as pd

# Path of the file to read
iowa_file_path = './data/home-data-for-train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics in next line
print(home_data.describe())
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data['LotArea'].mean())
# As of today, how old is the newest home (current year - the date in which it was built)
import datetime as dt
newest_home_age = (dt.datetime.now().year - home_data['YearBuilt'].max())


```