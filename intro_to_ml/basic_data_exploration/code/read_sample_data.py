# third party libraries
import pandas as pd
import datetime as dt
# set the sample data file path

melborne_housing_data_file_path = "../data/melb_data.csv"

melborne_housing_dataframe = pd.read_csv(melborne_housing_data_file_path)
print(round(melborne_housing_dataframe['Rooms'].mean()))