import streamlit as st
import joblib
import pandas as pd
#load model
model = joblib.load('result.pkl')


st.title('House Price Prediction')
Rooms = st.number_input('Rooms', min_value=1, value=5)
Bedroom2 = st.number_input('Bedroom2', min_value=1, value=3)
Landsize = st.number_input('Landsize', min_value=0, value=180)
YearBuild = st.number_input('YearBuilt', min_value=1800, value=2023)
Bathroom = st.number_input('Bathroom', min_value=0, value=2)
Distance = st.number_input('Distance', min_value=0.0, value=10.5)
Postcode = st.number_input('Postcode', min_value=1000, value=2500)
BuildingArea = st.number_input('BuildingArea', min_value=0.0, value=100.00)


if st.button('Predict'):

    features = pd.DataFrame([{
        'Rooms': Rooms,
        'Bedroom2': Bedroom2,
        'Landsize': Landsize,
        'YearBuilt': YearBuild,
        'Bathroom': Bathroom,
        'Distance': Distance,
        'Postcode': Postcode,
        'BuildingArea': BuildingArea
    }])

    result = model.predict(features)
    st.success(f'Predicted House Price: ${result[0]:,.2f}')