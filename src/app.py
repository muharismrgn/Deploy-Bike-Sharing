import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt
from PIL import Image
from preprocessing import *
from pycaret.regression import *

# Load the model
model = load_model('model/CatBoostModel')

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def predict(model, input_df):
    prediciton_df = predict_model(estimator=model, data=input_df)
    predictions = prediciton_df['prediction_label']
    return predictions

def main():
    # Load picture
    image_side = Image.open('../img/capital-bike-share.jpg')

    # Add option to select online or offline prediction
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
        )

    # Add explanatory text and picture in the sidebar
    st.sidebar.info('This app is created to predict bike sharing demand')    
    st.sidebar.image(image_side)

    # Add title
    st.title("Bike Sharing Demand Prediction")

    if add_selectbox == 'Online':

        # Set up the form to fill in the required data 
        dteday = st.date_input(
            'Date (DD-MM-YYYY)', 
            min_value=dt.date(2011, 1, 1), 
            max_value=dt.date(2012, 12, 31), 
            value=dt.date(2011, 1, 1),
            format="DD-MM-YYYY",
        )

        hr = st.number_input(
            'Hour', min_value=0, max_value=23, value=0)
        
        holiday_choice = {
            0: 'No', 
            1: 'Yes',
        }
        holiday = st.selectbox(
            "Holiday", 
            holiday_choice.keys(), 
            format_func=lambda x: holiday_choice[x],
            )

        season = st.selectbox(
            "Season", ['Winter', 'Spring', 'Summer', 'Fall'])
        if season == 'Winter':
            season = 1
        elif season == 'Spring':
            season = 2
        elif season == 'Summer':
            season = 3
        elif season == 'Fall':
            season = 4

        weathersit = st.selectbox(
            "Weathersit", ['Clear', 'Mist', 'Light Rain/Light Snow', 'Heavy Rain/Snow'])
        if weathersit == 'Clear':
            weathersit = 1
        elif weathersit == 'Mist':
            weathersit = 2
        elif weathersit == 'Light Rain/Light Snow':
            weathersit = 3
        elif weathersit == 'Heavy Rain/Snow':
            weathersit = 4

        hum = st.number_input(
            'Humidity', min_value=0, max_value=100, value=0)

        temp = st.number_input(
            'Temprature', min_value=-8, max_value=39, value=0)
        
        windspeed = st.number_input(
            'Windpeed', min_value=0, max_value=67, value=0)
        
        # Convert form to data frame
        temp_min, temp_max = -8, 39
        
        # Set a variabel to store the output
        output = ""

        input_df = pd.DataFrame([
                {
                    'dteday': dteday,
                    'hr': hr,
                    'holiday': holiday,
                    'season': season,
                    'weathersit': weathersit,
                    'hum': hum/100,
                    'atemp': 0,
                    'temp': (temp-temp_min)/(temp_max-temp_min),
                    'windspeed': windspeed/67,
                    'casual': 0,
                    'registered': 0,
                }
            ]
        )

        input_df['dteday'] = pd.to_datetime(input_df['dteday'])
        input_df = model[0: 2].transform(input_df)
        input_df.rename(columns={
            'feature 4': 'dteday_day',
            'feature 5': 'dteday_month',
            'feature 6': 'dteday_year',  
            'feature 7': 'hr', 
            'feature 8': 'holiday', 
            'feature 9': 'season', 
            'feature 10': 'weathersit',
            'feature 11': 'hum',
            'feature 12': 'temp',
            'feature 13': 'windspeed'
        }, inplace=True)

        # Make a prediction 
        if st.button("Predict"):
            # st.write(input_df)
            result = np.round(model[-1].predict(input_df)[0], 0)
            output = f"Bikes Demand Prediction Result : {result}"

        # Show prediction result
        st.success(output)    

    if add_selectbox == 'Batch':

        # Add a feature to upload the file to be predicted
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            # Convert the file to data frame
            data = pd.read_csv(file_upload)

            # Select only columns required by the model
            data = data[[
                'dteday',
                'hr',
                'holiday',
                'season',
                'weathersit',
                'hum',
                'temp',
                'windspeed',
                'atemp',
                'casual',
                'registered'
                ]
            ]
            # Convert into datetime
            data['dteday'] = pd.to_datetime(data['dteday'])

            # Make predictions
            data['DemandPrediction'] = predict(model, data)

            # Show the result on page
            st.write(data)

            # Add a button to download the prediction result file 
            st.download_button(
                "Press to Download",
                convert_df(data),
                "Prediction Result.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == '__main__':
    main()
