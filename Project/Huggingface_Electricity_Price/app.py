import gradio as gr
import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import json
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta, date
from pandas import json_normalize
import tensorflow as tf
from keras.layers import LSTM
from urllib.request import urlopen
from sklearn.preprocessing import LabelEncoder, StandardScaler


# from keras.layers import *
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("SE3_elec_price_model", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/electricity_price.pkl")

def get_price_forecast():
    today, tomorrow = get_date()

    df_entsoe = get_entsoe_data(today, tomorrow)

    # get timestamps that temp dataset should match on
    entsoe_earliest = df_entsoe["datetime"].iloc[0]
    entsoe_latest = df_entsoe["datetime"].iloc[-1]
    df_temp = get_temp(entsoe_earliest, entsoe_latest)
    
    df = df_entsoe.merge(df_temp, how='inner', on='datetime')
    df.set_index('datetime',inplace=True)

    ## pre-process before predict
    sc_x=StandardScaler()
    df_scaled=sc_x.fit_transform(df)
    sc_y=StandardScaler()
    sc_y=sc_y.fit(df[['day_ahead_price']])

    step_back=24
    no_records=len(df_scaled)
    no_cols=4
    X_train_shape_pred=[]
    for i in range(step_back,no_records):
        X_train_shape_pred.append(df_scaled[i-step_back:i])
    X_train_shape_pred=np.array(X_train_shape_pred)
    print(X_train_shape_pred.shape)

    ## predict
    pred_price = model.predict(X_train_shape_pred)
    final_pred=sc_y.inverse_transform(pred_price)
    print(final_pred.shape)

    # append time for prediction
    predict_time_from = datetime.fromtimestamp(entsoe_latest / 1e3)
    # calculating timestamps for the next 24 h
    timestamp_list = [predict_time_from + timedelta(hours=x) for x in range(len(final_pred))]
    
    # iterating through timestamp_list
    # for i, x in enumerate(timestamp_list):
    #     print(x, final_pred[i])
    # print(final_pred.shape)
    df_prediction = pd.DataFrame(
        {'Datetime': timestamp_list,
        'Price forecast [EUR/MWh]': final_pred.flatten(),
        })
    
    #df_predictions = pd.DataFrame([timestamp_list, final_pred], columns=["datetime", "Price prediction"])
    # print(len(final_pred), len(timestamp_list), len(final_pred), len(final_pred[0]))

    return df_prediction
    #[today, temp, day_ahead_price, pred_price, total_load, total_generation]

# # Returns yesterday and tomorrows date
def get_date():
    # yesterday = datetime.today() - timedelta(days=1)
    # yesterday = yesterday.date().strftime('%Y%m%d')
    # tomorrow = (datetime.strptime(yesterday, '%Y%m%d') + timedelta(days=2)).strftime('%Y%m%d')

    date_from = datetime.now() - timedelta(days=3)
    date_from = date_from.date().strftime('%Y%m%d')
    date_to = (datetime.strptime(date_from, '%Y%m%d') + timedelta(days=4)).strftime('%Y%m%d')

    return date_from, date_to

def get_entsoe_data(date_from, date_to):
    # Client
    client = EntsoePandasClient(api_key="cb3a29b2-3276-4a4c-aba3-6507120d99be")

    # Date and country
    start = pd.Timestamp(date_from, tz='Europe/Stockholm')
    end = pd.Timestamp(date_to, tz='Europe/Stockholm')
    country_code = 'SE_3'  

    df_day_price = client.query_day_ahead_prices(country_code, start=start,end=end)
    df_generation_per_prod = client.query_generation(country_code, start=start,end=end, psr_type=None)    
    df_load = client.query_load(country_code, start=start,end=end)

    df_entsoe = df_generation_per_prod.join(df_day_price.rename("day_ahead_price"))
    df_entsoe = df_entsoe.join(df_load)

    df_entsoe_clean = df_entsoe.reset_index()
    df_entsoe_clean = df_entsoe_clean.rename(columns = {'index':'DateTime'})
    df_entsoe_clean['DateTime'] = df_entsoe_clean.DateTime.values.astype('int64') // 10 ** 6

    col_list = ["Hydro Water Reservoir", "Nuclear", "Other", "Solar", "Wind Onshore"]
    df_entsoe_clean['total_generation'] = df_entsoe_clean[list(col_list)].sum(axis=1)

    df_entsoe_clean.drop(col_list + ["Fossil Gas"], axis=1, inplace=True)
    df_entsoe_clean.rename(columns={"Actual Load": "total_load", "DateTime":"datetime"}, inplace=True)

    return df_entsoe_clean.tail(48)

def get_temp(timeseries_from, timeseries_to):

    url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/1/station/71420/period/latest-months/data.json"
    response = urlopen(url)

    # convert response to json, to dataframe
    data_json = json.loads(response.read())
    df_smhi_data = json_normalize(data_json['value']) 

    # extract only the temperature in the time stamp interval
    df_smhi_data = df_smhi_data.loc[(df_smhi_data['date'] >= timeseries_from) & (df_smhi_data['date'] <= timeseries_to)]
    df_smhi_data = df_smhi_data.reset_index().rename(columns = {'date':'datetime'})

    df_smhi_data.drop(["index", "quality"], axis=1, inplace=True)
    df_smhi_data["value"] = df_smhi_data["value"].astype(float)
    df_smhi_data.rename(columns={"value": "temperature"}, inplace=True)

    return df_smhi_data

demo = gr.Interface(
    fn = get_price_forecast,
    title = "SE3 Electricity Day-Ahead Price Prediction",
    description ="SE3 Electricity Day-Ahead Price Prediction, based on electricity production, generation and temperature",
    allow_flagging = "never",
    inputs = [],
    outputs = [
        gr.DataFrame(x="datetime", y="Price prediction [EUR/MWh]")
        # gr.Textbox(label="Date"),
        # gr.Textbox(label="Temperature Forecast [℃]"),
        # gr.Textbox(label="Total Load Forecast [MWh]"),
        # gr.Textbox(label="Total Generation Forecast [MWh]"),
        # gr.Textbox(label="Predicted Day-Ahead Price [EUR/MWh]"),   
    ]
)

demo.launch()