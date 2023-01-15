# Modal script for future daily features
# For retrieving daily data through scheduled scripts, Modal is used in which the following function is uploaded and sheduled to run on hourly basis

import os
import modal

def get_daily_features():
    import hopsworks
    import pandas as pd
    from datetime import datetime, timedelta, date
    from entsoe import EntsoePandasClient
    import pandas as pd
    import json
    from urllib.request import urlopen
    from pandas import json_normalize

    # Get yesterdays date
    date_from = datetime.now() - timedelta(days=1)
    date_from = date_from.date().strftime('%Y%m%d')
    date_to = (datetime.strptime(date_from, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')

    # get eletricity data
    client = EntsoePandasClient(api_key="cb3a29b2-3276-4a4c-aba3-6507120d99be")

    start = pd.Timestamp(date_from, tz='Europe/Stockholm')
    end = pd.Timestamp(date_to, tz='Europe/Stockholm')
    country_code = 'SE_3'  

    df_day_price = client.query_day_ahead_prices(country_code, start=start,end=end)
    df_generation_per_prod = client.query_generation(country_code, start=start,end=end, psr_type=None)
    df_load = client.query_load(country_code, start=start,end=end)
    
    df_eletricity = df_generation_per_prod.join(df_day_price.rename("day_ahead_price"))
    df_eletricity = df_eletricity.join(df_load)
    df_eletricity = df_eletricity.reset_index()
    df_eletricity = df_eletricity.rename(columns = {'index':'datetime'})
    df_eletricity['datetime'] = df_eletricity.datetime.values.astype('int64') // 10 ** 6  ## divide by 10^6 to convert from ns to ms    
   
    # Get temperature data
    timeseries_from = df_eletricity["datetime"].iloc[0]
    timeseries_to = df_eletricity["datetime"].iloc[-1]

    url = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/1/station/71420/period/latest-months/data.json"
    response = urlopen(url)

    # convert response to json, to dataframe
    data_json = json.loads(response.read())
    df_smhi_data = json_normalize(data_json['value']) 

    # extract only the temperature in the timestamp interval
    df_smhi_data = df_smhi_data.loc[(df_smhi_data['date'] >= timeseries_from) & (df_smhi_data['date'] <= timeseries_to)]
    df_smhi_data = df_smhi_data.reset_index().rename(columns = {'date':'datetime'})

    # combine & clean data
    df_feature_data = df_eletricity.merge(df_smhi_data, how='inner', on='datetime')

    col_list = ["Hydro Water Reservoir", "Nuclear", "Other", "Solar", "Wind Onshore"]
    df_feature_data['total_generation'] = df_feature_data[list(col_list)].sum(axis=1)

    df_feature_data.drop(col_list + ["Fossil Gas", "index", "quality"], axis=1, inplace=True)

    df_feature_data["value"] = df_feature_data["value"].astype(float)

    df_feature_data.rename(columns={"Actual Load": "total_load", "value": "temperature", "datetime":"datetime"}, inplace=True)

    # upload to hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    electricity_data_fg = fs.get_feature_group(name = 'new_electricity_data_fg', version = 2)
    electricity_data_fg.insert(df_feature_data, write_options={"wait_for_job" : False})

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image",
   "entsoe-py", "datetime"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("abyel-hopsworks-secret"))
   def modal_feature_daily():
       get_daily_features()


if __name__ == "__main__":
    if LOCAL == True :
        get_daily_features()
    else:
        stub.deploy("modal_feature_daily")
        # with stub.run():
        #     f()