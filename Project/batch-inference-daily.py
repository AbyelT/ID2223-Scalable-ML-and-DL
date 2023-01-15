import modal

LOCAL = False

def get_batch_pred():
    import hopsworks
    import joblib
    import pandas as pd
    import numpy as np
    import json
    import dataframe_image as dfi
    from datetime import datetime, timedelta, date
    from entsoe import EntsoePandasClient
    from urllib.request import urlopen
    from pandas import json_normalize
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import os

    # Get feature group & model from hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    feature_group = fs.get_feature_group(name="new_electricity_data_fg", version=2)

    mr = project.get_model_registry()
    model = mr.get_model("SE3_elec_price_model", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/electricity_price.pkl")

    # Predict the price for current day

    last_48h = 48
    X_pred = feature_group.read().tail(last_48h)
    print("Latest 48 hour instances: \n{}".format(X_pred))

    X_pred.set_index('datetime',inplace=True)
    print(X_pred)

    ## pre-process before predict
    sc_x=StandardScaler()
    x_scaled=sc_x.fit_transform(X_pred)
    sc_y=StandardScaler()
    sc_y=sc_y.fit(X_pred[['day_ahead_price']])

    step_back=24
    no_records=len(x_scaled)
    no_cols=4
    X_train_shape_pred=[]
    for i in range(step_back,no_records):
        X_train_shape_pred.append(x_scaled[i-step_back:i])
    X_train_shape_pred=np.array(X_train_shape_pred)

    # predict prices for next 24 h
    pred_price = model.predict(X_train_shape_pred)
    final_pred=sc_y.inverse_transform(pred_price)

    # compare with predictions
    date_from = datetime.now()
    date_from = date_from.date().strftime('%Y%m%d')
    date_to = (datetime.strptime(date_from, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')

    client = EntsoePandasClient(api_key="cb3a29b2-3276-4a4c-aba3-6507120d99be")
    start = pd.Timestamp(date_from, tz='Europe/Stockholm')
    end = pd.Timestamp(date_to, tz='Europe/Stockholm')
    country_code = 'SE_3'  

    df_day_price = client.query_day_ahead_prices(country_code, start=start,end=end)
    # df_generation = client.query_generation_forecast(country_code, start=start,end=end)
    # df_load = client.query_load_forecast(country_code, start=start,end=end)
    
    inference_df = pd.DataFrame({'datetime': df_day_price.index, 'day_ahead_price': df_day_price.values})[:24]
    inference_df['prediction'] = final_pred
    inference_df['datetime'] = inference_df['datetime'].astype(str)

    monitor_fg = fs.get_or_create_feature_group(name="new_electricity_prediction_fg",
                                                    version=1,
                                                    primary_key=["datetime"],
                                                    description="SE3 electricity price Prediction/Forecasted price Monitoring")

    monitor_fg.insert(inference_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    
    history_df = pd.concat([history_df, inference_df], ignore_index=True)

    # MAE
    y_pred = history_df['prediction']
    y_test = history_df['day_ahead_price']
    mean_error = mean_absolute_error(y_test, y_pred)
    print("MAE: {}".format(mean_error))  # in MWh

    ## Generate figures for daily monitoring

    # generate 24h recent prediction
    dataset_api = project.get_dataset_api()
    dfi.export(history_df.tail(24), './df_se3_elec_price_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./df_se3_elec_price_recent.png", "Resources/images", overwrite=True)

    # generate comparision plot between Entsoe forecast and prediction
    aa=[x for x in range(len(history_df))]
    plt.figure(0, figsize=(14,4))
    plt.plot(aa, history_df['day_ahead_price'], marker='.', label="forecast (Entsoe)")
    plt.plot(aa, history_df['prediction'], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Electricity Day Ahead Price [EUR/MWh]', size=15)
    plt.xlabel('Time step [Hour]', size=30)
    plt.legend(fontsize=15)

    plt.savefig("./df_se3_elec_price_prediction.png")
    dataset_api.upload("./df_se3_elec_price_prediction.png", "Resources/images", overwrite=True)

    # Generate barplot of error metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = ['RMSE', 'MAE']
    values = [rmse, mae]
    colors = ['blue', 'red']
    plt.figure(1)
    plt.bar(metrics, values, color=colors)
    plt.title('Error metrics', fontsize=14)
    plt.xlabel('Metric type', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True)

    plt.savefig("./df_se3_error_metrics.png")
    dataset_api.upload("./df_se3_error_metrics.png", "Resources/images", overwrite=True)

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn==1.0.2", "entsoe-py",
        "dataframe-image", "matplotlib", "numpy", "pandas", "datetime", "tensorflow", "keras"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("abyel-hopsworks-secret"))
    def modal_batch_pred():
        get_batch_pred()

if __name__ == "__main__":
    if LOCAL:
        get_batch_pred()
    else:
        stub.deploy("modal_batch_pred")