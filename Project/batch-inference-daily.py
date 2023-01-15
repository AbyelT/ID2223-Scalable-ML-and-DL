import modal

LOCAL = True

def get_batch_pred():
    import hopsworks
    import joblib
    import pandas as pd
    import numpy as np
    import json
    import dataframe_image as dfi
    from datetime import datetime, timedelta, date
    from urllib.request import urlopen
    from pandas import json_normalize
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error
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

    # Get 48h data
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
    print(X_train_shape_pred.shape)

    # predict prices for next 24 h
    pred_price = model.predict(X_train_shape_pred)
    final_pred=sc_y.inverse_transform(pred_price)
    print(final_pred.shape)

    # compare with predictions
    
    #transform

    #predict

    #get final pred


    # # predict and get latest (daily) feature
    # y_pred = model.predict(X_pred.drop(columns=['demand', 'date']))
    # print("Prediction: {}".format(y_pred[0]))

    # prediction_date = X_pred.iloc[0]['date']
    # prediction_date = prediction_date.date()
    # print("Prediction date: {}".format(prediction_date))




    # # get demand (forecast) from EIA (for comparison)
    # url = ('https://api.eia.gov/v2/electricity/rto/daily-region-data/data/'
    #        '?frequency=daily'
    #        '&data[0]=value'
    #        '&facets[respondent][]=NY'
    #        '&facets[timezone][]=Eastern'
    #        '&facets[type][]=DF'
    #        '&sort[0][column]=period'
    #        '&sort[0][direction]=desc'
    #        '&offset=0'
    #        '&length=5000')

    # url = url + '&start={}&end={}&api_key={}'.format(prediction_date, prediction_date, os.environ.get('EIA_API_KEY'))
    # data = requests.get(url).json()['response']['data']
    # forecast = data[0]['value']
    # print("EIA forecast: {}".format(forecast))

    # # create DF for monitoring data
    # now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    # data = {
    #     'prediction': y_pred,
    #     'actual': [X_pred.iloc[0]['demand']],
    #     'forecast_eia': [forecast],
    #     'prediction_date': [prediction_date],
    #     'datetime': [now],
    # }
    # monitor_df = pd.DataFrame(data)

    # # create monitoring FG
    # monitor_fg = fs.get_or_create_feature_group(name="ny_elec_predictions",
    #                                             version=1,
    #                                             primary_key=["datetime"],
    #                                             description="NY Electricity Prediction/Outcome Monitoring")

    # monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    # history_df = monitor_fg.read()
    # # Add our prediction to the history, as the history_df won't have it -
    # # the insertion was done asynchronously, so it will take ~1 min to land on App
    # # TODO: commented for now since we can wait in a notebook, remember to uncomment
    # #  if running e.g. in a modal job!
    # history_df = pd.concat([history_df, monitor_df], ignore_index=True)

    # # MAE
    # y_pred = history_df['prediction']
    # y_test = history_df['actual']
    # mean_error = mean_absolute_error(y_test, y_pred)
    # print("MAE: {}".format(mean_error))  # in MWh

    # # create "recents" table for UI and upload
    # dataset_api = project.get_dataset_api()
    # dfi.export(history_df.tail(5), './df_ny_elec_recent.png', table_conversion='matplotlib')
    # dataset_api.upload("./df_ny_elec_recent.png", "Resources/images", overwrite=True)

    # # create "prediction" chart for UI and upload
    # data = {'label': ['Predicted demand', 'Actual demand', 'EIA forecast'],
    #         'value': [monitor_df[l][0] for l in ['prediction', 'actual', 'forecast_eia']]}
    # pred_df = pd.DataFrame(data)
    # pred_plot = sns.barplot(data=pred_df, y='value', x='label')
    # plt.ylabel('Demand [MWh]')
    # plt.xlabel('')
    # plt.ylim(pred_df['value'].min() - 10000, pred_df['value'].max() + 5000)
    # plt.title('Predicted and actual demands for {}'.format(monitor_df['prediction_date'][0]))
    # fig = pred_plot.get_figure()
    # fig.savefig("./df_ny_elec_prediction.png")
    # dataset_api.upload("./df_ny_elec_prediction.png", "Resources/images", overwrite=True)

    # # create MAE trend graph for UI and upload
    # latest_history_df = history_df.loc[-5:]  # TODO: might want/need to change this somewhen
    # no_entries = len(latest_history_df)
    # mae = []
    # for i in range(no_entries):
    #     df = latest_history_df.loc[:i]
    #     mae.append([mean_absolute_error(df['actual'], df['prediction']),
    #                 mean_absolute_error(df['actual'], df['forecast_eia']),
    #                 pd.to_datetime(df['datetime'][i]).date()])
    # mae_df = pd.DataFrame(mae, columns=['Prediction', 'EIA forecast', 'Date'])
    # mae_plot = sns.lineplot(data=mae_df.melt(id_vars=['Date'],
    #                                          value_vars=['Prediction', 'EIA forecast']),
    #                         x='Date', y='value', hue='variable')
    # plt.ylabel('Demand [MWh]')
    # plt.title('Mean absolute error (MAE) for last {} predictions'.format(no_entries))
    # mae_plot.legend().set_title('MAE')
    # fig = mae_plot.get_figure()
    # fig.savefig("./df_ny_elec_mae.png")
    # dataset_api.upload("./df_ny_elec_mae.png", "Resources/images", overwrite=True)


if LOCAL == False:
    stub = modal.Stub() #apt_install(["libgomp1"])
    image = modal.Image.debian_slim().pip_install([
        "hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn==1.0.2", "entsoe-py",
        "dataframe-image", "matplotlib", "numpy", "pandas", "datetime", "tensorflow", "keras"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    # @stub.function(image=image, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def modal_batch_pred():
        get_batch_pred()

if __name__ == "__main__":
    if LOCAL:
        get_batch_pred()
    else:
        stub.deploy("modal_batch_pred")
        # with stub.run():
        #    modal_batch_elec()