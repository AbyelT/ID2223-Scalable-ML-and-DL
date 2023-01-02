from entsoe import EntsoePandasClient
import pandas as pd

# get client
client = EntsoePandasClient(api_key="<YOUR API KEY>")

# date and country
start = pd.Timestamp('20220101', tz='Europe/Brussels')
end = pd.Timestamp('20221231', tz='Europe/Brussels')
country_code = 'SE'  

# Generation per type
client.query_generation(country_code, start=start,end=end, psr_type=None)

# Day-ahead price
client.query_day_ahead_prices(country_code, start=start,end=end)

# Actual load (consumption)
client.query_load(country_code, start=start,end=end)