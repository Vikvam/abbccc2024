import openmeteo_requests

import requests_cache
import pandas as pd
from networkx.algorithms.efficiency_measures import efficiency
from retry_requests import retry
import numpy as np
from dataclasses import make_dataclass

"""
#####################################################################################
										WIND
#####################################################################################
"""

latitude = 51.52
longitude = 13.41


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": latitude,
	"longitude": longitude,
	"hourly": "wind_speed_180m",
	"past_days": 91
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_wind_speed_180m = hourly.Variables(0).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["wind_speed_180m"] = hourly_wind_speed_180m

hourly_dataframe_wind = pd.DataFrame(data = hourly_data)

hourly_dataframe_wind = pd.DataFrame(data = hourly_data)
hourly_dataframe_wind = hourly_dataframe_wind.wind_speed_180m
hourly_dataframe_wind = pd.Series.to_numpy(hourly_dataframe_wind)
#print(hourly_dataframe_wind)

"""
#####################################################################################
										SOLAR
#####################################################################################
"""

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": latitude,
	"longitude": longitude,
	"hourly": "direct_normal_irradiance",
	"past_days": 91
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_direct_normal_irradiance = hourly.Variables(0).ValuesAsNumpy()
#print(hourly_direct_normal_irradiance)

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance

hourly_dataframe_solar = pd.DataFrame(data = hourly_data)
time = hourly_dataframe_solar.date
hourly_dataframe_solar = hourly_dataframe_solar.direct_normal_irradiance
hourly_dataframe_solar = pd.Series.to_numpy(hourly_dataframe_solar)

# WIND
air_density = 1.225
radius_of_turbines = 40
area_of_turbines = np.pi * (radius_of_turbines**2)
efficiency_of_turbines = 0.4  # 40 %
number_of_turbines = 10
wind_params = [air_density, area_of_turbines, efficiency_of_turbines, number_of_turbines]

# SOLAR
area = 100 * 100  # 10000 m2
panel_efficiency = 0.22  # 22 %
performance_ratio = 0.85  # 85 %
solar_params = [area, panel_efficiency, performance_ratio]

def optimal_wind_power(data, air_density, area_of_turbine, efficiency, number_of_turbines):
	return 0.5 * (data**3) * number_of_turbines * air_density * area_of_turbine * efficiency

def optimal_solar_power(data, area, panel_efficiency, performance_ratio):
	return data * area * panel_efficiency * performance_ratio


def main(hourly_dataframe_wind, hourly_dataframe_solar, wind_params, solar_params, latitude, longitude, time):
	wind_power = optimal_wind_power(hourly_dataframe_wind, wind_params[0], wind_params[1],
									wind_params[2], wind_params[3])
	solar_power = optimal_solar_power(hourly_dataframe_solar, solar_params[0],
									   solar_params[1], solar_params[2])
	"""for i in range(len(hourly_dataframe_wind)):
		print("wind power: ", wind_power[i], "  solar power", solar_power[i])
		print("wind data: ", hourly_dataframe_wind[i])"""
	#print(wind_power,"here")
	return wind_power, solar_power, latitude, longitude, time

class Dataset:
	latitude: float
	longitude: float
	elevation: float
	radiation: str
	slope: int
	azimuth: int
	nominal_power: float
	losses: float
	data: pd.DataFrame

	def load(self):
		global longitude, latitude, time
		#print(longitude, latitude, time)
		data1, data2, latitude, longitude, time = main(hourly_dataframe_wind, hourly_dataframe_solar,
													   wind_params, solar_params, latitude, longitude, time)
		elevation = 530.0
		radiation = 'PVGIS-ERA5'
		slope = 33
		azimuth = -3
		nominal_power = 200000.0
		losses = 14.0
		data = data1 + data2
		#power = make_dataclass("P",[("P", float)])
		d = {'P': data}
		df = pd.DataFrame(data=d)
		print(df)



if __name__ == "__main__":
	#data1, data2, latitude, longitude, time = main(hourly_dataframe_wind, hourly_dataframe_solar, wind_params, solar_params, latitude, longitude, time)
	dataset = Dataset().load()