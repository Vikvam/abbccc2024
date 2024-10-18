import io
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@dataclass
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

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as file:
            lines = file.readlines()

        latitude = float(lines[0].split(":")[1].strip())
        longitude = float(lines[1].split(":")[1].strip())
        elevation = float(lines[2].split(":")[1].strip())
        radiation = lines[3].split(":")[1].strip()
        slope = int(lines[6].split(":")[1].split()[0])
        azimuth = int(lines[7].split(":")[1].split()[0])
        nominal = float(lines[8].split(":")[1].strip())
        losses = float(lines[9].split(":")[1].strip())

        data = pd.read_csv(io.StringIO("".join(lines[10:-10])))
        data["time"] = pd.to_datetime(data["time"], format="%Y%m%d:%H%M")

        return cls(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            radiation=radiation,
            slope=slope,
            azimuth=azimuth,
            nominal_power=nominal,
            losses=losses,
            data=data
        )

    @property
    def data_power(self):
        return self.data["P"]

    @property
    def data_time(self):
        return self.data["time"]

    @property
    def data_irradiance(self):
        return self.data["G(i)"]

    @property
    def data_sun_height(self):
        return self.data["H_sun"]

    @property
    def data_air_temp(self):
        return self.data["T2m"]

    @property
    def data_wind_speed(self):
        return self.data["WS10m"]

def constants(power):
    data = pd.Series.to_numpy(power)
    data_length = len(data)
    time_array = np.arange(data_length)
    return data, data_length, time_array

def day_time_E_func(data, time_length):
    #time_length = 24  # 24 h
    day_data_len = len(data)//time_length
    day_time_array = np.arange(day_data_len)
    day_time_E = np.zeros(day_data_len)
    sum_day = 0
    hours = 0
    idx = 0
    for i in range(len(data)):
        sum_day += data[i]
        hours += 1
        if hours == time_length:
            #print("here")
            day_time_E[idx] = sum_day/1000000000  # GW
            sum_day = 0
            idx += 1
            hours = 0
    return day_time_array, day_time_E

def plot(x,y,year):
    start = 365 * (year - 1)
    end = 365 * year
    plt.figure(figsize=(15, 10))
    plt.plot(x[start:end],y[start:end])
    plt.show()

def plot_week(data, day):
    t = np.arange(len(data))
    hours = 24
    plt.figure(figsize=(15, 10))
    plt.plot(t[day*hours:(day+7)*hours], data[day*hours:(day+7)*hours])
    plt.show()

def mean_E_in_hour(data):
    return np.sum(data)/len(data)

def wanted_mean_func(mean_E, reduction_in_procents):
    reduction = reduction_in_procents / 100
    return mean_E * (1 - reduction)

def mean_E_by_days(data):
    day = 24
    time_array = np.zeros(day)
    for i in range(0, len(data), day):
        time_array += data[i:i+day]
        if i < 100:
            print(time_array)
    time_array /= len(data)/day
    return time_array


def main(power):
    data, data_length, time_array = constants(power)
    mean_E = mean_E_in_hour(data)
    print("Průměrná výkon elektrárny za hodinu: ", mean_E)
    reduction_in_procents = 0.1
    wanted_mean_E = wanted_mean_func(mean_E,reduction_in_procents)
    print("Požadovaný výkon na elektrolyzérech za hodinu: ", wanted_mean_E)
    time_length = 24  # in hours (day)
    day_time_array, day_time_E = day_time_E_func(data, time_length)
    year = 5
    day = 300
    mean_E_in_hours = mean_E_by_days(data)
    print("Průměrný výkon elektrárny během dne v hodinách: ", mean_E_in_hours)
    plot_week(data, day)
    #plot(day_time_array, day_time_E, year)

if __name__ == "__main__":
    dataset = Dataset.load("../data/Timeseries_33.153_-100.213_E5_200000kWp_crystSi_14_33deg_-3deg_2013_2023.csv")
    power = dataset.data_power
    #print(dataset.data_time)
    main(power)
