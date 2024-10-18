import io
from dataclasses import dataclass
import pandas as pd


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


if __name__ == "__main__":
    dataset = Dataset.load("../data/Timeseries_33.153_-100.213_E5_200000kWp_crystSi_14_33deg_-3deg_2013_2023.csv")
    print(dataset.data_power)
