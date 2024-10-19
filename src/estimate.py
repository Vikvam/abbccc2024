from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False

def find_threshold(power: np.array, threshold):
    for i, p in enumerate(power):
        if p > threshold:
            return i
    return None

def find_min(dataset: Dataset):
    return np.min(dataset.data_power)

def find_max(dataset: Dataset):
    return np.max(dataset.data_power)

def check_energy(dataset: Dataset, pavg: float):
    time = np.arange(len(dataset.data_power))
    power = np.array(dataset.data_power)

    if DEBUG:
        plt.plot(time, power)
        plt.show()

    max_energy = np.max(power)
    min_energy = np.min(power)

    if pavg > max_energy:
        print("Too much energy")
        return None

    start_index = find_threshold(power, pavg)
    if start_index == None:
        return None

    accumulator = 0
    up = True
    below_threshold = False

    for i in range(start_index, len(dataset.data_power)):
        cur_work = (power[i] - pavg) # nasobeni 1 - jakoby watt hodina
        if up:
            if cur_work < 0:
                # switch state
                up = False
            accumulator += cur_work
        else:
            if cur_work > 0:
                # switch state
                up = True
            accumulator += cur_work # kladne -> ok

            if (accumulator < 0):
                below_threshold = True
                break
        
    if DEBUG:
        pmax_arr = np.full(len(dataset.data_power), pavg)
        plt.plot(time, power, pmax_arr)
        plt.show()

    return below_threshold


def estimate_pavg(dataset):
    """
    """
  
    max_power = find_max(dataset)
    up_power = max_power
    down_power = 0
    max_iter = 20
    middle = 0

    # bisection
    for i in range(max_iter):
        middle = (up_power + down_power)/2
        below_threshold = check_energy(dataset, middle)
        if below_threshold == None:
            return None # chyba!
        
        if below_threshold:
            # too much wanted, up_power must be lowered
            up_power = middle
        else:
            # we can increase the power
            down_power = middle

    return middle
    
def calculate_ideal_parameters_only_awe(dataset: Dataset, awe_wh_h2: float, awe_power: float, awe_efficiency: float):
    """
    awe_wh_h2: how much wh needed to make 1kg of h2
    awe_power: how many Watts needed for the electrolyser to run
    awe_efficiency: efficiency of the awe
    """

    if (awe_efficiency < 0 or awe_efficiency > 1):
        return None

    pavg = estimate_pavg(dataset)
    max_power = find_max(dataset)
    min_power = find_min(dataset)

    awe_p_effective = awe_power * awe_efficiency

    min_n_of_awe = min_power // awe_power
    extra_n_of_awe = (pavg // awe_p_effective) - min_n_of_awe

    h2_per_hour = (extra_n_of_awe + min_n_of_awe)*awe_p_effective/awe_wh_h2

    return [min_n_of_awe, extra_n_of_awe, h2_per_hour]

if __name__ == "__main__":
    dataset = Dataset.load("../data/Timeseries_33.153_-100.213_E5_200000kWp_crystSi_14_33deg_-3deg_2013_2023.csv")
    data = calculate_ideal_parameters_only_awe(dataset, 53000, 1e6, 0.75)
    print(data)



