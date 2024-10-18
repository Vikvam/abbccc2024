from typing import List
from abc import ABC, abstractmethod

class Block(ABC):
    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

class Electrolyser(Block):
    def __init__(self, h2_energy_ratio: float, switch_on_delay: int, switch_off_delay: int, max_power: float):
        """
        h2_power_ratio: how much kwh needed to produce one kg of h2
        switch_on_delay: how many samples needed for it to turn on
        switch_off_delay: how many samples needed for it to turn off
        max_power: how much kw it can handle
        """

        # Stale parameters
        self._h2_energy_ratio = h2_energy_ratio
        self._switch_on_delay = switch_on_delay
        self._switch_off_delay = switch_off_delay
        self._max_power = max_power

        self._cur_switch_on_delay = switch_on_delay
        self._cur_switch_off_delay = switch_off_delay
        self._on = True
        self._current_h2_output: float = 0

    # state update
    def compute(self, input_energy: float):
        if self._on:
            if self._cur_switch_on_delay >= self._switch_on_delay:
                self._current_h2_output = input_energy / self._h2_energy_ratio
            else:
                self.current_h2_output = 0
                self._cur_switch_on_delay += 1
        else:
            if self._cur_switch_off_delay >= self._switch_off_delay:
                self._current_h2_output = 0
            else:
                self.current_h2_output = input_energy / self._h2_energy_ratio
                self._cur_switch_off_delay += 1

    def set_parameters(self, parameters):
        on = parameters[0] 
        if on == True:
            self._on = True
            self._cur_switch_off_delay = 0
        else:
            self._on = False
            self._cur_switch_on_delay = 0

    def get_parameters(self):
        return [self._on]

    def output(self):
        return self._current_h2_output

class Buffer(Block):
    def __init__(self, max_energy_storage: float, max_input_energy: float, max_output_energy: float):
        self.output = False
        self.input = False

        self._max_energy_storage = max_energy_storage
        self._max_output_energy = max_output_energy
        self._max_input_energy = max_input_energy
        self._cur_energy = 0
        self._current_energy_output = 0

    # state update
    def compute(self, input_energy: float):
        if self.input:
            self._cur_energy += min(input_energy, self._max_input_energy)
            if self._cur_energy > self._max_energy_storage:
                self._cur_energy = self._max_energy_storage
    
    def output(self):
        if self.output:
            ret = self._current_energy_output
            self._cur_energy -= ret
            return ret
        else:
            return 0

    @property
    def cur_energy(self):
        return self._cur_energy

    def get_parameters(self):
        return [self._input, self._output]

    def set_parameters(self, parameters):
        inp = parameters[0]
        output = parameters[1]
        self._input = inp
        self._output = output

class ABBCCCModel():
    def __init__(self, electrolysers: List[Electrolyser], buffers: List[Buffer], sampling_period: int):
        """
        electrolysers: array of electrolysers
        buffers: array of buffers
        sampling_period: how many samples per hour
            e.g.: 20 means the sampling period of 3m
        """

        self._electrolysers: List[Electrolyser] = electrolysers
        self._buffers: List[Buffer] = buffers
        self._n_electrolysers = len(electrolysers)
        self._n_buffers = len(buffers) 
        self._sampling_period = sampling_period

        self._last_energy = 0

    def step(self, input_energy: float):
        # first, all outputs must be obtained
        h2_output = 0    # the output of h2 made
        for electrolyser in self._electrolysers:
            h2_output += electrolyser.output()

        buffer_energy = 0  # the output of buffers
        for buffer in self._buffers:
            buffer_energy += buffer.output()

        electrolyser_parameters = []
        buffer_parameters = []
        for electrolyser in self._electrolysers:
            electrolyser_parameters.append(electrolyser.get_parameters())
        for buffer in self._buffers:
            buffer_parameters.append(buffer.get_parameters())

        # update block parameters using a new strategy
        # TODO

        for i, electrolyser in enumerate(self._electrolysers):
            electrolyser.set_parameters(electrolyser_parameters[i])
        for i, buffer in enumerate(self._buffers):
            buffer.set_parameters(buffer_parameters[i]) 

        # now, update block states
        self._last_energy = input_energy + buffer_energy   # sumator
        for buffer in self._buffers:
            buffer.compute(self._last_energy)
        for electrolyser in self._electrolysers:
            electrolyser.compute(self._last_energy)


    def simulate(self, time_series_w):
        for n in time_series_w:
            self.step(n / self._sampling_period)

if __name__ == "__main__":

    # first: Quest One
    electrolysers = [Electrolyser(53, 20, 5, 150), Electrolyser(60, 5, 2, 50)]
    buffers = []

    model = ABBCCCModel(electrolysers, buffers, 12)

    # in kw
    time_series_w = [10000, 10000, 10000, 10000]

    model.simulate(time_series_w)
    