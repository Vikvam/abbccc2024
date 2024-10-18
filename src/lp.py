from dataclasses import dataclass
from typing import Optional

import gurobipy as grb
import numpy as np


@dataclass
class Consumer:
    t_delta: int
    p_max: float
    min_load: float
    H2_per_kW: float


PEM = Consumer(0, 200*10e3, .5, np.nan)
AWE = Consumer(2, 2.17*10e6, .3, np.nan)


@dataclass
class Result:
    consumption: np.array
    load: np.array


class LP:
    def __init__(self, production: np.array, PEM_amount: int, AWE_amount: int, buffer_capacity: int):
        self.P = production
        self.T = len(production)
        self.N = PEM_amount + AWE_amount
        self.p_max = [PEM.p_max] * PEM_amount + [AWE.p_max] * AWE_amount
        self.l_min = [PEM.min_load] * PEM_amount + [AWE.min_load] * AWE_amount
        self.t_delta = [PEM.t_delta] * PEM_amount + [AWE.t_delta] * AWE_amount
        self.b = buffer_capacity

        self.model = self._init_model()

    def _init_model(self) -> (grb.Model, dict[str, grb.Var]):
        model = grb.Model()

        L = model.addVars(self.N, self.T, vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name="L")  # input, desired load
        l = model.addVars(self.N, self.T, vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name="l")   # output, current load
        diff_abs = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS, lb=0)                    # | production - demand |
        diff = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS)                              # helper, production - demand
        on = model.addVars(self.N, self.T, vtype=grb.GRB.BINARY, name="on")                 # input, on / off

        for t in range(self.T):
            for n in range(self.N):
                if self.t_delta[n] > 0:
                    model.addConstr(l[n,t] == 1/self.t_delta[n] * grb.quicksum(L[n,i] for i in range(max(0, t-self.t_delta[n]), t)))
                model.addGenConstrIndicator(on[n,t], 1, L[n,t] >= self.l_min[n])
                model.addGenConstrIndicator(on[n,t], 0, L[n,t] == 0)
            model.addConstr(diff[t] == self.P[t] - grb.quicksum(l[n,t] * self.p_max[n] for n in range(self.N)))
            model.addConstr(diff_abs[t] == grb.abs_(diff[t]))

        model.setObjective(grb.quicksum(diff_abs), sense=grb.GRB.MINIMIZE)
        model.update()

        return model

    def solve(self) -> grb.Model:
        self.model.optimize()
        return self.model

    def get_result(self) -> Optional[Result]:
        if self.model.status == grb.GRB.OPTIMAL:
            load = [[self.model.getVarByName(f"l[{n},{t}]").x for t in range(self.T)] for n in range(self.N)]
            p = [[self.model.getVarByName(f"l[{n},{t}]").x * self.p_max[n] for t in range(self.T)] for n in range(self.N)]
            return Result(consumption=np.array(p), load=np.array(load))
        else:
            return None
