import logging
from dataclasses import dataclass
from typing import Optional

import gurobipy as grb
import numpy as np


@dataclass
class Consumer:
    t_delta: int
    p_max: float
    load_min: float
    H2_per_W: float


PEM = Consumer(0, 200*10e3, .4, 1 / 4.7)
AWE = Consumer(3, 2.17*10e6, .2, 1 / 4.1)


@dataclass
class Result:
    obj: float
    H: np.array
    P: np.array
    load: np.array
    buffered: np.array
    buffer_capacity: np.array
    production_gap: np.array
    P_delta: float
    P_avg: float

    def __str__(self):
        return f"obj: {self.obj}, buffer_capacity: {self.buffer_capacity}, P_delta: {self.P_delta}"


class LP:
    def __init__(self, production: np.array, PEM_amount: int, AWE_amount: int, eps_production: int, b_max: float = np.inf):
        self.P = production
        self.T = len(production)
        self.N = PEM_amount + AWE_amount

        self.b_max = b_max

        self.p_max = [PEM.p_max] * PEM_amount + [AWE.p_max] * AWE_amount
        self.l_min = [PEM.load_min] * PEM_amount + [AWE.load_min] * AWE_amount
        self.t_delta = [PEM.t_delta] * PEM_amount + [AWE.t_delta] * AWE_amount
        self.H = [PEM.H2_per_W] * PEM_amount + [AWE.H2_per_W] * AWE_amount

        self.model = self._init_model()

    def _init_model(self) -> (grb.Model, dict[str, grb.Var]):
        model = grb.Model()

        # variables
        l = model.addVars(self.N, self.T, vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name="l")       # system output, current load
        L = model.addVars(self.N, self.T, vtype=grb.GRB.CONTINUOUS, lb=0, ub=1, name="L")       # system input, desired load
        H = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS, lb=0, name="H")                     # amount of H2
        b = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=self.b_max, name="b")                              # buffer capacity
        B = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS, name="B")                           # rate of charging
        # auxiliary variables
        on = model.addVars(self.N, self.T, vtype=grb.GRB.BINARY)                                # helper, to limit load
        P_out = model.addVars(self.T, lb=0, vtype=grb.GRB.CONTINUOUS)                           # helper, consumption
        P_delta_ = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS, name="P_delta")
        P_delta = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS)
        P_avg_ = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS)
        P_avg = model.addVars(self.T, vtype=grb.GRB.CONTINUOUS, name="P_avg")

        # load in <load_min, 1>
        for t in range(self.T):
            for n in range(self.N):
                model.addGenConstrIndicator(on[n, t], 1, L[n, t] >= self.l_min[n])
                model.addGenConstrIndicator(on[n,t], 0, L[n,t] == 0)
        # consumer system → l
        model.addConstrs((
            l[n,t] == 1 / self.t_delta[n] * grb.quicksum(L[n, i] for i in range(max(0, t - self.t_delta[n]), t))
            for t in range(self.T) for n in range(self.N)
            if self.t_delta[n] > 0
        ))
        # production of H
        model.addConstrs((
            H[t] == grb.quicksum(l[n,t] * self.p_max[n] * self.H[n] for n in range(self.N))
            for t in range(self.T)
        ))
        # buffer capacity
        model.addConstr(b <= self.b_max)
        model.addConstrs((B[t] <= b for t in range(self.T)))
        model.addConstrs((B[t] >= -b for t in range(self.T)))
        # power in /out
        model.addConstrs((
            P_out[t] == grb.quicksum(l[n,t] * self.p_max[n] for n in range(self.N))
            for t in range(self.T)
        ))
        # power difference
        model.addConstrs((
            P_delta_[t] == self.P[t] - B[t] - P_out[t]
            for t in range(self.T)
        ))
        model.addConstrs((
            P_delta[t] == grb.abs_(P_delta_[t])
            for t in range(self.T)
        ))
        # power average over T
        avg = np.mean(self.P[t])
        model.addConstrs((
            P_avg_[t] == avg - P_out[t]
            for t in range(self.T)
        ))
        model.addConstrs((
            P_avg[t] == grb.abs_(P_avg_[t])
            for t in range(self.T)
        ))

        # stability constraints
        # model.addConstr(grb.quicksum(P_delta) <= 10e5)
        # model.addConstr(grb.quicksum(P_avg) <= 10e5)

        model.setObjectiveN(
            - grb.quicksum(H) / (np.sum(self.p_max) + self.T) * 1
            + grb.quicksum(P_avg[t] for t in range(self.T)) / avg
        , index=0)
        model.setObjectiveN(b, index=1)
        model.ModelSense = grb.GRB.MINIMIZE

        model.Params.NumericFocus = 3
        model.Params.TimeLimit = 180

        model.update()
        return model

    def solve(self) -> grb.Model:
        self.model.optimize()
        return self.model

    def get_result(self) -> Result:
        feasible = self.model.status == grb.GRB.OPTIMAL or (self.model.status == grb.GRB.TIME_LIMIT and self.model.SolCount > 0)
        if feasible:
            load = [[self.model.getVarByName(f"l[{n},{t}]").x for t in range(self.T)] for n in range(self.N)]
            P = [[self.model.getVarByName(f"l[{n},{t}]").x * self.p_max[n] for t in range(self.T)] for n in range(self.N)]
            H = [self.model.getVarByName(f"H[{t}]").x for t in range(self.T)]
            B = [self.model.getVarByName(f"B[{t}]").x for t in range(self.T)]
            P_avg = [self.model.getVarByName(f"P_avg[{t}]").x for t in range(self.T)]
            result = Result(
                obj=-np.sum(self.model.obj),
                H=np.array(H),
                P=np.array(P),
                load=np.array(load),
                buffered=np.array(B),
                buffer_capacity=self.model.getVarByName(f"b").x,
                P_delta=None,
                P_avg=np.array(P_avg),
                production_gap=[]
            )
            self.validate_result(result)
            return result
        else:
            raise RuntimeError(f"Model infeasible or unbounded")

    def validate_result(self, result: Result) -> bool:
        produced = round(np.sum(self.P))
        consumed = round(np.sum(result.P, axis=(0,1)))
        if not np.isclose(produced, consumed):
            logging.warning(f"Produced {produced}. Consumed {consumed}. {produced-consumed}.")
            return False
        return True
