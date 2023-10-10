import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from country_optimization import CountriesAlgorithm


class BaseCompartmentModel:

    configuration_matrix_target = None
    outputs_target = None
    volumes_target = None
    __optim = False

    def __init__(self, configuration_matrix, outputs, volumes=None):
        self.configuration_matrix = np.array(configuration_matrix)
        self.configuration_matrix_target_count = 0
        if np.any(self.configuration_matrix == None):
            self.configuration_matrix_target = np.where(self.configuration_matrix == None)
            self.configuration_matrix_target_count = np.sum(self.configuration_matrix == None)
        self.outputs = np.array(outputs)
        self.outputs_target_count = 0
        if np.any(self.outputs == None):
            self.outputs_target = np.where(self.outputs == None)
            self.outputs_target_count = np.sum(self.outputs == None)
        if not volumes:
            self.volumes = np.ones(self.outputs.size)
        else:
            self.volumes = np.array(volumes)
        self.volumes_target_count = 0
        if np.any(self.volumes == None):
            self.volumes_target = np.where(self.volumes == None)
            self.volumes_target_count = np.sum(self.volumes == None)
        self.last_result = None

    def _сompartment_model(self, t, c):
        dc_dt = (self.configuration_matrix.T @ (c * self.volumes) \
            - self.configuration_matrix.sum(axis=1) * (c * self.volumes)\
            - self.outputs * (c * self.volumes)) / self.volumes
        return dc_dt

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01, t_eval=None):
        if not self.__optim:
            assert (not any([self.configuration_matrix_target, self.outputs_target, self.volumes_target])), \
                "It is impossible to make a calculation with unknown parameters"
        assert any([c0 is not None, d, compartment_number is not None]), "Need to set c0 or d and compartment_number"
        if c0 is None:
            assert all([d, compartment_number is not None]), "Need to set d and compartment_number"
            c0 = np.zeros(self.outputs.size)
            c0[compartment_number] = d / self.volumes[compartment_number]
        else:
            c0 = np.array(c0)
        ts = [0, t_max]
        res = solve_ivp(
            fun=self._сompartment_model,
            t_span=ts,
            y0=c0,
            max_step=max_step,
            t_eval=t_eval
        )
        self.last_result = res
        return res

    def _target_function(self, x, max_step=0.01):
        if self.configuration_matrix_target:
            self.configuration_matrix[self.configuration_matrix_target] = x[:self.configuration_matrix_target_count]
        if self.outputs_target:
            self.outputs[self.outputs_target] = x[self.configuration_matrix_target_count:self.configuration_matrix_target_count + self.outputs_target_count]
        if self.volumes_target:
            self.volumes[self.volumes_target] = x[self.configuration_matrix_target_count + self.outputs_target_count:]
        self(
            t_max=np.max(self.teoretic_x),
            c0=self.c0,
            t_eval=self.teoretic_x,
            max_step=max_step
        )
        target_results = self.last_result.y[tuple(self.know_compartments), :]
        return np.sum(np.sum((target_results - self.teoretic_y) ** 2, axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))

    def load_optimization_data(self, teoretic_x, teoretic_y, know_compartments, w=None, c0=None, d=None, compartment_number=None):
        self.teoretic_x = np.array(teoretic_x)
        self.teoretic_y = np.array(teoretic_y)
        self.know_compartments = know_compartments
        self.teoretic_avg = np.average(self.teoretic_y, axis=1)
        self.teoretic_avg = np.repeat(self.teoretic_avg, self.teoretic_x.size)
        self.teoretic_avg = np.reshape(self.teoretic_avg, self.teoretic_y.shape)
        assert any([c0, d, compartment_number is not None]), "Need to set c0 or d and compartment_number"
        if not c0:
            assert all([d, compartment_number is not None]), "Need to set d and compartment_number"
            self.c0 = np.zeros(self.outputs.size)
            self.c0[compartment_number] = d / self.volumes[compartment_number]
        else:
            self.c0 = np.array(c0)

    def optimize(self, method=None, max_step=0.01, **kwargs):
        self.__optim = True
        f = lambda x: self._target_function(x, max_step=max_step)
        if method == 'country_optimization':
            CA = CountriesAlgorithm(
                f=f,
                **kwargs
            )
            CA.start()
        else:
            res = minimize(
                fun=f,
                method=method,
                **kwargs
            )
            x = res.x
        if self.configuration_matrix_target:
            self.configuration_matrix[self.configuration_matrix_target] = x[:self.configuration_matrix_target_count]
        if self.outputs_target:
            self.outputs[self.outputs_target] = x[
                                                self.configuration_matrix_target_count:self.configuration_matrix_target_count + self.outputs_target_count]
        if self.volumes_target:
            self.volumes[self.volumes_target] = x[
                                                self.configuration_matrix_target_count + self.outputs_target_count:]
        self.configuration_matrix_target = None
        self.outputs_target = None
        self.volumes_target = None
        self.__optim = False