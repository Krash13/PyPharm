import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class BaseCompartmentModel:

    def __init__(self, configuration_matrix, outputs, volumes=None):
        self.configuration_matrix = np.array(configuration_matrix)
        self.outputs = np.array(outputs)
        if not volumes:
            self.volumes = np.ones(self.outputs.size)
        else:
            self.volumes = np.array(volumes)
        self.last_result = None

    def _сompartment_model(self, t, c):
        dc_dt = (self.configuration_matrix.T @ (c * self.volumes) \
            - self.configuration_matrix.sum(axis=1) * (c * self.volumes)\
            - self.outputs * (c * self.volumes)) / self.volumes
        return dc_dt

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01):
        assert any([c0, d, compartment_number is not None]), "Need to set c0 or d and compartment_number"
        if not c0:
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
            max_step=max_step
        )
        self.last_result = res
        return res


model = BaseCompartmentModel([[0, 0.4586], [0.1919, 0]], [0.0309, 0], volumes=[228, 629])

res = model(90, d=5700, compartment_number=0)
print()
plt.plot(res.t, res.y[0, :], 'g')
teoretic_x = [1, 2.5, 4, 6, 10, 20, 40, 60, 90]
teoretic_y = [16.1, 10, 7.9, 7.1, 6.9, 5, 4.6, 4, 3.5]
plt.plot(teoretic_x, teoretic_y, '*r')
plt.show()
