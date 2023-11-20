import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from .country_optimization import CountriesAlgorithm
from .country_optimization_v2 import CountriesAlgorithm_v2
from numba import njit


class BaseCompartmentModel:

    configuration_matrix_target = None
    outputs_target = None
    volumes_target = None
    _optim = False
    numba_option = False

    def __init__(self, configuration_matrix, outputs, volumes=None, numba_option=False):
        """
        Базовая камерная модель для описания фармакокинетики системы

        Неизвестные параметры при необходимости задаются как None
        например configuration_matrix = [[0, 1], [None, 0]]

        Args:
            configuration_matrix: Настроечная матрица модели, отображающая константы перехода между матрицами
            outputs: Вектор констант перехода во вне камер
            volumes: Объемы камер
        """
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
        self.numba_option = numba_option

    def _сompartment_model(self, t, c):
        """
        Функция для расчета камерной модели

        Args:
            t: Текущее время
            c: Вектор концентраций

        Returns:
            Вектор изменений концентраций (c) в момент времени (t)
        """
        dc_dt = (self.configuration_matrix.T @ (c * self.volumes) \
            - self.configuration_matrix.sum(axis=1) * (c * self.volumes)\
            - self.outputs * (c * self.volumes)) / self.volumes
        return dc_dt

    @staticmethod
    @njit
    def _numba_сompartment_model(t, c, configuration_matrix, outputs, volumes):
        """
        Функция для расчета камерной модели

        Args:
            t: Текущее время
            c: Вектор концентраций

        Returns:
            Вектор изменений концентраций (c) в момент времени (t)
        """
        dc_dt = (configuration_matrix.T @ (c * volumes) \
            - configuration_matrix.sum(axis=1) * (c * volumes)\
            - outputs * (c * volumes)) / volumes
        return dc_dt

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01, t_eval=None):
        """
        Расчет кривых концентраций по фармакокинетической модели

        Args:
            t_max: Предельное время расчета
            c0: Вектор нулевых концентраций
            d: Вводимая доза
            compartment_number: Номер камеры в которую вводится доза
            max_step: Максимальный шаг при решении СДУ
            t_eval: Временные точки, в которых необходимо молучить решение

        Returns:
            Результат работы решателя scipy solve_ivp
        """
        if not self._optim:
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
        self.last_result = solve_ivp(
            fun=self._сompartment_model if not self.numba_option else lambda t, c: self._numba_сompartment_model(t, c, self.configuration_matrix.astype(np.float64), self.outputs.astype(np.float64), self.volumes.astype(np.float64)),
            t_span=ts,
            y0=c0,
            max_step=max_step,
            t_eval=t_eval
        )
        return self.last_result

    def _target_function(self, x, max_step=0.01):
        """
        Функция расчета значения целевой функции

        Args:
            x: Значение искомых параметров модели
            max_step: Максимальный шаг при решении СДУ

        Returns:
            Значение целевой функции, характеризующее отклонение от эксперементальных данных
        """
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

    def load_optimization_data(self, teoretic_x, teoretic_y, know_compartments, c0=None, d=None, compartment_number=None):
        """
        Функция загрузки в модель эксперементальных данных

        Args:
            teoretic_x: Вектор временных точек теоретических значений
            teoretic_y: Матрица с теоретическими значениями
            know_compartments: Вектор с номерами камер, по которым есть данные
            c0: Вектор нулевых концентраций
            d: Вводимая доза
            compartment_number: Номер камеры в которую вводится доза

        Returns:
            None
        """
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
        """
        Функция оптимизации модели

        Args:
            method: Метод оптимизации, любой доступный minimize + 'country_optimization' и 'country_optimization_v2'
            max_step: Максимальный шаг при решении СДУ
            **kwargs: Дополнительные именованные аргументы

        Returns:
            None
        """
        self._optim = True
        f = lambda x: self._target_function(x, max_step=max_step)
        if method == 'country_optimization':
            CA = CountriesAlgorithm(
                f=f,
                **kwargs
            )
            CA.start()
            x = CA.countries[0].population[0].x
        elif method == 'country_optimization_v2':
            CA = CountriesAlgorithm_v2(
                f=f,
                **kwargs
            )
            CA.start()
            x = CA.countries[0].population[0].x
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
            self.volumes[self.volumes_target] = x[self.configuration_matrix_target_count + self.outputs_target_count:self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count]
        self.configuration_matrix_target = None
        self.outputs_target = None
        self.volumes_target = None
        self._optim = False
        return x


class MagicCompartmentModelWith(BaseCompartmentModel):

    need_magic_optimization = False

    def __init__(self, configuration_matrix, outputs, volumes=None, magic_coefficient=1, exclude_compartments=[], numba_option=False):
        super().__init__(configuration_matrix, outputs, volumes, numba_option)
        self.magic_coefficient = magic_coefficient
        self.exclude_compartments = np.array(exclude_compartments)
        self.need_magic_optimization = self.magic_coefficient is None

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01, t_eval=None):
        if not self._optim and not self.magic_coefficient:
            raise Exception("Magic_coefficient parameter not specified")
        res = super().__call__(t_max, c0, d, compartment_number, max_step, t_eval)
        magic_arr = np.ones(self.configuration_matrix.shape[0]) * self.magic_coefficient
        if self.exclude_compartments:
            magic_arr[self.exclude_compartments] = 1
        magic_arr = np.repeat(magic_arr, res.y.shape[1])
        magic_arr = np.reshape(magic_arr, res.y.shape)
        res.y = magic_arr * res.y
        self.last_result = res
        return res

    def _target_function(self, x, max_step=0.01):
        if self.configuration_matrix_target:
            self.configuration_matrix[self.configuration_matrix_target] = x[:self.configuration_matrix_target_count]
        if self.outputs_target:
            self.outputs[self.outputs_target] = x[self.configuration_matrix_target_count:self.configuration_matrix_target_count + self.outputs_target_count]
        if self.volumes_target:
            self.volumes[self.volumes_target] = x[self.configuration_matrix_target_count + self.outputs_target_count:self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count]
        if self.need_magic_optimization:
            self.magic_coefficient = x[-1]
        self(
            t_max=np.max(self.teoretic_x),
            c0=self.c0,
            t_eval=self.teoretic_x,
            max_step=max_step
        )
        target_results = self.last_result.y[tuple(self.know_compartments), :]
        return np.sum(np.sum((target_results - self.teoretic_y) ** 2, axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))

    def optimize(self, method=None, max_step=0.01, **kwargs):
        x = super().optimize(method, max_step, **kwargs)
        if self.need_magic_optimization:
            self.magic_coefficient = x[-1]
            self.need_magic_optimization = False
        return x


class ReleaseCompartmentModel(BaseCompartmentModel):

    def __init__(self, release_parameters, v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_parameters = np.array(release_parameters)
        self.release_parameters_target_count = 0
        if np.any(self.release_parameters == None):
            self.release_parameters_target = np.where(self.release_parameters == None)
            self.release_parameters_target_count = np.sum(self.release_parameters == None)
        self.v = v

    def _runge_kutta(self, y, x, dx):
        k1 = dx * self._сompartment_model(x, y)
        k2 = dx * self._сompartment_model(x + 0.5 * dx, y + 0.5 * k1)
        k3 = dx * self._сompartment_model(x + 0.5 * dx, y + 0.5 * k2)
        k4 = dx * self._сompartment_model(x + dx, y + k3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def release_function(self, t, c0):
        """
        Функция для поправки на высвобождение
        """
        m, b, c = self.release_parametrs
        return c0 * c * t ** b / (t ** b + m)

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01, t_eval=None):
        if not self._optim:
            assert (not any([self.configuration_matrix_target, self.outputs_target, self.volumes_target])), \
                "It is impossible to make a calculation with unknown parameters"
        assert any([c0 is not None, d, compartment_number is not None]), "Need to set c0 or d and compartment_number"
        if c0 is None:
            assert all([d, compartment_number is not None]), "Need to set d and compartment_number"
            c0 = d / self.v
        t = 0
        old_release_correction = 0
        result_x = np.array([])
        result_y = np.array([])
        y = np.zeros(self.outputs.size)
        while t <= t_max:
            np.append(result_x, t)
            np.append(result_y, y)
            release_correction = self.release_function(t + max_step, c0)
            y = self._runge_kutta(y, t, max_step)
            y[compartment_number] += release_correction - old_release_correction
            #y[0] = breeding_function(dx, t12, y[0])  # делаем поправочку на распад
            old_release_correction = release_correction
            t += max_step
        return result_x, result_y
