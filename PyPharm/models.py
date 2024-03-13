from multiprocessing import shared_memory

import numpy as np
from scipy.integrate import solve_ivp, RK45
from scipy.integrate._ivp.rk import rk_step, SAFETY, MAX_FACTOR, MIN_FACTOR
from scipy.optimize import minimize
from .country_optimization import CountriesAlgorithm
from .country_optimization_v2 import CountriesAlgorithm_v2
from numba import njit
import matplotlib.pyplot as plt


class BaseCompartmentModel:

    configuration_matrix_target = None
    outputs_target = None
    volumes_target = None
    _optim = False
    numba_option = False

    def __init__(self, configuration_matrix, outputs, volumes=None, numba_option=False, use_shared_memory=False):
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
        self.use_shared_memory = use_shared_memory
        if self.use_shared_memory:
            self.memory_size = 2 + self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count
            self.memory = shared_memory.ShareableList(self.memory_size * [None])
            self.memory_name = self.memory.shm.name

    def __del__(self):
        if getattr(self, 'memory', None):
            self.memory.shm.close()
            self.memory.shm.unlink()

    def _compartment_model(self, t, c):
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
    def _numba_compartment_model(t, c, configuration_matrix, outputs, volumes):
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
            fun=self._compartment_model if not self.numba_option else lambda t, c: self._numba_compartment_model(t, c, self.configuration_matrix.astype(np.float64), self.outputs.astype(np.float64), self.volumes.astype(np.float64)),
            t_span=ts,
            y0=c0,
            max_step=max_step,
            t_eval=t_eval
        )
        return self.last_result
    
    def load_data_from_list(self, x):
        if self.configuration_matrix_target:
            self.configuration_matrix[self.configuration_matrix_target] = x[:self.configuration_matrix_target_count]
        if self.outputs_target:
            self.outputs[self.outputs_target] = x[
                                                self.configuration_matrix_target_count:self.configuration_matrix_target_count + self.outputs_target_count]
        if self.volumes_target:
            self.volumes[self.volumes_target] = x[self.configuration_matrix_target_count + self.outputs_target_count:self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count]

    def _target_function(self, x, max_step=0.01, metric='R2'):
        """
        Функция расчета значения целевой функции

        Args:
            x: Значение искомых параметров модели
            max_step: Максимальный шаг при решении СДУ

        Returns:
            Значение целевой функции, характеризующее отклонение от эксперементальных данных
        """
        self.load_data_from_list(x)
        c0 = self.c0
        if c0 is None:
            c0 = np.zeros(self.outputs.size)
            c0[self.compartment_number] = self.d / self.volumes[self.compartment_number]
        self(
            t_max=np.max(self.teoretic_x),
            c0=c0,
            t_eval=self.teoretic_x,
            max_step=max_step
        )
        target_results = self.last_result.y[tuple(self.know_compartments), :]
        if metric == 'R2':
            return np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))
        elif metric == 'norm':
            return np.linalg.norm(target_results - self.teoretic_y)
        else:
            return np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))


    def load_optimization_data(self, teoretic_x, teoretic_y, know_compartments, w = None, c0=None, d=None, compartment_number=None):
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
            self.d = d
            self.compartment_number = compartment_number
            self.c0 = None
        else:
            self.c0 = np.array(c0)
        self.w = np.ones(self.teoretic_y.shape) if w is None else np.array(w)

    def optimize(self, method=None, max_step=0.01, metric='R2', **kwargs):
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
        f = lambda x: self._target_function(x, max_step=max_step, metric=metric)
        if method == 'country_optimization':
            CA = CountriesAlgorithm(
                f=f,
                memory_list=getattr(self, 'memory', None),
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

    def plot_model(self, compartment_numbers=None, compartment_names={}, left=None, right=None, y_lims={}, **kwargs):
        """
        Функция для построения графиков модели

        Args:
            compartment_numbers: Камеры, которые нужно отобразить (если не указать, отобразим все)
            compartment_names: Имена камер
        """
        if compartment_numbers:
            compartment_numbers = np.array(compartment_numbers)
        else:
            compartment_numbers = np.arange(self.outputs.size)
        self(**kwargs)
        for i in compartment_numbers:
            if hasattr(self, "teoretic_x") and hasattr(self, "teoretic_y") and i in self.know_compartments:
                plt.plot(self.teoretic_x, self.teoretic_y[self.know_compartments.index(i)], "*r")
            plt.plot(self.last_result.t, self.last_result.y[i])
            plt.title(compartment_names.get(i, i))
            plt.xlim(left=left, right=right)
            if y_lims.get(i):
                plt.ylim(y_lims.get(i))
            plt.grid()
            plt.show()


class MagicCompartmentModel(BaseCompartmentModel):

    need_magic_optimization = False

    def __init__(self, configuration_matrix, outputs, volumes=None, magic_coefficient=1, exclude_compartments=[], numba_option=False, use_shared_memory=False):
        super().__init__(configuration_matrix, outputs, volumes, numba_option, use_shared_memory)
        self.magic_coefficient = magic_coefficient
        self.exclude_compartments = np.array(exclude_compartments)
        self.need_magic_optimization = self.magic_coefficient is None
        if getattr(self, "memory", None):
            self.memory.shm.close()
            self.memory.shm.unlink()
            self.memory_size += int(self.need_magic_optimization)
            self.memory = shared_memory.ShareableList(
                sequence=self.memory_size * [None]
            )
            self.memory_name = self.memory.shm.name

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

    def load_data_from_list(self, x):
        super().load_data_from_list(x)
        if self.need_magic_optimization:
            self.magic_coefficient = x[-1]
        
    def optimize(self, method=None, max_step=0.01, **kwargs):
        x = super().optimize(method, max_step, **kwargs)
        if self.need_magic_optimization:
            self.magic_coefficient = x[-1]
            self.need_magic_optimization = False
        return x


class ReleaseCompartmentModel(BaseCompartmentModel):

    need_v_release_optimization = False

    class ReleaseRK45(RK45):

        def __init__(self, fun, t0, y0, t_bound, release_function, compartment_number, c0, max_step=np.inf,
                     rtol=1e-3, atol=1e-6, vectorized=False,
                     first_step=None, **extraneous):
            super().__init__(fun, t0, y0, t_bound, max_step=max_step,
                     rtol=rtol, atol=atol, vectorized=vectorized,
                     first_step=first_step, **extraneous)
            self.release_function = release_function
            self.compartment_number = compartment_number
            self.c0 = c0
            self.old_release_correction = 0

        def _step_impl(self):
            result = super()._step_impl()
            release_correction = self.release_function(self.t, self.c0)
            self.y[self.compartment_number] += release_correction - self.old_release_correction
            self.old_release_correction = release_correction
            return result

    def __init__(self, v_release, release_parameters, release_compartment, release_function=None, *args, **kwargs):
        """
        Камерная модель с высвобождением для описания фармакокинетики системы

        Неизвестные параметры при необходимости задаются как None
        например configuration_matrix = [[0, 1], [None, 0]]

        Args:
            configuration_matrix: Настроечная матрица модели, отображающая константы перехода между матрицами
            outputs: Вектор констант перехода во вне камер
            volumes: Объемы камер
            v_release: Объем гепотетической камеры из которой происходит высвобождение
            release_parameters: Параметры функции высвобождения
            release_compartment: Номер камеры в которую происходит высвобождение
            release_function: Функция высвобождения по умолчанию f(t,m,b,c) = c0 * c * t ** b / (t ** b + m)
        """
        super().__init__(*args, **kwargs)
        self.release_parameters = np.array(release_parameters)
        self.release_parameters_target_count = 0
        if np.any(self.release_parameters == None):
            self.release_parameters_target = np.where(self.release_parameters == None)
            self.release_parameters_target_count = np.sum(self.release_parameters == None)
        self.v_release = v_release
        if self.v_release is None:
            self.need_v_release_optimization = True
        self.release_compartment = release_compartment
        self.release_function = release_function

        if getattr(self, "memory", None):
            self.memory.shm.close()
            self.memory.shm.unlink()
            self.memory_size += self.release_parameters_target_count + int(self.need_v_release_optimization)
            self.memory = shared_memory.ShareableList(
                sequence=self.memory_size * [None]
            )
            self.memory_name = self.memory.shm.name

    def load_data_from_list(self, x):
        super().load_data_from_list(x)
        s = self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count
        if self.release_parameters_target:
            self.release_parameters[self.release_parameters_target] = x[s:s + self.release_parameters_target_count]
        if self.need_v_release_optimization:
            self.v_release = x[s + self.release_parameters_target_count]

    def _default_release_function(self, t, c0):
        """
        Функция для поправки на высвобождение
        """
        m, b, c = self.release_parameters
        return c0 * c * t ** b / (t ** b + m)

    def get_release_function(self):
        if self.release_function is not None:
            return lambda t, c0: self.release_function(t, c0, *self.release_parameters)
        else:
            return self._default_release_function

    def __call__(self, t_max, c0=None, d=None, max_step=0.01, t_eval=None, **kwargs):
        """
                Расчет кривых концентраций по фармакокинетической модели

                Args:
                    t_max: Предельное время расчета
                    c0: Начальная концентрация в камере из которой высвобождается вещество
                    d: Вводимая доза
                    max_step: Максимальный шаг при решении СДУ
                    t_eval: Временные точки, в которых необходимо молучить решение

                Returns:
                    Результат работы решателя scipy solve_ivp
                """
        if not self._optim:
            assert (not any([self.configuration_matrix_target, self.outputs_target, self.volumes_target])), \
                "It is impossible to make a calculation with unknown parameters"
        assert any([c0 is not None, d]), "Need to set c0 or d and compartment_number"
        if c0 is None:
            assert d, "Need to set d"
            c0 = d / self.v_release
        ts = [0, t_max]
        y0 = np.zeros(self.outputs.shape)
        self.last_result = solve_ivp(
            fun=self._сompartment_model if
            not self.numba_option
            else lambda t, c: self._numba_сompartment_model(t, c,
                                                             self.configuration_matrix.astype(
                                                                 np.float64),
                                                             self.outputs.astype(
                                                                 np.float64),
                                                             self.volumes.astype(
                                                     np.float64)),
            t_span=ts,
            y0=y0,
            max_step=max_step,
            t_eval=t_eval,
            method=self.ReleaseRK45,
            release_function=self.get_release_function(),
            compartment_number=self.release_compartment,
            c0=c0
        )
        return self.last_result

    def _target_function(self, x, max_step=0.01, metric='R2'):
        """
        Функция расчета значения целевой функции

        Args:
            x: Значение искомых параметров модели
            max_step: Максимальный шаг при решении СДУ

        Returns:
            Значение целевой функции, характеризующее отклонение от эксперементальных данных
        """
        self.load_data_from_list(x)
        c0 = self.c0
        if c0 is None:
            c0 = self.d / self.v_release
        self(
            t_max=np.max(self.teoretic_x),
            c0=c0,
            t_eval=self.teoretic_x,
            max_step=max_step
        )
        target_results = self.last_result.y[tuple(self.know_compartments), :]
        if metric == 'R2':
            return np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))
        elif metric == 'norm':
            return np.linalg.norm(target_results - self.teoretic_y)
        else:
            return np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))

    def load_optimization_data(self, teoretic_x, teoretic_y, know_compartments, w = None, c0=None, d=None, compartment_number=None):
        """
        Функция загрузки в модель эксперементальных данных

        Args:
            teoretic_x: Вектор временных точек теоретических значений
            teoretic_y: Матрица с теоретическими значениями
            know_compartments: Вектор с номерами камер, по которым есть данные
            c0: Начальная концентрация в камере из которой высвобождается вещество
            d: Вводимая доза

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
            self.d = d
            self.c0 = None
        else:
            self.c0 = np.array(c0)
        self.w = np.ones(self.teoretic_y.shape) if w is None else np.array(w)

    def optimize(self, method=None, max_step=0.01, **kwargs):
        x = super().optimize(method, max_step, **kwargs)
        s = self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count
        if self.release_parameters_target:
            self.release_parameters[self.release_parameters_target] = x[s:s + self.release_parameters_target_count]
        if self.need_v_release_optimization:
            self.v_release = x[s:s + self.release_parameters_target_count + 1]
            self.need_v_release_optimization = False
        return x
