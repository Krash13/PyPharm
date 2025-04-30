from multiprocessing import shared_memory

import numpy as np
from scipy.integrate import solve_ivp, RK45
from scipy.integrate import simps
from scipy.optimize import minimize
from PyPharm.algorithms.country_optimization import CountriesAlgorithm
from PyPharm.algorithms.country_optimization_v2 import CountriesAlgorithm_v2
from PyPharm.algorithms.genetic_optimization import GeneticAlgorithm
from numba import njit
import matplotlib.pyplot as plt

BIG_VALUE = 2 ** 24


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
            t_eval=t_eval,
            method='LSODA'
        )
        return self.last_result

    def get_kinetic_params(self, t_max, d, compartment_number, max_step=0.01):
        one_hour_result = self(t_max=1, d=d, compartment_number=compartment_number, max_step=max_step)
        auc_1h = simps(one_hour_result.y[compartment_number], one_hour_result.t)
        self(t_max=t_max, d=d, compartment_number=compartment_number, max_step=max_step)
        auc = simps(self.last_result.y[compartment_number], self.last_result.t)
        result_dict = {
            'c_max': self.last_result.y[compartment_number].max(),
            'V': self.volumes[compartment_number] if self.volumes is not None else None,
            'AUC': auc,
            'AUC_1h': auc_1h,
            'Cl': d / auc
        }
        return result_dict


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

    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', max_step=0.01, metric='R2', **kwargs):
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
        if user_method is not None:
            if method_is_func:
                x = user_method(f, **kwargs)
            else:
                optimization_obj = user_method(f, **kwargs)
                x = getattr(optimization_obj, optimization_func_name)()
        else:
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
            elif method == 'GA':
                CA = GeneticAlgorithm(
                    f=f,
                    **kwargs
                )
                x = CA.start()
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

    def plot_model(self, compartment_numbers=None, compartment_names=None, left=None, right=None, y_lims={}, **kwargs):
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
        if not compartment_names:
            compartment_names = {}
        if not y_lims:
            y_lims = {}
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
            try:
                plt.show()
            except AttributeError:
                plt.savefig(f'{compartment_names.get(i, str(i))}.png')
                plt.cla()


class MagicCompartmentModel(BaseCompartmentModel):

    need_magic_optimization = False

    def __init__(self, configuration_matrix, outputs, volumes=None, magic_coefficient=1, exclude_compartments=None, numba_option=False, use_shared_memory=False):
        super().__init__(configuration_matrix, outputs, volumes, numba_option, use_shared_memory)
        self.magic_coefficient = magic_coefficient
        self.exclude_compartments = np.array(exclude_compartments) if bool(exclude_compartments) else np.array([])
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
        
    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', max_step=0.01, **kwargs):
        x = super().optimize(method, user_method, method_is_func, optimization_func_name, max_step, **kwargs)
        if self.need_magic_optimization:
            self.magic_coefficient = x[-1]
            self.need_magic_optimization = False
        return x


class ReleaseCompartmentModel(BaseCompartmentModel):

    need_v_release_optimization = False
    release_parameters_target = None
    accumulation_parameters_target = None

    class ReleaseRK45(RK45):

        def __init__(self, fun, t0, y0, t_bound, release_function, compartment_number, c0, max_step=np.inf,
                     with_accumulate=False, accumulation_function=None, accumulation_type=1, rtol=1e-3, atol=1e-6, vectorized=False,
                     first_step=None,  **extraneous):
            super().__init__(fun, t0, y0, t_bound, max_step=max_step,
                     rtol=rtol, atol=atol, vectorized=vectorized,
                     first_step=first_step, **extraneous)
            self.release_function = release_function
            self.compartment_number = compartment_number
            self.c0 = c0
            self.old_release_correction = 0
            self.old_accumulation_correction = c0
            self.with_accumulate = with_accumulate
            self.accumulation_function = accumulation_function
            self.accumulation_type = accumulation_type

        def _step_impl(self):
            result = super()._step_impl()
            release_correction = self.release_function(self.t, self.c0)
            minus_accumulation = 0
            if self.with_accumulate:
                if self.accumulation_type == 1:
                    accumulation_correction = self.accumulation_function(self.t, self.c0 )
                    minus_accumulation = self.old_accumulation_correction - accumulation_correction
                elif self.accumulation_type == 2:
                    coef_accumulation = self.accumulation_function(self.t, self.c0) / self.c0
            plus_release = release_correction - self.old_release_correction
            all_corrections = plus_release
            if self.accumulation_type == 1:
                if plus_release - minus_accumulation > 0:
                    all_corrections = plus_release - minus_accumulation
            elif self.accumulation_type == 2:
                all_corrections *= coef_accumulation
            self.y[self.compartment_number] += all_corrections
            self.old_release_correction = release_correction
            if self.with_accumulate and self.accumulation_type == 1:
                self.old_accumulation_correction = accumulation_correction
            return result

    def __init__(self, v_release, release_parameters, release_compartment,
                 release_function=None, with_accumulate=False, accumulation_function=None,
                 accumulation_parameters=[None], accumulation_type=1, *args, **kwargs):
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
        self.with_accumulate = with_accumulate
        self.accumulation_type = accumulation_type

        if self.with_accumulate:
            self.accumulation_function = accumulation_function
            self.accumulation_parameters = np.array(accumulation_parameters)
            if np.any(self.accumulation_parameters == None):
                self.accumulation_parameters_target = np.where(self.accumulation_parameters == None)
                self.accumulation_parameters_target_count = np.sum(self.accumulation_parameters == None)

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
        if self.with_accumulate:
            if self.accumulation_parameters_target:
                s += self.release_parameters_target_count + int(self.need_v_release_optimization)
                self.accumulation_parameters[self.accumulation_parameters_target] = x[s:s + self.accumulation_parameters_target_count]

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

    def _default_accumulation_function(self, t, c0):
        """
        Функция для поправки на накопление
        """
        k, = self.accumulation_parameters
        return c0 * np.exp(-k * t)

    def get_accumulation_function(self):
        if self.accumulation_function is not None:
            return lambda t, c0: self.accumulation_function(t, c0, *self.accumulation_parameters)
        else:
            return self._default_accumulation_function

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
            fun=self._compartment_model if
            not self.numba_option
            else lambda t, c: self._numba_compartment_model(t, c,
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
            with_accumulate=self.with_accumulate,
            accumulation_function=self.get_accumulation_function() if self.with_accumulate else None,
            accumulation_type=self.accumulation_type,
            c0=c0
        )
        self.last_result.model_realized = c0 - self.get_release_function()(self.last_result.t, c0)
        if self.with_accumulate:
            model_accumulation = self.get_accumulation_function()(self.last_result.t, c0)
            if self.accumulation_type == 1:
                self.last_result.model_realized = model_accumulation - self.get_release_function()(self.last_result.t, c0)
            elif self.accumulation_type == 2:
                accumulation_coeffs = model_accumulation / c0
                self.last_result.model_realized= accumulation_coeffs * self.get_release_function()(self.last_result.t, c0)
                self.last_result.model_realized = model_accumulation - self.last_result.model_realized
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
            plus = 0
            if self.teoretic_realized is not None:
                model_realized = self.last_result.model_realized
                plus = np.sum(((model_realized - self.teoretic_realized) ** 2) / ((self.teoretic_realized - self.teoretic_realized_avg) ** 2))
            return plus + np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))
        elif metric == 'norm':
            plus = 0
            if self.teoretic_realized is not None:
                model_realized = self.last_result.model_realized
                plus = np.linalg.norm(self.teoretic_realized - model_realized)
            return plus + np.linalg.norm(target_results - self.teoretic_y)
        else:
            return np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))

    def load_optimization_data(self, teoretic_x, teoretic_y, know_compartments,
                               w = None, c0=None, d=None, compartment_number=None, teoretic_realized=None):
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
        self.teoretic_realized = np.array(teoretic_realized) if teoretic_realized is not None else teoretic_realized
        if teoretic_realized is not None:
            self.teoretic_realized_avg = np.average(self.teoretic_realized)
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

    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', max_step=0.01, **kwargs):
        x = super().optimize(method, user_method, method_is_func, optimization_func_name, max_step, **kwargs)
        s = self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count
        if self.release_parameters_target:
            self.release_parameters[self.release_parameters_target] = x[s:s + self.release_parameters_target_count]
        if self.need_v_release_optimization:
            self.v_release = x[s:s + self.release_parameters_target_count + 1]
            self.need_v_release_optimization = False
        if self.with_accumulate:
            if self.accumulation_parameters_target:
                s += self.release_parameters_target_count + int(self.need_v_release_optimization)
                self.accumulation_parameters[self.accumulation_parameters_target] = x[s:s + self.accumulation_parameters_target_count]
        return x

    def plot_model(self, compartment_numbers=None, compartment_names={},
                   left=None, right=None, y_lims={}, plot_accumulation=False, **kwargs):
        super().plot_model(compartment_numbers, compartment_names, left, right, y_lims, **kwargs)
        if plot_accumulation:
            if hasattr(self, "teoretic_x") and hasattr(self, "teoretic_realized"):
                plt.plot(self.teoretic_x, self.teoretic_realized, "*r")
            plt.plot(self.last_result.t, self.last_result.model_realized)
            plt.title(compartment_names.get('realized', 'realized'))
            plt.xlim(left=left, right=right)
            if y_lims.get('realized'):
                plt.ylim(y_lims.get('realized'))
            plt.grid()
            plt.show()


class TwoSubstancesCompartmentModel(MagicCompartmentModel):

    released_configuration_matrix_target = None
    released_outputs_target = None
    release_parameters_target = None
    has_teoretic_y = False
    has_teoretic_released = False

    class TwoSubstancesRK45(RK45):

        def __init__(self, fun, t0, y0, t_bound, release_function, max_step=np.inf,
                     rtol=1e-3, atol=1e-6, vectorized=False,
                     first_step=None,  **extraneous):
            self.old_corrections = 0
            super().__init__(fun, t0, y0, t_bound, max_step=max_step,
                     rtol=rtol, atol=atol, vectorized=vectorized,
                     first_step=first_step, **extraneous)
            self.release_function = release_function

        def _step_impl(self):
            result = super()._step_impl()
            release_correction: float = self.release_function(self.y, self.t)
            correction = release_correction - self.old_corrections
            c = self.y[:self.y.size // 2]
            real_release_correction = np.append(-1 * correction * c, correction * c)
            self.y += real_release_correction
            self.old_corrections = release_correction
            return result

    def __init__(self, configuration_matrix, outputs, released_configuration_matrix, released_outputs,
                 release_parameters, release_function=None,
                 volumes=None, magic_coefficient=1, exclude_compartments=None,
                 numba_option=False, use_shared_memory=False):
        super().__init__(
            configuration_matrix=configuration_matrix,
            outputs=outputs,
            volumes=volumes,
            magic_coefficient=magic_coefficient,
            exclude_compartments=exclude_compartments,
            numba_option=numba_option,
            use_shared_memory=use_shared_memory
        )
        self.released_configuration_matrix = np.array(released_configuration_matrix)
        self.released_configuration_matrix_target_count = 0
        if np.any(self.released_configuration_matrix == None):
            self.released_configuration_matrix_target = np.where(self.released_configuration_matrix == None)
            self.released_configuration_matrix_target_count = np.sum(self.released_configuration_matrix == None)
        self.released_outputs = np.array(released_outputs)
        self.released_outputs_target_count = 0
        if np.any(self.released_outputs == None):
            self.released_outputs_target = np.where(self.released_outputs == None)
            self.released_outputs_target_count = np.sum(self.released_outputs == None)
        self.release_parameters = np.array(release_parameters)
        if np.any(self.release_parameters == None):
            self.release_parameters_target = np.where(self.release_parameters == None)
            self.release_parameters_target_count = np.sum(self.release_parameters == None)
        self.release_function = release_function
        if getattr(self, "memory", None):
            self.memory.shm.close()
            self.memory.shm.unlink()
            self.memory_size += (self.release_parameters_target_count +
                                 self.released_configuration_matrix_target_count + self.released_outputs_target_count)
            self.memory = shared_memory.ShareableList(
                sequence=self.memory_size * [None]
            )
            self.memory_name = self.memory.shm.name

    def __call__(self, t_max, c0=None, d=None, compartment_number=None, max_step=0.01, t_eval=None, **kwargs):
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
            assert all([d, compartment_number is not None]), "Need to set d and compartment_number"
            c0 = np.zeros((2, self.outputs.size))
            c0[0][compartment_number] = d / self.volumes[compartment_number]
        else:
            c0 = np.array(c0)
        c0 = c0.reshape((1, c0.size))[0]
        ts = [0, t_max]
        res = solve_ivp(
            fun=self._compartment_model if
            not self.numba_option
            else lambda t, c: self._numba_compartment_model(t, c,
                                                             self.configuration_matrix.astype(
                                                                 np.float64),
                                                            self.released_configuration_matrix.astype(
                                                                np.float64),
                                                             self.outputs.astype(
                                                                 np.float64),
                                                            self.released_outputs.astype(
                                                                np.float64),
                                                             self.volumes.astype(
                                                     np.float64)),
            t_span=ts,
            y0=c0,
            max_step=max_step,
            t_eval=t_eval,
            method=self.TwoSubstancesRK45,
            release_function=self.get_release_function()
        )
        magic_arr = np.ones(self.configuration_matrix.shape[0] + self.released_configuration_matrix.shape[0]) * self.magic_coefficient
        if self.exclude_compartments.size:
            magic_arr[self.exclude_compartments] = 1
        magic_arr = np.repeat(magic_arr, res.y.shape[1])
        magic_arr = np.reshape(magic_arr, res.y.shape)
        res.y = magic_arr * res.y
        self.last_result = res
        return self.last_result

    def _default_release_function(self, current_c, t):
        """
        Функция для поправки на высвобождение
        """
        # c_resh = c.reshape((2, c.size // 2))
        # k = self.release_parameters[0]
        # return k * c_resh[0]
        m, b, c = self.release_parameters
        return c * t ** b / (t ** b + m)

    def get_release_function(self):
        if self.release_function is not None:
            return lambda c, t: self.release_function(c, t, *self.release_parameters)
        else:
            return self._default_release_function

    def _compartment_model(self, t, c):
        """
        Функция для расчета камерной модели

        Args:
            t: Текущее время
            c: Вектор концентраций

        Returns:
            Вектор изменений концентраций (c) в момент времени (t)
        """
        c_resh = c.reshape((2, c.size // 2))
        c_primary = c_resh[0]
        dc_dt = (self.configuration_matrix.T @ (c_primary * self.volumes) \
            - self.configuration_matrix.sum(axis=1) * (c_primary * self.volumes)\
            - self.outputs * (c_primary * self.volumes)) / self.volumes
        c_released = c_resh[1]
        dc_dt_released = (self.released_configuration_matrix.T @ (c_released * self.volumes) \
                 - self.released_configuration_matrix.sum(axis=1) * (c_released * self.volumes) \
                 - self.released_outputs * (c_released * self.volumes)) / self.volumes
        np.append(dc_dt, dc_dt_released)

    @staticmethod
    @njit
    def _numba_compartment_model(t, c, configuration_matrix, released_configuration_matrix, outputs, released_outputs, volumes):
        """
        Функция для расчета камерной модели

        Args:
            t: Текущее время
            c: Вектор концентраций

        Returns:
            Вектор изменений концентраций (c) в момент времени (t)
        """
        c_resh = c.reshape((2, c.size // 2))
        c_primary = c_resh[0]
        dc_dt = (configuration_matrix.T @ (c_primary * volumes) \
                 - configuration_matrix.sum(axis=1) * (c_primary * volumes) \
                 - outputs * (c_primary * volumes)) / volumes
        c_released = c_resh[1]
        dc_dt_released = (released_configuration_matrix.T @ (c_released * volumes) \
                          - released_configuration_matrix.sum(axis=1) * (c_released * volumes) \
                          - released_outputs * (c_released * volumes)) / volumes
        return np.append(dc_dt, dc_dt_released)

    def load_optimization_data(self, teoretic_x, teoretic_y=None, teoretic_released=None, know_compartments=None, know_released_compartments=None,
                               w=None, released_w=None, c0=None, d=None, compartment_number=None):
        self.has_teoretic_y = bool(teoretic_y)
        self.has_teoretic_released = bool(teoretic_released)
        self.teoretic_x = np.array(teoretic_x)
        if self.has_teoretic_y:
            self.teoretic_y = np.array(teoretic_y if teoretic_y is not None else [[]])
            self.know_compartments = know_compartments if know_compartments is not None else []
            self.teoretic_avg = np.average(self.teoretic_y, axis=1)
            self.teoretic_avg = np.repeat(self.teoretic_avg, self.teoretic_x.size)
            self.teoretic_avg = np.reshape(self.teoretic_avg, self.teoretic_y.shape)
            self.w = np.ones(self.teoretic_y.shape) if w is None else np.array(w)
        if self.has_teoretic_released:
            self.teoretic_released = np.array(teoretic_released if teoretic_released is not None else [[]])
            self.know_released_compartments = know_released_compartments if know_released_compartments is not None else []
            self.released_avg = np.average(self.teoretic_released, axis=1)
            self.released_avg = np.repeat(self.released_avg, self.teoretic_x.size)
            self.released_avg = np.reshape(self.released_avg, self.teoretic_released.shape)
            self.released_w = np.ones(self.teoretic_released.shape) if released_w is None else np.array(released_w)

        assert any([c0, d, compartment_number is not None]), "Need to set c0 or d and compartment_number"
        if not c0:
            assert all([d, compartment_number is not None]), "Need to set d and compartment_number"
            self.d = d
            self.compartment_number = compartment_number
            self.c0 = None
        else:
            self.c0 = np.array(c0)


    def load_data_from_list(self, x):
        super().load_data_from_list(x)
        n = self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count
        if self.released_configuration_matrix_target:
            self.released_configuration_matrix[self.released_configuration_matrix_target] = x[n:n + self.released_configuration_matrix_target_count]
        n += self.released_configuration_matrix_target_count
        if self.released_outputs_target:
            self.released_outputs[self.released_outputs_target] = x[n:n + self.released_outputs_target_count]
        n += self.released_outputs_target_count
        if self.release_parameters_target:
            self.release_parameters[self.release_parameters_target] = x[n:n + self.release_parameters_target_count]

    def _target_function(self, x, max_step=0.01, metric='R2'):
        self.load_data_from_list(x)
        c0 = self.c0
        if c0 is None:
            c0 = np.zeros((2, self.outputs.size))
            c0[0][self.compartment_number] = self.d / self.volumes[self.compartment_number]
        self(
            t_max=np.max(self.teoretic_x),
            c0=c0,
            t_eval=self.teoretic_x,
            max_step=max_step
        )
        if not self.last_result.success:
            return BIG_VALUE
        if self.has_teoretic_y:
            target_results = self.last_result.y[tuple(self.know_compartments), :]
        if self.has_teoretic_released:
            t = tuple([self.configuration_matrix.shape[0] + i for i in self.know_released_compartments])
            released_target_results = self.last_result.y[t, :]
        if metric == 'R2':
            if self.has_teoretic_y:
                a = np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum((self.teoretic_avg - self.teoretic_y) ** 2, axis=1))
            else:
                a = 0
            if self.has_teoretic_released:
                b = np.sum(np.sum(self.released_w * ((released_target_results - self.teoretic_released) ** 2), axis=1) / np.sum(
                    (self.released_avg - self.teoretic_released) ** 2, axis=1))
            else:
                b = 0
            return a + b
        elif metric == 'norm':
            if self.has_teoretic_y:
                a = np.linalg.norm(target_results - self.teoretic_y)
            else:
                a = 0
            if self.has_teoretic_released:
                b = np.linalg.norm(released_target_results - self.teoretic_released)
            else:
                b = 0
            return a + b
        else:
            if self.has_teoretic_y:
                a = np.sum(np.sum(self.w * ((target_results - self.teoretic_y) ** 2), axis=1) / np.sum(
                    (self.teoretic_avg - self.teoretic_y) ** 2, axis=1))
            else:
                a = 0
            if self.has_teoretic_released:
                b = np.sum(np.sum(self.released_w * ((released_target_results - self.teoretic_released) ** 2), axis=1) / np.sum(
                    (self.released_avg - self.teoretic_released) ** 2, axis=1))
            else:
                b = 0
            return a + b

    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', max_step=0.01, **kwargs):
        x = super().optimize(method, user_method, method_is_func, optimization_func_name, max_step, **kwargs)
        n = self.configuration_matrix_target_count + self.outputs_target_count + self.volumes_target_count + int(
            self.need_magic_optimization)
        if self.released_configuration_matrix_target:
            self.released_configuration_matrix[self.released_configuration_matrix_target] = x[n:n + self.released_configuration_matrix_target_count]
        n += self.released_configuration_matrix_target_count
        if self.released_outputs_target:
            self.released_outputs[self.released_outputs_target] = x[n: n + self.released_outputs_target_count]
        n += self.released_outputs_target_count
        if self.release_parameters_target:
            self.release_parameters[self.release_parameters_target] = x[n:n + self.release_parameters_target_count]
        self.released_configuration_matrix_target = None
        self.released_outputs_target = None
        return x

    def plot_model(self, compartment_numbers=None, released_compartment_numbers=None,
                   released_compartment_names=None, compartment_names=None,  left=None, right=None,
                   y_lims=None, released_y_lims=None, **kwargs):
        """
        Функция для построения графиков модели

        Args:
            compartment_numbers: Камеры, которые нужно отобразить (если не указать, отобразим все)
            compartment_names: Имена камер
        """
        super().plot_model(compartment_numbers=compartment_numbers, compartment_names=compartment_names,
                           left=left, right=right, y_lims=y_lims, **kwargs)

        if not released_compartment_numbers:
            released_compartment_numbers = []
        if not released_compartment_names:
            released_compartment_names = {}
        if not released_y_lims:
            released_y_lims = {}
        for i in released_compartment_numbers:
            j = i + self.outputs.size
            if hasattr(self, "teoretic_x") and hasattr(self, "teoretic_released") and i in self.know_released_compartments:
                plt.plot(self.teoretic_x, self.teoretic_released[self.know_compartments.index(i)], "*r")
            plt.plot(self.last_result.t, self.last_result.y[j])
            plt.title(released_compartment_names.get(i, j))
            plt.xlim(left=left, right=right)
            if released_y_lims.get(i):
                plt.ylim(released_y_lims.get(i))
            plt.grid()
            try:
                plt.show()
            except AttributeError:
                plt.savefig(f'{released_compartment_names.get(i, str(j))}.png')
                plt.cla()

