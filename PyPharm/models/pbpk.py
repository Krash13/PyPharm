import inspect
from copy import copy
from numbalsoda import lsoda_sig, lsoda
import datetime
import numpy as np
from scipy.integrate import solve_ivp, LSODA
from scipy.optimize import minimize
from PyPharm.algorithms.country_optimization import CountriesAlgorithm
from PyPharm.algorithms.country_optimization_v2 import CountriesAlgorithm_v2
from PyPharm.algorithms.genetic_optimization import GeneticAlgorithm
from PyPharm.constants import MODEL_CONST, ORGAN_NAMES
from numba import njit, types, cfunc
from numba.typed import Dict
import matplotlib.pyplot as plt

cnst_rat = copy(MODEL_CONST['rat'])
cnst_human = copy(MODEL_CONST['human'])



class PBPKmod:

    _organs = ['lung', 'heart', 'brain', 'muscle', 'adipose', 'skin', 'bone', 'kidney',
                  'liver', 'gut', 'spleen', 'stomach', 'pancreas', 'venous_blood', 'arterial_blood']
    _cl_organs = ['kidney', 'liver']
    
    def __init__(self, know_k=None, know_cl=None, numba_option=False, lsoda_option=False):

        self.know_k = know_k if know_k is not None else {}
        self.know_cl = know_cl if know_cl is not None else {}
        self._optim = False
        self.numba_option = numba_option
        self.lsoda_option = lsoda_option
        if numba_option:
            self.cnst_v_rat = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64
            )
            for k, v in cnst_rat.items():
                self.cnst_v_rat[k] = v['V']
            self.cnst_v_human = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64
            )
            for k, v in cnst_human.items():
                self.cnst_v_human[k] = v['V']
            self.cnst_q_rat = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64
            )
            for k, v in cnst_rat.items():
                if v.get('Q'):
                    self.cnst_q_rat[k] = v['Q']
            self.cnst_q_human = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.float64
            )
            for k, v in cnst_human.items():
                if v.get('Q'):
                    self.cnst_q_human[k] = v['Q']

    def load_optimization_data(self, time_exp, dict_c_exp, start_c_in_venous, is_human=False):
        self.time_exp = time_exp
        self.dict_c_exp = dict_c_exp
        self.start_c_in_venous = start_c_in_venous
        self.is_human = is_human

    def _get_sol_difurs(self):
        return self(max(self.time_exp), self.start_c_in_venous, self.is_human)

    def fitness(self, k_cl):

        self.k_cl = k_cl

        sol_difurs = self._get_sol_difurs()
        # Список для хранения результатов
        present_organs_indices = []

        # Проверяем, какие ключи из 'organs' есть в 'dict_n'
        for organ in self._organs:
            if organ in self.dict_c_exp:
                index = self._organs.index(organ)  # Получаем индекс органа в списке organs
                present_organs_indices.append((organ, index))

        rez_err = 0
        for organ, index in present_organs_indices:
            mean_y = sum(sol_difurs[index]) / len(sol_difurs[index])
            a = [(sol_difurs[index][self.time_exp[i]] - self.dict_c_exp[organ][i]) ** 2 for i in range(len(self.dict_c_exp[organ]))]
            a = sum(a)
            b = [(mean_y - self.dict_c_exp[organ][i]) ** 2 for i in
                 range(len(self.dict_c_exp[organ]))]
            b = sum(b)
            rez_err += a / b
            # rez_err += sum([abs(sol_difurs[:, index][self.time_exp[i]] - self.dict_c_exp[organ][i]) for i in
            #                 range(len(self.dict_c_exp[organ]))])

        return rez_err
    def _get_result(self, fun, t, max_time, K_CL, is_human):
        return solve_ivp(
            fun=fun,
            t_span=[0, max_time],
            y0=self.y0,
            t_eval=t,
            method=LSODA,
        )

    def _prepare_result(self, t, res):
        self._res = res.y
        self.last_result = {
            't': t * 60
        }
        for organ in self._organs:
            index = self._organs.index(organ)
            self.last_result[organ] = res.y[index]

    def __call__(self, max_time, start_c_in_venous, is_human=False, step=1):
        self.y0 = [0 for _ in range(15)]  # всего в модели 15 органов
        self.y0[-2] = start_c_in_venous
        t = np.linspace(0, max_time, max_time + 1 if self._optim else int(1 / step * max_time) + 1) / 60

        if not hasattr(self, 'k_cl'):
            self.k_cl = []

        full_k = []
        i = 0
        for name in self._organs:
            know_k = self.know_k.get(name)
            if know_k is not None:
                full_k.append(know_k)
            else:
                full_k.append(self.k_cl[i])
                i += 1
        full_cl = []

        for name in self._cl_organs:
            know_k = self.know_cl.get(name)
            if know_k is not None:
                full_cl.append(know_k)
            else:
                full_cl.append(self.k_cl[i])
                i += 1
        if not self.numba_option and not self.lsoda_option:
            res = self._get_result(
                fun=lambda time, y: self.fullPBPKmodel(y, time, [*full_k, *full_cl], is_human),
                t=t,
                max_time=max_time,
                K_CL=[*full_k, *full_cl],
                is_human=is_human
            )
        elif self.lsoda_option:
            if is_human:
                cnst_v = self.cnst_v_human
                cnst_q = self.cnst_q_human
            else:
                cnst_v = self.cnst_v_rat
                cnst_q = self.cnst_q_rat
            f = self.make_lsoda_func(cnst_q, cnst_v)
            res, success = lsoda(
                funcptr=f.address,
                u0=self.y0,
                t_eval=t,
                data=np.array([*full_k, *full_cl])
            )
        else:
            k_cl = np.array([*full_k, *full_cl])
            if is_human:
                cnst_v = self.cnst_v_human
                cnst_q = self.cnst_q_human
            else:
                cnst_v = self.cnst_v_rat
                cnst_q = self.cnst_q_rat
            function = lambda time, c: self.numba_fullPBPK_for_optimization(
                y=c,
                t=time,
                K_CL=k_cl.astype(np.float64),
                cnst_q=cnst_q,
                cnst_v=cnst_v
            )
            res = self._get_result(
                fun=function,
                t=t,
                max_time=max_time,
                K_CL=[*full_k, *full_cl],
                is_human=is_human
            )
        self._prepare_result(t, res)
        return self._res

    def plot_last_result(self, organ_names=[], left=None, right=None, user_names={}, theoretic_data={}, y_lims={}):
        if hasattr(self, 'last_result'):
            for name in organ_names:
                if theoretic_data.get(name):
                    plt.plot(theoretic_data[name]['x'], theoretic_data[name]['y'], '*r')
                plt.plot(
                    self.last_result['t'],
                    self.last_result.get(name),
                )
                plt.title(user_names.get(name, name))
                plt.xlim(left=left, right=right)
                if y_lims.get(name):
                    plt.ylim(y_lims.get(name))
                plt.grid()
                plt.show()

    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', **kwargs):
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
        f = lambda x: self.fitness(x)
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
        self._optim = False
        return x

    def update_know_params(self, k_cl=None):
        if k_cl:
            i = 0
            for name in self._organs:
                know_k = self.know_k.get(name)
                if know_k is None:
                    self.know_k[name] = k_cl[i]
                    i += 1
            for name in self._cl_organs:
                know_cl = self.know_cl.get(name)
                if know_cl is None:
                    self.know_cl[name] = k_cl[i]
                    i += 1

    def get_unknown_params(self):
        result = []
        for name in self._organs:
            know_k = self.know_k.get(name)
            if know_k is None:
                result.append(f"k_{name}")
        for name in self._cl_organs:
            know_cl = self.know_cl.get(name)
            if know_cl is None:
                result.append(f"cl_{name}")
        return result

    def fullPBPKmodel(self, y, t, K_CL, is_human=False):  # V, Q, K, CL):
        # 15 органов
        if is_human:
            cnst = cnst_human
        else:
            cnst = cnst_rat
        C_lung, C_heart, C_brain, C_muscle, C_fat, C_skin, C_bone, \
            C_kidney, C_liver, C_gut, C_spleen, C_stomach, C_pancreas, C_V, C_A = y

        K_lung, K_heart, K_brain, K_muscle, K_fat, K_skin, K_bone, \
            K_kidney, K_liver, K_gut, K_spleen, K_stomach, K_pancreas, K_liver_cl, K_kidney_cl = K_CL[:15]
        CL_kidney, CL_liver = K_CL[15:]

        dC_lung_dt = cnst['lung']['Q'] * (C_V - C_lung / K_lung) / cnst['lung']['V']
        dC_heart_dt = cnst['heart']['Q'] * (C_A - C_heart / K_heart) / cnst['heart']['V']
        dC_brain_dt = cnst['brain']['Q'] * (C_A - C_brain / K_brain) / cnst['brain']['V']
        dC_muscle_dt = cnst['muscle']['Q'] * (C_A - C_muscle / K_muscle) / cnst['muscle']['V']
        dC_fat_dt = cnst['adipose']['Q'] * (C_A - C_fat / K_fat) / cnst['adipose']['V']
        dC_skin_dt = cnst['skin']['Q'] * (C_A - C_skin / K_skin) / cnst['skin']['V']
        dC_bone_dt = cnst['bone']['Q'] * (C_A - C_bone / K_bone) / cnst['bone']['V']
        # Kidney        V(Kidney)*dC(Kidney)/dt = Q(Kidney)*C(A)-Q(Kidney)*CV(Kidney)-CL(Kidney,int)*CV(Kidney,int)?
        dC_kidney_dt = (cnst['kidney']['Q'] * (C_A - C_kidney / K_kidney) - CL_kidney * C_kidney / K_kidney_cl) / \
                       cnst['kidney']['V']  # ???

        # Liver         V(Liver)*dC(Liver)/dt = (Q(Liver)-Q(Spleen)-Q(Gut)-Q(Pancreas)-Q(Stomach))*C(A) + Q(Spleen)*CV(Spleen) +
        #                                     + Q(Gut)*CV(Gut) + Q(Pancreas)*CV(Pancreas) + Q(Stomach)*CV(Stomach) -
        #                                     - Q(Liver)*CV(Liver) - CL(Liver,int)*CV(Liver,int)? # тут скорее всего нужно вычитать потоки из друг друга дополнительно по крови что бы сохранить массовый баланс
        Q_liver_in_from_art = cnst['liver']['Q'] - cnst['gut']['Q'] - cnst['spleen']['Q'] - \
                              cnst['pancreas']['Q'] - cnst['stomach']['Q']
        dC_liver_dt = (
                              Q_liver_in_from_art * C_A + cnst['gut']['Q'] * C_gut / K_gut
                              + cnst['spleen']['Q'] * C_spleen / K_spleen
                              + cnst['stomach']['Q'] * C_stomach / K_stomach
                              + cnst['pancreas']['Q'] * C_pancreas / K_pancreas
                              - cnst['liver']['Q'] * C_liver / K_liver
                              - CL_liver * C_liver / K_liver_cl  # ???
                      ) / cnst['liver']['V']

        dC_gut_dt = cnst['gut']['Q'] * (C_A - C_gut / K_gut) / cnst['gut']['V']
        dC_spleen_dt = cnst['spleen']['Q'] * (C_A - C_spleen / K_spleen) / cnst['spleen']['V']
        dC_stomach_dt = cnst['stomach']['Q'] * (C_A - C_stomach / K_stomach) / cnst['stomach']['V']
        dC_pancreas_dt = cnst['pancreas']['Q'] * (C_A - C_pancreas / K_pancreas) / cnst['pancreas']['V']

        dC_venouse_dt = (
                                cnst['heart']['Q'] * C_heart / K_heart
                                + cnst['brain']['Q'] * C_brain / K_brain
                                + cnst['muscle']['Q'] * C_muscle / K_muscle
                                + cnst['skin']['Q'] * C_skin / K_skin
                                + cnst['adipose']['Q'] * C_fat / K_fat
                                + cnst['bone']['Q'] * C_bone / K_bone
                                + cnst['kidney']['Q'] * C_kidney / K_kidney
                                + cnst['liver']['Q'] * C_liver / K_liver
                                - cnst['lung']['Q'] * C_V
                        ) / cnst['venous_blood']['V']

        dC_arterial_dt = cnst['lung']['Q'] * (C_lung / K_lung - C_A) / cnst['arterial_blood']['V']

        y_new = [dC_lung_dt, dC_heart_dt, dC_brain_dt, dC_muscle_dt, dC_fat_dt, dC_skin_dt, dC_bone_dt, \
                 dC_kidney_dt, dC_liver_dt, dC_gut_dt, dC_spleen_dt, dC_stomach_dt, dC_pancreas_dt, dC_venouse_dt,
                 dC_arterial_dt]
        return y_new

    @staticmethod
    @njit
    def numba_fullPBPK_for_optimization(y, t, K_CL, cnst_q, cnst_v):
        C_lung, C_heart, C_brain, C_muscle, C_fat, C_skin, C_bone, \
            C_kidney, C_liver, C_gut, C_spleen, C_stomach, C_pancreas, C_V, C_A = y

        K_lung, K_heart, K_brain, K_muscle, K_fat, K_skin, K_bone, \
            K_kidney, K_liver, K_gut, K_spleen, K_stomach, K_pancreas, K_liver_cl, K_kidney_cl = K_CL[:15]
        CL_kidney, CL_liver = K_CL[15:]

        dC_lung_dt = cnst_q['lung'] * (C_V - C_lung / K_lung) / cnst_v['lung']
        dC_heart_dt = cnst_q['heart'] * (C_A - C_heart / K_heart) / cnst_v['heart']
        dC_brain_dt = cnst_q['brain'] * (C_A - C_brain / K_brain) / cnst_v['brain']
        dC_muscle_dt = cnst_q['muscle'] * (C_A - C_muscle / K_muscle) / cnst_v['muscle']
        dC_fat_dt = cnst_q['adipose'] * (C_A - C_fat / K_fat) / cnst_v['adipose']
        dC_skin_dt = cnst_q['skin'] * (C_A - C_skin / K_skin) / cnst_v['skin']
        dC_bone_dt = cnst_q['bone'] * (C_A - C_bone / K_bone) / cnst_v['bone']
        # Kidney        V(Kidney)*dC(Kidney)/dt = Q(Kidney)*C(A)-Q(Kidney)*CV(Kidney)-CL(Kidney,int)*CV(Kidney,int)?
        dC_kidney_dt = (cnst_q['kidney'] * (C_A - C_kidney / K_kidney) - CL_kidney * C_kidney / K_kidney_cl) / \
                       cnst_v['kidney']  # ???

        # Liver         V(Liver)*dC(Liver)/dt = (Q(Liver)-Q(Spleen)-Q(Gut)-Q(Pancreas)-Q(Stomach))*C(A) + Q(Spleen)*CV(Spleen) +
        #                                     + Q(Gut)*CV(Gut) + Q(Pancreas)*CV(Pancreas) + Q(Stomach)*CV(Stomach) -
        #                                     - Q(Liver)*CV(Liver) - CL(Liver,int)*CV(Liver,int)? # тут скорее всего нужно вычитать потоки из друг друга дополнительно по крови что бы сохранить массовый баланс
        Q_liver_in_from_art = cnst_q['liver'] - cnst_q['gut'] - cnst_q['spleen'] - \
                              cnst_q['pancreas'] - cnst_q['stomach']
        dC_liver_dt = (
                              Q_liver_in_from_art * C_A + cnst_q['gut'] * C_gut / K_gut
                              + cnst_q['spleen'] * C_spleen / K_spleen
                              + cnst_q['stomach'] * C_stomach / K_stomach
                              + cnst_q['pancreas'] * C_pancreas / K_pancreas
                              - cnst_q['liver'] * C_liver / K_liver
                              - CL_liver * C_liver / K_liver_cl  # ???
                      ) / cnst_v['liver']

        dC_gut_dt = cnst_q['gut'] * (C_A - C_gut / K_gut) / cnst_v['gut']
        dC_spleen_dt = cnst_q['spleen'] * (C_A - C_spleen / K_spleen) / cnst_v['spleen']
        dC_stomach_dt = cnst_q['stomach'] * (C_A - C_stomach / K_stomach) / cnst_v['stomach']
        dC_pancreas_dt = cnst_q['pancreas'] * (C_A - C_pancreas / K_pancreas) / cnst_v['pancreas']

        dC_venouse_dt = (
                                cnst_q['heart'] * C_heart / K_heart
                                + cnst_q['brain'] * C_brain / K_brain
                                + cnst_q['muscle'] * C_muscle / K_muscle
                                + cnst_q['skin'] * C_skin / K_skin
                                + cnst_q['adipose'] * C_fat / K_fat
                                + cnst_q['bone'] * C_bone / K_bone
                                + cnst_q['kidney'] * C_kidney / K_kidney
                                + cnst_q['liver'] * C_liver / K_liver
                                - cnst_q['lung'] * C_V
                        ) / cnst_v['venous_blood']

        dC_arterial_dt = cnst_q['lung'] * (C_lung / K_lung - C_A) / cnst_v['arterial_blood']

        y_new = np.array([dC_lung_dt, dC_heart_dt, dC_brain_dt, dC_muscle_dt, dC_fat_dt, dC_skin_dt, dC_bone_dt, \
                 dC_kidney_dt, dC_liver_dt, dC_gut_dt, dC_spleen_dt, dC_stomach_dt, dC_pancreas_dt, dC_venouse_dt,
                 dC_arterial_dt]).astype(np.float64)
        return y_new

    @staticmethod
    def make_lsoda_func(cnst_q, cnst_v):
        @cfunc(lsoda_sig)
        def lsoda_fullPBPK_for_optimization(t, y, y_new, K_CL):
            C_lung, C_heart, C_brain, C_muscle, C_fat, C_skin, C_bone, \
                C_kidney, C_liver, C_gut, C_spleen, C_stomach, C_pancreas, C_V, C_A = y

            K_lung, K_heart, K_brain, K_muscle, K_fat, K_skin, K_bone, \
                K_kidney, K_liver, K_gut, K_spleen, K_stomach, K_pancreas, K_liver_cl, K_kidney_cl = K_CL[:15]
            CL_kidney, CL_liver = K_CL[15:]

            dC_lung_dt = cnst_q['lung'] * (C_V - C_lung / K_lung) / cnst_v['lung']
            dC_heart_dt = cnst_q['heart'] * (C_A - C_heart / K_heart) / cnst_v['heart']
            dC_brain_dt = cnst_q['brain'] * (C_A - C_brain / K_brain) / cnst_v['brain']
            dC_muscle_dt = cnst_q['muscle'] * (C_A - C_muscle / K_muscle) / cnst_v['muscle']
            dC_fat_dt = cnst_q['adipose'] * (C_A - C_fat / K_fat) / cnst_v['adipose']
            dC_skin_dt = cnst_q['skin'] * (C_A - C_skin / K_skin) / cnst_v['skin']
            dC_bone_dt = cnst_q['bone'] * (C_A - C_bone / K_bone) / cnst_v['bone']
            # Kidney        V(Kidney)*dC(Kidney)/dt = Q(Kidney)*C(A)-Q(Kidney)*CV(Kidney)-CL(Kidney,int)*CV(Kidney,int)?
            dC_kidney_dt = (cnst_q['kidney'] * (C_A - C_kidney / K_kidney) - CL_kidney * C_kidney / K_kidney_cl) / \
                           cnst_v['kidney']  # ???

            # Liver         V(Liver)*dC(Liver)/dt = (Q(Liver)-Q(Spleen)-Q(Gut)-Q(Pancreas)-Q(Stomach))*C(A) + Q(Spleen)*CV(Spleen) +
            #                                     + Q(Gut)*CV(Gut) + Q(Pancreas)*CV(Pancreas) + Q(Stomach)*CV(Stomach) -
            #                                     - Q(Liver)*CV(Liver) - CL(Liver,int)*CV(Liver,int)? # тут скорее всего нужно вычитать потоки из друг друга дополнительно по крови что бы сохранить массовый баланс
            Q_liver_in_from_art = cnst_q['liver'] - cnst_q['gut'] - cnst_q['spleen'] - \
                                  cnst_q['pancreas'] - cnst_q['stomach']
            dC_liver_dt = (
                                  Q_liver_in_from_art * C_A + cnst_q['gut'] * C_gut / K_gut
                                  + cnst_q['spleen'] * C_spleen / K_spleen
                                  + cnst_q['stomach'] * C_stomach / K_stomach
                                  + cnst_q['pancreas'] * C_pancreas / K_pancreas
                                  - cnst_q['liver'] * C_liver / K_liver
                                  - CL_liver * C_liver / K_liver_cl  # ???
                          ) / cnst_v['liver']

            dC_gut_dt = cnst_q['gut'] * (C_A - C_gut / K_gut) / cnst_v['gut']
            dC_spleen_dt = cnst_q['spleen'] * (C_A - C_spleen / K_spleen) / cnst_v['spleen']
            dC_stomach_dt = cnst_q['stomach'] * (C_A - C_stomach / K_stomach) / cnst_v['stomach']
            dC_pancreas_dt = cnst_q['pancreas'] * (C_A - C_pancreas / K_pancreas) / cnst_v['pancreas']

            dC_venouse_dt = (
                                    cnst_q['heart'] * C_heart / K_heart
                                    + cnst_q['brain'] * C_brain / K_brain
                                    + cnst_q['muscle'] * C_muscle / K_muscle
                                    + cnst_q['skin'] * C_skin / K_skin
                                    + cnst_q['adipose'] * C_fat / K_fat
                                    + cnst_q['bone'] * C_bone / K_bone
                                    + cnst_q['kidney'] * C_kidney / K_kidney
                                    + cnst_q['liver'] * C_liver / K_liver
                                    - cnst_q['lung'] * C_V
                            ) / cnst_v['venous_blood']

            dC_arterial_dt = cnst_q['lung'] * (C_lung / K_lung - C_A) / cnst_v['arterial_blood']

            y_new = np.array([dC_lung_dt, dC_heart_dt, dC_brain_dt, dC_muscle_dt, dC_fat_dt, dC_skin_dt, dC_bone_dt, \
                     dC_kidney_dt, dC_liver_dt, dC_gut_dt, dC_spleen_dt, dC_stomach_dt, dC_pancreas_dt, dC_venouse_dt,
                     dC_arterial_dt]).astype(np.float64)
            return y_new
        return lsoda_fullPBPK_for_optimization

class ReleasePBPKmod(PBPKmod):
    
    @staticmethod
    def ode_release(solver, t, y0, release_function, d, v):
        result = []
        new_y0 = y0
        old_release_correction = 0
        for i in range(1, len(t)):
            res = solver(new_y0, t[i - 1], t[i])
            y = res.y
            release_correction = release_function(t[i], d)
            plus_release = release_correction - old_release_correction
            all_corrections = plus_release
            y[-2][1] += all_corrections / v
            old_release_correction = release_correction
            if i == 1:
                result.append([y[i][0] for i in range(y.shape[0])])
            new_y0 = np.array([y[i][1] for i in range(y.shape[0])])
            result.append(new_y0)
        return np.array(result).T
    class ReleaseLSODA(LSODA):

        def __init__(self, fun, t0, y0, t_bound, release_function, d, v, max_step=np.inf,
                     rtol=1e-3, atol=1e-6, vectorized=False,
                     first_step=None,  **extraneous):
            self.release_function = release_function
            self.d = d
            self.old_release_correction = 0
            self.v = v
            # release_correction = self.release_function(max_step / 60, self.d)
            # plus_release = release_correction - self.old_release_correction
            # all_corrections = plus_release
            # y0[-2] = 1
            # self.old_release_correction = release_correction
            super().__init__(fun, t0, y0, t_bound, max_step=max_step,
                     rtol=rtol, atol=atol, vectorized=vectorized,
                     first_step=first_step, **extraneous)


        def _step_impl(self):
            result = super()._step_impl()
            release_correction = self.release_function(self.t, self.d)
            plus_release = release_correction - self.old_release_correction
            all_corrections = plus_release
            self.y[-2] += all_corrections / self.v
            self._lsoda_solver._y[-2] += all_corrections / self.v
            self.old_release_correction = release_correction
            return result

    def __init__(self, release_parameters: dict=None, release_function: callable=None, know_k=None, know_cl=None, numba_option=False):
        super().__init__(
            know_k=know_k, know_cl=know_cl, numba_option=numba_option
        )
        self.release_function = release_function
        if release_parameters is None:
            self.release_parameters = {}
        else:
            self.release_parameters = release_parameters
        self.know_release_parameters = set(self.release_parameters.keys())

    @staticmethod
    def _default_release_function(t, d, m, b, c):
        """
        Функция для поправки на высвобождение
        """
        return d * c * t ** b / (t ** b + m)

    def get_release_function(self):
        return lambda t, d: self._get_release_function()(t, d, **self.release_parameters)

    def _get_release_function(self):
        if self.release_function is not None:
            return self.release_function
        else:
            return self._default_release_function

    @property
    def _release_parameters_list(self) -> list[str]:
        method = self._get_release_function()
        arguments = inspect.getfullargspec(method).args
        return [arg for arg in arguments if arg not in {'t', 'd'}]

    def get_unknown_params(self):
        result = super().get_unknown_params()
        arguments = self._release_parameters_list
        for arg in arguments:
            know_arg = self.release_parameters.get(arg)
            if know_arg is None:
                result.append(f"release_{arg}")
        return result

    def update_know_params(self, k_cl=None, release_parameters=None):
        super().update_know_params(k_cl)
        if release_parameters is not None:
            self.release_parameters = release_parameters
            self.know_release_parameters = set(self.release_parameters.keys())

    def _get_sol_difurs(self):
        return self(max(self.time_exp), self.d, self.is_human)

    def fitness(self, x):

        n = len(self._organs) + len(self._cl_organs) - len(self.know_cl) - len(self.know_k)
        self.k_cl = x[:n]
        i = 0
        for arg in self._release_parameters_list:
            if not arg in self.know_release_parameters:
                self.release_parameters[arg] = x[n + i]
                i += 1
        return super().fitness(self.k_cl)

    def _get_result(self, fun, t, max_time, K_CL, is_human):
        solver = lambda y0, t_left, t_right: solve_ivp(
            fun=lambda time, y: self.fullPBPKmodel(y, time, K_CL, is_human),
            t_span=[t_left, t_right],
            y0=y0,
            t_eval=np.array([t_left, t_right]),
            method=LSODA
        )
        return self.ode_release(solver, t, self.y0, d=self.d, v=self.v, release_function=self.get_release_function())

    def _prepare_result(self, t, res):
        self._res = res
        self.last_result = {
            't': t * 60
        }
        for organ in self._organs:
            index = self._organs.index(organ)
            self.last_result[organ] = res[index]

    def __call__(self, max_time, d, is_human=False, step=1):
        self.d = d
        self.v = cnst_human['venous_blood']['V'] if is_human else cnst_rat['venous_blood']['V']
        return super().__call__(
            max_time=max_time,
            start_c_in_venous=0,
            is_human=is_human,
            step=step
        )

    def optimize(self, method=None, user_method=None, method_is_func=True,
                 optimization_func_name='__call__', **kwargs):

        return super().optimize(
            method=method, user_method=user_method, method_is_func=method_is_func,
            optimization_func_name=optimization_func_name, **kwargs
        )

    def load_optimization_data(self, time_exp, dict_c_exp, d, is_human=False):
        self.time_exp = time_exp
        self.dict_c_exp = dict_c_exp
        self.d = d
        self.is_human = is_human


k_cl = [3.99830945e-02, 2.52574960e-04, 1.67152209e-03, 9.99984743e-01
, 6.46468560e-01, 9.99984743e-01, 9.99923713e-01, 4.31106889e-03
, 1.79421350e-01, 4.35664912e-04, 2.14452369e-02, 2.22059968e-04
, 6.18754864e-04, 6.78692392e-01, 9.23025933e-01, 2.33676814e-02
, 6.34012360e-04]

model = PBPKmod(lsoda_option=True)
model.update_know_params(k_cl)
model(24 * 60, 150 * 1e-3 / (20 * 10e-3))
# s = datetime.datetime.now()
# model.load_optimization_data(
#     time_exp=[5, 60, 180, 5 * 60, 15 * 60, 24 * 60],
#     dict_c_exp = {
#         ORGAN_NAMES.LIVER: [150.94 * 1e-6, 142.60 * 1e-6, 128.68 * 1e-6, 110.05 * 1e-6, 67.12 * 1e-6, 43.32 * 1e-6],
#         ORGAN_NAMES.LUNG: [29.42 * 1e-6, 36.97 * 1e-6, 24.30 * 1e-6, 23.22 * 1e-6, 9.74 * 1e-6, 1.75 * 1e-6],
#         ORGAN_NAMES.SPLEEN: [15.22 * 1e-6, 19.30 * 1e-6, 17.13 * 1e-6, 16.21 * 1e-6, 4.90 * 1e-6, 1.52 * 1e-6]
#     },
#     d=150 * 1e-3 / (20 * 10e-3)
# )
# result = model.optimize(
#     method='GA',
#     x_min=[0.001, 0.001, 0],
#     x_max=[10, 10, 1],
#     genes=3 * [16],
# 	n=10,
#     child_percent=0.3,
#     mutation_chance=0.5,
#     max_mutation=5,
#     t_max=100,
#     printing=True,
# )
# model.update_know_params(k_cl)
#
# # res = model.test(max_time=24 * 60, start_c_in_venous=150 * 1e-3 / (20 * 10e-3) / MODEL_CONST['rat']['venous_blood']['V'])
# res2 = model(max_time=24 * 60, start_c_in_venous=150 * 1e-3 / (20 * 10e-3) / MODEL_CONST['rat']['venous_blood']['V'])
# print(res2)