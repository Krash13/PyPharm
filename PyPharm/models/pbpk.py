from multiprocessing import shared_memory
import datetime
import numpy as np
from scipy.integrate import solve_ivp, RK45, odeint
from scipy.integrate import simps
from scipy.optimize import minimize
from PyPharm.algorithms.country_optimization import CountriesAlgorithm
from PyPharm.algorithms.country_optimization_v2 import CountriesAlgorithm_v2
from PyPharm.algorithms.genetic_optimization import GeneticAlgorithm
from PyPharm.constants import MODEL_CONST, ORGAN_NAMES
from numba import njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt

cnst_rat = MODEL_CONST['rat']
cnst_human = MODEL_CONST['human']


class PBPKmod:

    _organs = ['lung', 'heart', 'brain', 'muscle', 'adipose', 'skin', 'bone', 'kidney',
                  'liver', 'gut', 'spleen', 'stomach', 'pancreas', 'venous_blood', 'arterial_blood']
    _cl_organs = ['kidney', 'liver']
    
    def __init__(self, know_k={}, know_cl={}, numba_option=False):
        self.know_k = know_k
        self.know_cl = know_cl
        self._optim = False
        self.numba_option = numba_option
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

    def fitness(self, k_cl):

        self.k_cl = k_cl

        sol_difurs = self(max(self.time_exp), self.start_c_in_venous, self.is_human)
        # Список для хранения результатов
        present_organs_indices = []

        # Проверяем, какие ключи из 'organs' есть в 'dict_n'
        for organ in self._organs:
            if organ in self.dict_c_exp:
                index = self._organs.index(organ)  # Получаем индекс органа в списке organs
                present_organs_indices.append((organ, index))

        rez_err = 0
        for organ, index in present_organs_indices:
            mean_y = sum(sol_difurs[:, index]) / len(sol_difurs[:, index])
            a = [(sol_difurs[:, index][self.time_exp[i]] - self.dict_c_exp[organ][i]) ** 2 for i in range(len(self.dict_c_exp[organ]))]
            a = sum(a)
            b = [(mean_y - self.dict_c_exp[organ][i]) ** 2 for i in
                 range(len(self.dict_c_exp[organ]))]
            b = sum(b)
            rez_err += a / b
            # rez_err += sum([abs(sol_difurs[:, index][self.time_exp[i]] - self.dict_c_exp[organ][i]) for i in
            #                 range(len(self.dict_c_exp[organ]))])

        return rez_err
    
    def __call__(self, max_time, start_c_in_venous, is_human=False, step=1):
        self.y0 = [0 for _ in range(15)]  # всего в модели 15 органов
        self.y0[-2] = start_c_in_venous
        t = np.linspace(0, max_time, max_time + 1 if self._optim else int(1 / step * max_time) + 1)
        
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
        if not self.numba_option:
            sol_difurs = odeint(
                self.fullPBPKmodel,
                self.y0,
                t,
                args=([*full_k, *full_cl], is_human)
            )
        else:
            k_cl = np.array([*full_k, *full_cl])
            if is_human:
                cnst_v = self.cnst_v_human
                cnst_q = self.cnst_q_human
            else:
                cnst_v = self.cnst_v_rat
                cnst_q = self.cnst_q_rat
            function = lambda c, t: self.numba_fullPBPK_for_optimization(
                y=c,
                t=t,
                K_CL=k_cl.astype(np.float64),
                cnst_q=cnst_q,
                cnst_v=cnst_v
            )
            sol_difurs = odeint(
                function,
                self.y0,
                t
            )
        if self._optim:
            return sol_difurs

        self.last_result = {
            't': t
        }
        for organ in self._organs:
            index = self._organs.index(organ)
            self.last_result[organ] = np.array([sol_difurs[i][index] for i in range(t.size)])
        return self.last_result

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

    def update_know_params(self, k_cl):
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


model = PBPKmod(numba_option=True)
print(model.get_unknown_params())
model.load_optimization_data(
    time_exp=[12, 60, 3 * 60, 5* 60, 15 * 60, 24 * 60],
    dict_c_exp = {ORGAN_NAMES.LIVER: [103.2 * 1e-6, 134.54 * 1e-6, 87.89 * 1e-6, 81.87 * 1e-6, 45.83 * 1e-6, 28.48 * 1e-6],
              ORGAN_NAMES.LUNG: [26.96 * 1e-6, 22.67 * 1e-6, 15.51 * 1e-6, 12.07 * 1e-6, 4.53 * 1e-6, 0 * 1e-6],
              ORGAN_NAMES.SPLEEN: [11.84 * 1e-6, 12.22 * 1e-6, 8.52 * 1e-6, 7.01 * 1e-6, 3.65 * 1e-6, 2.16 * 1e-6]
    },
    start_c_in_venous=150 * 1e-3 / MODEL_CONST['rat']['venous_blood']['V']
)

result = model.optimize(
    method='country_optimization',
    Xmin=17 * [0.0001],
    Xmax=17 * [5],
    M=20,
    N=25,
    n=[1, 10],
    p=[0.00001, 2],
    m=[1, 8],
    k=8,
    l=3,
    ep=[0.2, 0.4],
    tmax=3,
    printing=True,
)
model.update_know_params(result)

result = model(max_time=24 * 60, start_c_in_venous=150 * 1e-3 / MODEL_CONST['rat']['venous_blood']['V'], step=0.1)

model_furs = PBPKmod(numba_option=True)

print()