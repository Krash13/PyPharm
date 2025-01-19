import random
import numpy as np
import copy
from operator import attrgetter
from math import ceil, cos, sin
import graycode

def rand_str():
    s = 'qwertyuiopasdfghjklzxcvbnm'
    k = np.random.choice(len(s), 8, replace=False)
    res = ''
    for ki in k:
        res += s[ki]
    return res

def check_population(country):
    return bool(country.population.size)

vector_check_population = np.vectorize(check_population, signature='()->()')

class Individual:

    def __init__(self, gray_code, x_min, x_max, genes, function):

        self.x_min = x_min
        self.x_max = x_max
        self.genes = genes
        self.function = function
        self.steps = (self.x_max - self.x_min) / (2 ** (self.genes) - 1)
        self.gray_code = gray_code
        self.f = self.function(self.real_x)
        self.ep_n = 0

    @property
    def real_x(self):
        decimal_x = []
        gray_code = copy.deepcopy(self.gray_code)
        for g in self.genes:
            decimal_x.append(graycode.gray_code_to_tc(int(gray_code[:g], base=2)))
            gray_code = gray_code[g:]
        return self.x_min + self.steps * np.array(decimal_x)

    @property
    def decimal_x(self):
        return (self.real_x - self.x_min) / self.steps

    @classmethod
    def generate_from_decimal(cls, decimal, x_min, x_max, genes, function):
        gray_code = ''
        decimal = np.clip(decimal, np.zeros(decimal.size), 2 ** (genes) - 1)
        for i in range(len(decimal)):
            s = '{:0' + str(genes[i]) + 'b}'
            gray_code += s.format(graycode.tc_to_gray_code(int(decimal[i])))

        return cls(gray_code, x_min, x_max, genes, function)

    def mutation(self, max_mutation=1):
        n = max(0, max_mutation - self.ep_n)
        if n:
            gray_code = copy.deepcopy(self.gray_code)
            k = np.random.choice(len(gray_code), n, replace=False)
            for ki in k:
                b = bool(int(gray_code[ki]))
                b = not b
                b = str(int(b))
                gray_code = gray_code[:ki] + b + gray_code[ki+1:]
            return Individual(gray_code, self.x_min, self.x_max, self.genes, self.function)
        else:
            return self

    @classmethod
    def crossover(cls, individual1, individual2):
        k = np.random.randint(1, len(individual1.gray_code) - 1, 2)
        while k[0] == k[1]:
            k = np.random.randint(1, len(individual1.gray_code) - 1, 2)
        k = np.sort(k)
        new_gray1 = individual1.gray_code[:k[0]] + individual2.gray_code[k[0]:k[1]] + individual1.gray_code[k[1]:]
        new_gray2 = individual2.gray_code[:k[0]] + individual1.gray_code[k[0]:k[1]] + individual2.gray_code[k[1]:]
        return [
            cls(new_gray1, individual1.x_min, individual1.x_max, individual1.genes, individual1.function),
            cls(new_gray2, individual1.x_min, individual1.x_max, individual1.genes, individual1.function),
        ]

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f


class Country:

    def __init__(self, N, x_min, x_max, genes, function):
        self.f = function
        self.N = N
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.genes = np.array(genes)
        self.name = rand_str()


        local_xmin = np.random.randint(np.zeros(self.genes.size), 2 ** self.genes - 1)
        local_xmax = np.random.randint(local_xmin, 2 ** self.genes)

        self.rand_n_individual = np.vectorize(lambda x: x * self.random_individual(local_xmin, local_xmax), signature='()->(n)')
        self.generate_population = np.vectorize(lambda decimal: Individual.generate_from_decimal(
            decimal, self.x_min, self.x_max, self.genes, self.f), signature='(n)->()')

        x = self.rand_n_individual(np.ones(self.N))
        self.population = self.generate_population(x)


        self.sort_population()
        self.action = None
        self.enemy = None
        self.ally = None

    def random_individual(self, x_min, x_max):
        return np.random.randint(x_min, x_max)

    @property
    def best_function(self):
        return self.population[0].f

    def roulette_function(self, f_min, f_max):
        return (f_max - self.population[0].f) / (f_max - f_min)

    @property
    def avg_function(self):
        return sum([individual.f for individual in self.population]) / self.population.size

    def sort_population(self):
        self.population.sort()

    def reproduction(self, n_min, n_max, f_min, f_max):
        n = ceil((n_max - n_min) * (f_max - self.avg_function) / (f_max - f_min) + n_min)
        n = np.clip(n, n_min, n_max)
        # p2 = (1 - ti / t_max) * (self.avg_function - f_min) / (f_max - f_min)
        new_individuals = []

        for i in range(n):
            if len(self.population) == 2 and self.population[0] == self.population[1]:
                new_individuals.extend(Individual.crossover(self.population[0], self.population[1],
                                                            self.x_min, self.x_max, self.genes, self.f))
                continue
            k1 = random.randint(0, len(self.population) - 1)
            individual1 = self.population[k1]
            k2 = k1
            while k2 == k1:
                k2 = random.randint(0, len(self.population) - 1)
            individual2 = self.population[k2]
            new_individuals += Individual.crossover(individual1, individual2)
        self.population = np.append(self.population, np.array(new_individuals))
        self.sort_population()

    def extinction(self, m_min, m_max, f_min, f_max):
        m = int((m_max - m_min) * (self.avg_function - f_min) / (f_max - f_min) + m_min)
        m = np.clip(m, m_min, m_max)
        self.population = self.population[:-m]

    def extinction1(self, n):
        if self.population.size < n:
            return n - self.population.size
        self.population = self.population[:n]
        return 0

    def select_action(self, countries):
        self.action = random.randint(0, 3)
        if self.action == 1:
            ally_list = [country for country in countries if country.action is None and country != self]
            if ally_list:
                self.ally = ally_list.pop(random.randint(0, len(ally_list) - 1))
                self.ally.action = 1
                self.ally.ally = self
            else:
                self.action = random.choice([0, 3])
        if self.action == 2:
            enemy_list = [country for country in countries if country.action is None and country != self]
            if enemy_list:
                self.enemy = enemy_list.pop(random.randint(0, len(enemy_list) - 1))
                self.enemy.action = 2
                self.enemy.enemy = self
            else:
                self.action = random.choice([0, 3])

    def epedemic(self, elite, dead, max_mutation):
        if max_mutation < 1:
            max_mutation = 1
        n_elite = ceil(elite * len(self.population))
        n_dead = ceil(dead * len(self.population))
        self.population = self.population[:-n_dead]
        for i in range(n_elite, self.population.size):
            self.population[i] = self.population[i].mutation(max_mutation)
        self.sort_population()
        self.action = None

    # def sabotage(self, n_copy):
    #     for i in range(n_copy):
    #         self.enemy.population.append(copy.copy(self.population[0]))
    #     self.action = None
    #     self.enemy = None

    def motion(self):
        x_best = self.population[0].decimal_x
        for i in range(1, len(self.population)):
            self.population[i] = Individual.generate_from_decimal(
                self.population[i].decimal_x + np.int64(random.uniform(0, 2) * (x_best - self.population[i].decimal_x)), self.x_min, self.x_max, self.genes, self.f
            )
        self.sort_population()
        self.action = None

    @staticmethod
    def trade(country1, country2, k):
        if country1.population.size <= k or country2.population.size <= k:
            k = min(country1.population.size, country2.population.size) // 2
        indexes1 = np.random.choice(country1.population.size, k, replace=False)
        indexes2 = np.random.choice(country2.population.size, k, replace=False)
        country2.population = np.concatenate([country2.population, country1.population[indexes1]])
        country1.population = np.concatenate([country1.population, country2.population[indexes2]])
        country1.population = np.delete(country1.population, indexes1)
        country2.population = np.delete(country2.population, indexes2)
        country1.sort_population()
        country2.sort_population()
        country1.action = None
        country2.action = None
        country1.ally = None
        country2.ally = None

    @staticmethod
    def war(country1, country2, l):
        if country1.population.size <= l or country2.population.size <= l:
            l = min(country1.population.size, country2.population.size)
        indexes1 = np.random.choice(country1.population.size, l, replace=False)
        indexes2 = np.random.choice(country2.population.size, l, replace=False)
        war_list1 = country1.population[indexes1]
        war_list2 = country2.population[indexes2]
        country1.population = np.delete(country1.population, indexes1)
        country2.population = np.delete(country2.population, indexes2)
        wins1 = np.where(war_list1 > war_list2)
        wins2 = np.where(war_list2 > war_list2)
        if wins1[0].size > wins2[0].size:
            np.concatenate([country1.population, war_list1])
            np.concatenate([country1.population, war_list2])
        elif wins2[0].size > wins1[0].size:
            np.concatenate([country2.population, war_list1])
            np.concatenate([country2.population, war_list2])
        else:
            np.concatenate([country1.population, war_list1])
            np.concatenate([country2.population, war_list2])
        country1.sort_population()
        country2.sort_population()
        country1.action = None
        country2.action = None
        country1.enemy = None
        country2.enemy = None


class CountriesAlgorithm:

    def __init__(self, f, Xmin, Xmax, genes, M, N, n, m, k, l, ep, max_mutation, tmax, printing=False, memory_list=None):
        self.f = f
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.n = n
        self.genes = genes
        self.m = m
        self.k = k
        self.M = M
        self.N = N
        self.l = l
        self.ep = ep
        self.max_mutation = max_mutation
        self.tmax = tmax
        self.countries = []
        self.printing = printing
        self.memory_list = memory_list
        for i in range(M):
            self.countries.append(Country(N, self.Xmin, self.Xmax, self.genes, self.f))

    def start(self):
        ti = 0
        motion = 0
        trade = 0
        war = 0
        epedemic = 0
        if self.memory_list is not None:
            self.memory_list[0] = False
        while ti <= self.tmax:
            ti += 1
            for country in self.countries:
                if country.action is None:
                    country.select_action(self.countries)
            for country in self.countries:
                if country.action == 0:
                    motion += 1
                    country.motion(                    )
                elif country.action == 1:
                    trade += 1
                    Country.trade(
                        country1=country,
                        country2=country.ally,
                        k=self.k
                    )
                elif country.action == 2:
                    war += 1
                    Country.war(
                        country1=country,
                        country2=country.enemy,
                        l=self.l
                    )
                elif country.action == 3:
                    epedemic += 1
                    country.epedemic(
                        elite=self.ep[0],
                        dead=self.ep[1],
                        max_mutation=int((1 - ti / self.tmax) * self.max_mutation),
                    )
            indexes = np.where(vector_check_population(self.countries) == True)
            self.countries = [self.countries[i] for i in indexes[0]]

            self.countries = sorted(self.countries, key=attrgetter('avg_function'))
            if not self.countries:
                break
            f_min = self.countries[0].avg_function
            f_max = self.countries[-1].avg_function
            if f_min == f_max:
                self.countries = sorted(self.countries, key=attrgetter('best_function'))
                result = self.countries[0].population[0]
                break
            e_individuals = []
            for country in self.countries:
                if len(country.population) == 1:
                    e_individuals.append(country.population[0])
                    continue
                if country.population.size:
                    country.reproduction(
                        n_min=self.n[0],
                        n_max=self.n[1],
                        f_min=f_min,
                        f_max=f_max
                    )
                    country.extinction(
                        m_min=self.m[0],
                        m_max=self.m[1],
                        f_min=f_min,
                        f_max=f_max
                    )
            self.countries = [country for country in self.countries if len(country.population)]
            # self.countries = sorted(self.countries, key=attrgetter('best_function'))
            # f_min = self.countries[0].best_function
            # f_max = self.countries[-1].best_function
            # s = sum(country.roulette_function(f_min, f_max) for country in self.countries if len(country.population) > 1)
            # self.countries.reverse()
            # for country in self.countries:
            #     plus = 0
            #     if len(country.population) >= 1:
            #         res = country.extinction1(
            #             max(self.N // 2, ceil(country.roulette_function(f_min, f_max) / s * self.N * self.M)) + plus
            #         )
            #         if res:
            #             plus += res
            #         else:
            #             plus = 0
            #
            # indexes = np.where(vector_check_population(self.countries) == True)
            # self.countries = [self.countries[i] for i in indexes[0]]

            for individual in e_individuals:
                random_country = self.countries[random.randint(0, len(self.countries) - 1)]
                np.append(random_country.population, np.array([individual]))
                random_country.sort_population()
            self.countries = sorted(self.countries, key=attrgetter('best_function'))
            if not self.countries:
                break
            result = self.countries[0].population[0]

            if self.printing:
                print(f"{ti}) Лучшее решение: {result.real_x} - {result.f}, Стран осталось: {len(self.countries)}, Движение/Обмен/Войны/Эпидемии: {motion}/{trade}/{war}/{epedemic}")
                print(f"Общее количество особей: {sum([len(country.population) for country in self.countries])}")
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")
                for i, country in enumerate(self.countries):
                    print(f'{i + 1})', country.name, len(country.population), country.best_function, country.avg_function)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++")

            if self.memory_list is not None:
                self.memory_list[0] = ti
                for i in range(len(result.real_x)):
                    self.memory_list[i + 1] = float(result.real_x[i])
                self.memory_list[-1] = float(result.f)

        return (result.real_x, result.f, False, ti)



def f(x):
    return sum([(xi ** 4 - 16 * xi ** 2 + 5 *xi) / 2 for xi in x])
#
# k = 0
# for i in range(100):
CA = CountriesAlgorithm(
    f=f,
    Xmin=[-5.12 for i in range(20)],
    Xmax=[5.12 for i in range(20)],
    genes=[16 for i in range(20)],
    M=20,
    N=15,
    n=[1, 10],
    m=[3, 8],
    k=8,
    l=3,
    ep=[0.2, 0.4],
    max_mutation=16,
    tmax=300,
    printing=True,
)
r = CA.start()

#     print(i, r[2], r[0], r[1])
#     if r[2]:
#         k += 1
#
# print(k/100)
