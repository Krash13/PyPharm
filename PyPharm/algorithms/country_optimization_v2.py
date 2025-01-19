import random
import numpy as np
from math import ceil


def random_matrix(x, x_min, x_max):
    return x * np.random.uniform(x_min, x_max)

def check_population(country):
    return bool(country.population.size)

def get_avg(country):
    return country.avg_function

def get_best(country):
    return country.best_function

vector_check_population = np.vectorize(check_population, signature='()->()')
vector_min = np.vectorize(min, signature='(),()->()')
vector_max = np.vectorize(max, signature='(),()->()')
vector_avg = np.vectorize(get_avg, signature='()->()')
vector_best = np.vectorize(get_best, signature='()->()')

class Individual:

    def __init__(self, x, function):
        self.x = x
        self.f = function(self.x)
        self.ep_n = 0

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    def __add__(self, other):
        return self.f + other.f

    def __radd__(self, other):
        return self.f + other

    def __truediv__(self, other):
        return self.f / other

    def update_function(self, function):
        self.f = function(self.x)

    @classmethod
    def crossing(cls, individual1, individual2, p, function, Xmin, Xmax):
        alpha = p
        c_min = vector_min(individual1.x, individual2.x)
        c_max = vector_max(individual1.x, individual2.x)
        I = c_max - c_min
        new_x = np.random.uniform(c_min - I * alpha, c_max + I * alpha)
        new_x = np.clip(new_x, Xmin, Xmax)
        return [cls(new_x, function)]

    def mutation(self, Xmin, Xmax, function, pmax):
        self.ep_n += 1
        self.x = self.x + pmax * np.random.uniform(-self.x, self.x) / self.ep_n
        np.clip(self.x, Xmin, Xmax)
        self.update_function(function)


class Country:

    def __init__(self, Xmin, Xmax, N, function):
        x_min = np.random.uniform(Xmin, Xmax)
        x_max = np.random.uniform(x_min, Xmax)
        self.population = []
        ind_init = np.vectorize(lambda x: Individual(x, function), signature='(n)->()')
        rand_mat = np.vectorize(lambda x: random_matrix(x, x_min, x_max), signature='()->(n)')
        v = rand_mat(np.ones(N))
        self.population = ind_init(v)
        self.sort_population()
        self.action = None
        self.enemy = None
        self.ally = None

    @property
    def best_function(self):
        return self.population[0].f

    @property
    def avg_function(self):
        return np.average(self.population)

    def update_population(self, function):
        for individual in self.population:
            individual.update_function(function)

    def sort_population(self):
        self.population = np.sort(self.population)

    def reproduction(self, n_min, n_max, p_min, p_max, f_min, f_max, ti, t_max, function, Xmin, Xmax):
        n = ceil((n_max - n_min) * (f_max - self.avg_function) / (f_max - f_min) + n_min)
        n = np.clip(n, n_min, n_max)
        p = (p_max - p_min) * (1 - ti / t_max) * (self.avg_function - f_min) / (f_max - f_min) + p_min
        p = np.clip(p, p_min, p_max)
        new_individuals = np.array([])
        for i in range(n):
            parents = np.random.choice(self.population, 2, replace=False)
            new_individuals = np.concatenate([new_individuals, Individual.crossing(parents[0], parents[1], p, function, Xmin, Xmax)])
        self.population = np.concatenate([self.population, new_individuals])
        self.sort_population()

    def extinction(self, m_min, m_max, f_min, f_max):
        m = int((m_max - m_min) * (self.avg_function - f_min) / (f_max - f_min) + m_min)
        m = m if m <= m_max else m_max
        m = m if m >= m_min else m_min
        self.population = self.population[:-m]

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

    def epedemic(self, elite, dead, function, Xmin, Xmax, p_max):
        n_elite = ceil(elite * len(self.population))
        n_dead = ceil(dead * len(self.population))
        self.population = self.population[:-n_dead]
        for individual in self.population[n_elite:]:
            individual.mutation(Xmin, Xmax, function, p_max)
        self.sort_population()
        self.action = None

    # def sabotage(self, n_copy):
    #     for i in range(n_copy):
    #         self.enemy.population.append(copy.copy(self.population[0]))
    #     self.action = None
    #     self.enemy = None

    def motion(self, function, Xmin, Xmax):
        x_best = self.population[0].x
        for i in range(1, len(self.population)):
            self.population[i].x = self.population[i].x + np.random.uniform(0, 2, self.population[i].x.size) * (x_best - self.population[i].x)
            np.clip(self.population[i].x, Xmin, Xmax)
        self.update_population(function)
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


class CountriesAlgorithm_v2:

    def __init__(self, f, Xmin, Xmax, M, N, n, p, m, k, l, ep, tmax, printing=False):
        self.f = f
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.n = n
        self.p = p
        self.m = m
        self.k = k
        self.l = l
        self.ep = ep
        self.tmax = tmax
        self.printing = printing
        country_init = np.vectorize(lambda x: Country(self.Xmin, self.Xmax, N, self.f))
        self.countries = country_init(np.ones(M))

    def sort(self, method):
        if method == 'avg_function':
            avg_arr = vector_avg(self.countries)
            indexes = np.argsort(avg_arr)
            self.countries = self.countries[indexes]
        else:
            avg_arr = vector_best(self.countries)
            indexes = np.argsort(avg_arr)
            self.countries = self.countries[indexes]

    def start(self):
        ti = 0
        motion = 0
        trade = 0
        war = 0
        epedemic = 0
        while ti <= self.tmax:
            ti += 1
            for country in self.countries:
                if country.action is None:
                    country.select_action(self.countries)
            for country in self.countries:
                if country.action == 0:
                    motion += 1
                    country.motion(
                        function=self.f,
                        Xmin=self.Xmin,
                        Xmax=self.Xmax
                    )
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
                        Xmin=self.Xmin,
                        Xmax=self.Xmax,
                        function=self.f,
                        p_max=self.p[1],
                    )
            indexes = np.where(vector_check_population(self.countries) == True)
            self.countries = self.countries[indexes]
            self.sort('avg_function')
            if not self.countries.size:
                break
            f_min = self.countries[0].avg_function
            f_max = self.countries[-1].avg_function
            if f_min == f_max:
                self.sort('best_function')
                result = self.countries[0].population[0]
                break
            e_individuals = []
            for country in self.countries:
                if country.population.size == 1:
                    e_individuals.append(country.population[0])
                    continue
                if country.population.size:
                    country.reproduction(
                        n_min=self.n[0],
                        n_max=self.n[1],
                        p_min=self.p[0],
                        p_max=self.p[1],
                        f_min=f_min,
                        f_max=f_max,
                        ti=ti,
                        t_max=self.tmax,
                        function=self.f,
                        Xmin=self.Xmin,
                        Xmax=self.Xmax
                    )
                    country.extinction(
                        m_min=self.m[0],
                        m_max=self.m[1],
                        f_min=f_min,
                        f_max=f_max
                    )
            indexes = np.where(vector_check_population(self.countries) == True)
            self.countries = self.countries[indexes]
            for individual in e_individuals:
                random_country = self.countries[random.randint(0, len(self.countries) - 1)]
                random_country.population = np.append(random_country.population, individual)
                random_country.sort_population()
            self.sort('best_function')
            if not self.countries.size:
                break
            result = self.countries[0].population[0]

            if self.printing:
                print(f"{ti}) Лучшее решение: {result.x} - {result.f}, Стран осталось: {len(self.countries)}, Движение/Обмен/Войны/Эпидемии: {motion}/{trade}/{war}/{epedemic}")
                print(f"Общее количество особей: {sum([len(country.population) for country in self.countries])}")
        return (result.x, result.f, False, ti)

