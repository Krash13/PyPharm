import random
import numpy as np
from operator import attrgetter
from math import ceil, cos, sin


class Individual:

    def __init__(self, x, function):
        self.x = x
        self.f = function(self.x)
        self.ep_n = 0

    def update_function(self, function):
        self.f = function(self.x)

    @classmethod
    def crossing(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        x2 = []
        for i in range(len(individual1.x)):
            if i % 2:
                x1.append(individual1.x[i])
                x2.append(individual2.x[i])
            else:
                x1.append(individual2.x[i])
                x2.append(individual1.x[i])
            x1[i] += random.uniform(-p * x1[i], p * x1[i])
            x1[i] = x1[i] if x1[i] >= Xmin[i] else Xmin[i]
            x1[i] = x1[i] if x1[i] <= Xmax[i] else Xmax[i]
            x2[i] += random.uniform(-p * x2[i], p * x2[i])
            x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
            x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        return [cls(x1, function), cls(x2, function)]

    @classmethod
    def crossing2(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        x2 = []
        random_list = list(range(len(individual1.x)))
        random.shuffle(random_list)
        random_list = random_list[:len(individual1.x) // 2]
        for i in range(len(individual1.x)):
            if i in random_list:
                x1.append(individual1.x[i])
                x2.append(individual2.x[i])
            else:
                x1.append(individual2.x[i])
                x2.append(individual1.x[i])
            x1[i] += random.uniform(-p * x1[i], p * x1[i])
            x1[i] = x1[i] if x1[i] >= Xmin[i] else Xmin[i]
            x1[i] = x1[i] if x1[i] <= Xmax[i] else Xmax[i]
            x2[i] += random.uniform(-p * x2[i], p * x2[i])
            x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
            x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        return [cls(x1, function), cls(x2, function)]

    @classmethod
    def crossing3(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        x2 = []
        max_n = len(individual1.x) // 2
        n = random.randint(1, max_n)
        l = []
        for i in range(len(individual1.x) // n):
            l += [j + 2 * i * n for j in range(n) if j + 2 * i * n < len(individual1.x)]
        for i in range(len(individual1.x)):
            if i in l:
                x1.append(individual1.x[i])
                x2.append(individual2.x[i])
            else:
                x1.append(individual2.x[i])
                x2.append(individual1.x[i])
            x1[i] += random.uniform(-p * x1[i], p * x1[i])
            x1[i] = x1[i] if x1[i] >= Xmin[i] else Xmin[i]
            x1[i] = x1[i] if x1[i] <= Xmax[i] else Xmax[i]
            x2[i] += random.uniform(-p * x2[i], p * x2[i])
            x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
            x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        return [cls(x1, function), cls(x2, function)]

    @classmethod
    def crossing4(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        x2 = []
        for i in range(len(individual1.x)):
            x1.append((individual1.x[i] + individual2.x[i]) / 2)
            x2.append((individual1.x[i] + individual2.x[i]) / 2)
            x2[i] += random.uniform(-p * x2[i], p * x2[i])
            x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
            x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        return [cls(x1, function), cls(x2, function)]

    @classmethod
    def crossing5(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        # x2 = []
        for i in range(len(individual1.x)):
            x1.append((individual1.x[i] + individual2.x[i]) / 2)
            # x2.append((individual1.x[i] + individual2.x[i]) / 2)
            # x2[i] += random.uniform(-p * x2[i], p * x2[i])
            # x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
            # x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        return [cls(x1, function)] #, cls(x2, function)]

    @classmethod
    def crossing6(cls, individual1, individual2, p, function, Xmin, Xmax):
        x1 = []
        k = random.random()
        if k > 0.5:
            x2 = []
        for i in range(len(individual1.x)):
            x1.append((individual1.x[i] + individual2.x[i]) / 2)
            if k > 0.5:
                x2.append((individual1.x[i] + individual2.x[i]) / 2)
                x2[i] += random.uniform(-p * x2[i], p * x2[i])
                x2[i] = x2[i] if x2[i] >= Xmin[i] else Xmin[i]
                x2[i] = x2[i] if x2[i] <= Xmax[i] else Xmax[i]
        if k > 0.5:
            return [cls(x1, function), cls(x2, function)]
        return [cls(x1, function)]

    @classmethod
    def crossing7(cls, individual1, individual2, p, function, Xmin, Xmax):
        k = random.random()
        if k <= 1 / 3:
            return Individual.crossing4(individual1, individual2, p, function, Xmin, Xmax)
        elif k <= 2 / 3:
            return Individual.crossing8(individual1, individual2, p, function, Xmin, Xmax)
        else:
            return Individual.crossing6(individual1, individual2, p, function, Xmin, Xmax)

    @classmethod
    def crossing8(cls, individual1, individual2, p, function, Xmin, Xmax):
        new_x = []
        alpha = 0.5
        for i in range(len(individual1.x)):
            c_min = min(individual1.x[i], individual2.x[i])
            c_max = max(individual1.x[i], individual2.x[i])
            I = c_max - c_min
            new_x.append(random.uniform(c_min - I * alpha, c_max + I * alpha))
        new_x = np.clip(new_x, Xmin, Xmax)
        return [cls(new_x, function)]


    @classmethod
    def crossing9(cls, individual1, individual2, p, function, Xmin, Xmax):
        new_x = []
        alpha = 0.5
        for i in range(len(individual1.x)):
            c_min = min(individual1.x[i], individual2.x[i])
            c_max = max(individual1.x[i], individual2.x[i])
            I = c_max - c_min
            new_x.append(random.uniform(c_min - I * alpha, c_max + I * alpha))
        new_x = np.clip(new_x, Xmin, Xmax)
        if random.random() <= 0.5:
            for i in range(len(individual1.x)):
                new_x[i] += random.uniform(-p * new_x[i], p * new_x[i])
        new_x = np.clip(new_x, Xmin, Xmax)
        return [cls(new_x, function)]

    @classmethod
    def crossing10(cls, individual1, individual2, p, function, Xmin, Xmax, p2):
        k = random.random()
        if k <= 1 / 3:
            return Individual.crossing9(individual1, individual2, p, function, Xmin, Xmax, p2)
        else:
            return Individual.crossing8(individual1, individual2, p, function, Xmin, Xmax)

    @classmethod
    def crossing11(cls, individual1, individual2, p, function, Xmin, Xmax):
        new_x = []
        alpha = p
        for i in range(len(individual1.x)):
            c_min = min(individual1.x[i], individual2.x[i])
            c_max = max(individual1.x[i], individual2.x[i])
            I = c_max - c_min
            new_x.append(random.uniform(c_min - I * alpha, c_max + I * alpha))
        new_x = np.clip(new_x, Xmin, Xmax)
        return [cls(new_x, function)]

    def mutation(self, Xmin, Xmax, function, pmax):
        self.ep_n += 1
        for i in range(len(self.x)):
            self.x[i] += pmax * random.uniform(-self.x[i], self.x[i]) / self.ep_n
            self.x[i] = self.x[i] if self.x[i] >= Xmin[i] else Xmin[i]
            self.x[i] = self.x[i] if self.x[i] <= Xmax[i] else Xmax[i]
        self.update_function(function)


class Country:

    def __init__(self, Xmin, Xmax, N, function):
        x_min = []
        x_max = []
        for i in range(len(Xmin)):
            x_min.append(random.uniform(Xmin[i], Xmax[i]))
            x_max.append(random.uniform(x_min[i], Xmax[i]))
        self.population = []
        for i in range(N):
            x = []
            for j in range(len(Xmin)):
                x.append(random.uniform(x_min[j], x_max[j]))
            self.population.append(Individual(x, function))
        self.sort_population()
        self.action = None
        self.enemy = None
        self.ally = None

    @property
    def best_function(self):
        return self.population[0].f

    @property
    def avg_function(self):
        return sum([individual.f for individual in self.population]) / len(self.population)

    def update_population(self, function):
        for individual in self.population:
            individual.update_function(function)

    def sort_population(self):
        self.population.sort(key=attrgetter('f'))

    def reproduction(self, n_min, n_max, p_min, p_max, f_min, f_max, ti, t_max, function, Xmin, Xmax):
        n = ceil((n_max - n_min) * (f_max - self.avg_function) / (f_max - f_min) + n_min)
        n = np.clip(n, n_min, n_max)
        p = (p_max - p_min) * (1 - ti / t_max) * (self.avg_function - f_min) / (f_max - f_min) + p_min
        p = np.clip(p, p_min, p_max)
        # p2 = (1 - ti / t_max) * (self.avg_function - f_min) / (f_max - f_min)
        new_individuals = []

        for i in range(n):
            if len(self.population) == 2 and self.population[0] == self.population[1]:
                new_individuals.extend(Individual.crossing11(self.population[0], self.population[1], p, function, Xmin, Xmax))
                continue
            k1 = random.randint(0, len(self.population) - 1)
            individual1 = self.population[k1]
            k2 = k1
            while k2 == k1:
                k2 = random.randint(0, len(self.population) - 1)
            individual2 = self.population[k2]
            new_individuals.extend(Individual.crossing11(individual1, individual2, p, function, Xmin, Xmax))
        self.population.extend(new_individuals)
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
            for j in range(len(x_best)):
                self.population[i].x[j] += random.uniform(0, 2) * (x_best[j] - self.population[i].x[j])
                self.population[i].x[j] = self.population[i].x[j] if self.population[i].x[j] >= Xmin[j] else Xmin[j]
                self.population[i].x[j] = self.population[i].x[j] if self.population[i].x[j] <= Xmax[j] else Xmax[j]
        self.update_population(function)
        self.sort_population()
        self.action = None

    @staticmethod
    def trade(country1, country2, k):
        trade_list1 = []
        trade_list2 = []
        if len(country1.population) <= k or len(country2.population) <= k:
            k = min(len(country1.population), len(country2.population)) // 2
        for i in range(k):
            trade_list1.append(country1.population.pop(random.randint(0, len(country1.population) - 1)))
            trade_list2.append(country2.population.pop(random.randint(0, len(country2.population) - 1)))
        country1.population.extend(trade_list2)
        country2.population.extend(trade_list1)
        country1.sort_population()
        country2.sort_population()
        country1.action = None
        country2.action = None
        country1.ally = None
        country2.ally = None

    @staticmethod
    def war(country1, country2, l):
        war_list1 = []
        war_list2 = []
        if len(country1.population) <= l or len(country2.population) <= l:
            l = min(len(country1.population), len(country2.population))
        for i in range(l):
            war_list1.append(country1.population.pop(random.randint(0, len(country1.population) - 1)))
            war_list2.append(country2.population.pop(random.randint(0, len(country2.population) - 1)))
        wins1 = 0
        wins2 = 0
        for i in range(l-1, -1, -1):
            if war_list1[i].f < war_list2[i].f:
                war_list2.pop(i)
                wins1 += 1
            elif war_list1[i].f > war_list2[i].f:
                war_list1.pop(i)
                wins2 += 1
        if wins1 > wins2:
            country1.population.extend(war_list1)
            country1.population.extend(war_list2)
        elif wins2 > wins1:
            country2.population.extend(war_list1)
            country2.population.extend(war_list2)
        else:
            country1.population.extend(war_list1)
            country2.population.extend(war_list2)
        country1.sort_population()
        country2.sort_population()
        country1.action = None
        country2.action = None
        country1.enemy = None
        country2.enemy = None


class CountriesAlgorithm:

    def __init__(self, f, Xmin, Xmax, M, N, n, p, m, k, l, ep, tmax, printing=False, memory_list=None):
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
        self.countries = []
        self.printing = printing
        self.memory_list = memory_list
        for i in range(M):
            self.countries.append(Country(self.Xmin, self.Xmax, N, self.f))

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
            self.countries = [country for country in self.countries if country.population]
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
                if country.population:
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
            self.countries = [country for country in self.countries if country.population]
            for individual in e_individuals:
                random_country = self.countries[random.randint(0, len(self.countries) - 1)]
                random_country.population.append(individual)
                random_country.sort_population()
            self.countries = sorted(self.countries, key=attrgetter('best_function'))
            if not self.countries:
                break
            result = self.countries[0].population[0]

            if self.printing:
                print(f"{ti}) Лучшее решение: {result.x} - {result.f}, Стран осталось: {len(self.countries)}, Движение/Обмен/Войны/Эпидемии: {motion}/{trade}/{war}/{epedemic}")
                print(f"Общее количество особей: {sum([len(country.population) for country in self.countries])}")

            if self.memory_list is not None:
                self.memory_list[0] = ti
                for i in range(len(result.x)):
                    self.memory_list[i + 1] = float(result.x[i])
                self.memory_list[-1] = float(result.f)
        return (result.x, result.f, False, ti)
