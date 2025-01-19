import numpy as np


class GoldDigger:

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

    def new_sigma_digger(self, x_min, x_max, sigma, function):
        new_x = np.array([self.x[i] + np.random.normal(0, (x_max[i] - x_min[i]) * sigma) for i in range(self.x.size)])
        new_x = np.clip(new_x, x_min, x_max)
        return GoldDigger(new_x, function)



class GoldDiggersAlgorithm:

    def __init__(self, f, n, k, l, u, a, b, sigma_b, sigma_e, Xmin, Xmax, m, t_max):
        self.f = f
        self.n = n
        self.k = k
        self.l = l
        self.u = u
        self.a = a
        self.b = b
        self.sigma_b = sigma_b
        self.sigma_e = sigma_e
        self.Xmin = np.array(Xmin)
        self.Xmax = np.array(Xmax)
        self.m = m
        self.t_max = t_max
        self.rand_mat = np.vectorize(lambda x: self.random_matrix(x, Xmin, Xmax), signature='()->(n)')
        self.digger_init = np.vectorize(lambda x: GoldDigger(x, self.f), signature='(n)->()')

    @staticmethod
    def random_matrix(x, x_min, x_max):
        return x * np.random.uniform(x_min, x_max)

    def start(self):
        x = self.rand_mat(np.ones(self.n))
        population = self.digger_init(x)

        t = 0
        intervals = []
        while t <= self.t_max:

            x = self.rand_mat(np.ones(self.n))
            in_interval = np.array([self.interval_in(xi, intervals) for xi in x])
            if np.any(in_interval == True):
                ind = np.where(in_interval == False)
                new_x = x[ind]
                if new_x.size:
                    population = np.append(population, self.digger_init(new_x))
                ind = np.where(in_interval == True)
                r = np.random.rand(ind[0].size)
                rand_ind = np.where(r <= self.a)
                ind = (ind[0][rand_ind])
                x = x[ind]
                if x.size:
                    population = np.append(population, self.digger_init(x))
            else:
                population = np.append(population, self.digger_init(x))

            sigma = (((self.t_max - t) / self.t_max) ** self.m) * (self.sigma_b - self.sigma_e) + self.sigma_e

            new_diggers = []
            for digger in population:
                for _ in range(self.k):
                    new_diggers.append(digger.new_sigma_digger(self.Xmin, self.Xmax, sigma, self.f))
            population = np.append(population, new_diggers)
            population = np.sort(population)

            worst_diggers = population[self.n:]
            r = np.random.rand(worst_diggers.size)
            ind = np.where(r <= self.b)
            worst_diggers = worst_diggers[ind]
            for digger in worst_diggers:
                intervals.append(
                    (digger.x - (self.Xmax - self.Xmin) * sigma, digger.x + (self.Xmax - self.Xmin)  * sigma)
                )
            if len(intervals) > self.u:
                intervals = intervals[len(intervals) - self.u - 1:len(intervals) - 1]
            population = population[:self.n]
            print("Поколение {}: {} {}".format(t, population[0].x, population[0].f))
            t += 1
        return population[0].x

    @staticmethod
    def interval_in(x, intervals):
        for interval in intervals:
            res = [x[i] >= interval[0][i] and x[i] <= interval[1][i] for i in range(len(x))]
            if any(res):
                return True
        return False


def f(x):
    return sum([(xi ** 4 - 16 * xi ** 2 + 5 *xi) / 2 for xi in x])

# gda = GoldDiggersAlgorithm(f, 15, 10, 10, 50, 0.5, 0.5, 0.5, 0.000001, [-5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12, -5.12], [5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 5.12], 2, 300)
# gda.start()