import graycode
import numpy as np
import copy

class Individual:

    x_min = None
    x_max = None
    genes = None
    function = None
    gray_code = None
    f = None

    def __init__(self, gray_code, x_min, x_max, genes, function, mutation_chance=None, max_mutation=1):

        self.x_min = x_min
        self.x_max = x_max
        self.genes = genes
        self.function = function
        self.steps = (self.x_max - self.x_min) / (2 ** (self.genes) - 1)
        self.gray_code = gray_code
        if mutation_chance is not None and np.random.random() < mutation_chance:
            self.mutation(max_mutation)
        self.f = self.function(self.real_x)

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

    @classmethod
    def crossover(cls, individual1, individual2, mutation_chance, max_mutation=1):
        k = np.random.randint(1, len(individual1.gray_code) - 1, 2)
        while k[0] == k[1]:
            k = np.random.randint(1, len(individual1.gray_code) - 1, 2)
        k = np.sort(k)
        new_gray1 = individual1.gray_code[:k[0]] + individual2.gray_code[k[0]:k[1]] + individual1.gray_code[k[1]:]
        new_gray2 = individual2.gray_code[:k[0]] + individual1.gray_code[k[0]:k[1]] + individual2.gray_code[k[1]:]
        return [
                cls(new_gray1, individual1.x_min, individual1.x_max, individual1.genes,
                    individual1.function, mutation_chance, max_mutation),
                cls(new_gray2, individual1.x_min, individual1.x_max, individual1.genes,
                    individual1.function, mutation_chance, max_mutation),
        ]

    def mutation(self, max_mutation=1):
        if max_mutation > 0:
            n = 1 + np.random.randint(1, max_mutation)
        else:
            n = 1
        k = np.random.choice(len(self.gray_code), n, replace=False)
        for ki in k:
            b = bool(int(self.gray_code[ki]))
            b = not b
            b = str(int(b))
            self.gray_code = self.gray_code[:ki] + b + self.gray_code[ki+1:]

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f


class GeneticAlgorithm:

    def __init__(self, f, n, child_percent, mutation_chance, max_mutation, x_min, x_max, genes, t_max, printing=False):
        self.f = f
        self.n = n
        self.child_percent = child_percent
        self.mutation_chance = mutation_chance
        self.max_mutation = max_mutation
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.genes = np.array(genes)
        self.t_max = np.array(t_max)
        self.printing = printing

        self.rand_n_individual = np.vectorize(lambda x: x * self.random_individual(), signature='()->(n)')
        self.generate_population = np.vectorize(lambda decimal: Individual.generate_from_decimal(
            decimal, self.x_min, self.x_max, self.genes, self.f), signature='(n)->()')

    def random_individual(self):
        return np.random.randint(np.zeros(self.genes.size), 2 ** self.genes)

    def start(self):
        x = self.rand_n_individual(np.ones(self.n))
        population = self.generate_population(x)
        child_number = int(self.child_percent * self.n)

        t = 0
        while t <= self.t_max:
            new_population = []
            for i in range(child_number):
                k = np.random.randint(0, self.n, 2)
                while k[0] == k[1]:
                    k = np.random.randint(0, self.n, 2)
                new_population += Individual.crossover(population[k[0]], population[k[1]],
                                                       self.mutation_chance, self.max_mutation)
            population = np.append(population, np.array(new_population))
            population.sort()
            population = population[:self.n]
            if self.printing:
                print("Поколение {}: {} {}".format(t, population[0].real_x, population[0].f))
            t += 1
        return population[0].real_x
