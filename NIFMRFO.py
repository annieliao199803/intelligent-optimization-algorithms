# --------------------------------------------------------%
#       Email: t111c52030@gmail.com
#       Github: https://github.com/annieliao199803
#       This code, written solely in Python, replicates the paper without being the creator of this method.
# --------------------------------------------------------%
import numpy as np
import csv  # Import CSV module
import math
from scipy.special import gamma

def ackley_function(x):
    """
    Compute the Ackley function value for a given x.
    Global Minimum = 0
    """
    x = np.array(x)
    x[0] = np.round(x[0])

    d = len(x)  # Dimensionality of the solution
    # First term of the Ackley function
    term1 = -0.2 * np.sqrt(np.sum(x**2) / d)
    # Second term of the Ackley function
    term2 = np.sum(np.cos(2 * np.pi * x)) / d
    # The Ackley function
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.e


def michalewicz_function(x, m=10):
    """
    Compute the Michalewicz function value for a given x.
    Parameters:
    x (numpy.ndarray): A numpy array of input values.
    m (int): The parameter that defines the steepness of the valleys and ridges; a larger m leads to a more difficult search.
    
    Global Minimum : -1.8013 (when d=2), -4.687658(when d=5), -9.66015(when d=10)
    """
    n = len(x)
    x[0] = np.round(x[0])

    sum_terms = 0
    for i in range(n):
        sum_terms += np.sin(x[i]) * (np.sin((i+1) * x[i]**2 / np.pi))**(2 * m)
    return -sum_terms

def mixed_integer_optimization_function(x):
    """
    Compute the value of a simplified four-dimensional mixed-integer optimization problem.
    
    Parameters:
    x (numpy.ndarray): A numpy array of input values, where x[:2] are continuous variables, and x[2:] are integer variables.
    
    Returns:
    float: The function value.
    """
    # 確保整數變量保持整數值
    x[0] = np.round(x[0])
    
    # 計算目標函數值
    f_value = (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2 + (x[3] - 4)**2
    
    return f_value

def griewank_function(x):
    x = np.array(x)
    x[0] = np.round(x[0])

    d = len(x)  # Dimensionality of the solution
    term1 = 1 + np.sum(x**2 / 4000)
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return term1 - term2

def dixon_price_function(x):
    """
    Calculate the Dixon-Price function value for a given input vector x.

    Parameters:
    x (array-like): Input vector for which to compute the function value.

    Returns:
    float: The Dixon-Price function value.
    """
    x = np.array(x)
    x[0] = np.round(x[0])

    d = len(x) #dimension
    term1 = (x[0] - 1)**2
    term2 = sum(i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, d))
    return term1 + term2


class Individual:
    def __init__(self, solution, fitness=None):
        self.solution = solution
        self.fitness = fitness

class OptimizationProblem:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.dimension = len(lb)


class NIFMRFO:
    """
    References
    [1] Liu, J., Chen, Y., Liu, X., Zuo, F., & Zhou, H. (2024).
    An efficient manta ray foraging optimization algorithm 
    with individual information interaction and fractional derivative mutation 
    for solving complex function extremum and engineering design problems. 
    Applied Soft Computing, 150, 111042.
    """
    def __init__(self, problem, max_iteration: int = 10000, pop_size: int = 100):
        """
        Args:
            problem : Including the upper and lower bounds of the problem and its dimensions
            max_iteration (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        self.max_iteration = max_iteration
        self.pop_size = pop_size
        self.generator = np.random.default_rng(2024)
        self.problem = problem
        self.pop = self.initialize_population()
        self.g_best = min(self.pop, key=lambda x: x.fitness)
        self.history = [self.pop]
        self.epsilon = 1e-8  # small value to avoid division by zero

    def initialize_population(self):
        """
        Initialize the position and fitness of each particle.
        """
        population = []
        for _ in range(self.pop_size):
            solution = np.array([np.random.uniform(low, high) for low, high in zip(self.problem.lb, self.problem.ub)])
            fitness = self.calculate_fitness(solution)
            individual = Individual(solution,fitness)
            population.append(individual)
        return population
    

    def cyclone_foraging(self,iteration,idx):
        r2 = self.generator.uniform()
        beta = 2 * np.exp(r2 * (self.max_iteration - iteration + 1) / self.max_iteration) * np.sin(2 * np.pi * r2)
        w = np.cos((np.pi * iteration) / (2 * self.max_iteration) - 0.2) + 0.25
        coef = -0.8 * w * np.cos(2 * np.pi + 1.2 * np.cos(2 * np.pi * w)) + 1

        if coef > self.generator.random():
            if idx == 0: #i=1
                x_next = self.g_best.solution + self.generator.uniform() * (self.g_best.solution - self.pop[idx].solution) + \
                        beta * (self.g_best.solution - self.pop[idx].solution)
            else: #i=2,...,N
                x_next = self.g_best.solution + self.generator.uniform() * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                        beta * (self.g_best.solution - self.pop[idx].solution)
        else:
            x_rand = self.generator.uniform(self.problem.lb, self.problem.ub) #Xrand = LB + rand(1,Dim) x (UB - LB)
            if idx == 0: #i=1
                x_next = x_rand + self.generator.uniform() * (x_rand - self.pop[idx].solution) + \
                        beta * (x_rand - self.pop[idx].solution)
            else: #i=2,...,N
                x_next = x_rand + self.generator.uniform() * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                        beta * (x_rand - self.pop[idx].solution)
                
        return x_next
    
    def chain_foraging(self,idx):
        r1 = self.generator.uniform()
        alpha = 2 * r1 * np.sqrt(np.abs(np.log(r1))) # equation 4
        if idx == 0: # i ==1
            x_next = self.pop[idx].solution + r1 * (self.g_best.solution - self.pop[idx].solution) + \
                    alpha * (self.g_best.solution - self.pop[idx].solution)# equation 3
        else: #i == 2,...,N
            x_next = self.pop[idx].solution + r1 * (self.pop[idx - 1].solution - self.pop[idx].solution) + \
                    alpha * (self.g_best.solution - self.pop[idx].solution)# equation 3
        return x_next
    
    def random_individuals_information_interaction(self, idx): 
        k = 0.2
        r5 = self.generator.uniform()
        m, n = self.generator.choice(self.pop_size, size=2, replace=False) #(m, n ∈ (1, N), m != n), and idx start at 0.
        x_next = self.pop[idx].solution + (k * (1 - r5) + r5) * (self.pop[m].solution - self.pop[n].solution)# equation 12
        return x_next
    
    def calculate_fitness(self, solution):
        return ackley_function(solution)
        # return griewank_function(solution)
        # return michalewicz_function(solution)
        # return mixed_integer_optimization_function(solution)
        # return dixon_price_function(solution)
    
    def update_global_best(self, individual : Individual = None):
        if individual.fitness < self.g_best.fitness:
            self.g_best = individual

    def sort_fitness(self, individuals, reverse_option):
        '''
        Sort by individual fitness value.
        '''
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=reverse_option)
        return sorted_individuals

    def calculate_x_minus(self, x_ij, h1, iteration, j):
        '''
        equation 23
        '''
        alpha = 0.4
        cos_factor = np.cos((np.pi * iteration) / (2 * self.max_iteration))
        sum_term = 0
        for k in range(5):# k range from 0 to 4
            binom_coeff = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
            sum_term += ((-1)**k) * binom_coeff * (self.g_best.solution[j] - x_ij - k * h1)
        x_minus = x_ij + cos_factor * (1 / (h1 + self.epsilon)**alpha) * sum_term
        
        return x_minus[0]
    
    def calculate_x_plus(self, x_ij, h2, iteration, j):
        '''
        equation 24
        '''
        alpha = 0.4
        cos_factor = np.cos((np.pi * iteration) / (2 * self.max_iteration))
        
        sum_term = 0
        for k in range(5):# k range from 0 to 4
            binom_coeff = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
            sum_term += ((-1)**k) * binom_coeff * (self.g_best.solution[j] - x_ij + k * h2)
        x_plus = x_ij + cos_factor * (1 / (h2 + self.epsilon)**alpha) * sum_term

        return x_plus[0]

    def run(self):
        
        for iteration in range( 1, self.max_iteration + 1 ):
            t1_individual = []
            for idx in range(self.pop_size):
                if self.generator.random() < 0.5:
                    x_next = self.cyclone_foraging(iteration, idx)
                else:
                    x_next = self.chain_foraging(idx)
                
                x_next = np.clip(x_next, self.problem.lb, self.problem.ub) # Ensure x_next is within bounds
                next_fitness = self.calculate_fitness(x_next)
                next_individual = (Individual(x_next, next_fitness)) #add x(t+i)'s individuals to history
                self.update_global_best(next_individual)
                t1_individual.append(next_individual)

            
            # self.history.append(t1_individual)

            for idx in range(self.pop_size):
                x_next = self.random_individuals_information_interaction(idx)
                x_next = np.clip(x_next, self.problem.lb, self.problem.ub) # Ensure x_next is within bounds
                next_fitness = self.calculate_fitness(x_next)
                next_individual = (Individual(x_next, next_fitness)) #add x(t+i)'s individuals to history
                self.update_global_best(next_individual)
                t1_individual[idx] = next_individual

            #對fitness進行排序
            t1_individual_sorted = self.sort_fitness(t1_individual,True)
            N = len(t1_individual_sorted)
            Y = round(N/3 - (N/3 - N/6) * iteration/self.max_iteration)
            rho = 4
            
            # Initialize x_minus and x_plus arrays with the correct shape
            x_minus = np.zeros((Y, self.problem.dimension))
            x_plus = np.zeros((Y, self.problem.dimension))

            for i in range(Y):
                r6 = self.generator.random()
                # Calculate the value of m using equation 20
                m = math.ceil((self.problem.dimension / 6) * (1 - iteration / self.max_iteration) * ((1 + r6) / 2))

                for j in range(m):
                    h1 = (self.problem.ub - t1_individual_sorted[i].solution[j]) / (rho + 1) * (1 - iteration / self.max_iteration)
                    h2 = (t1_individual_sorted[i].solution[j] - self.problem.lb) / (rho + 1) * (1 - iteration / self.max_iteration)
                    x_minus[i, j] = self.calculate_x_minus(t1_individual_sorted[i].solution[j], h1, iteration, j)
                    x_plus[i, j] = self.calculate_x_plus(t1_individual_sorted[i].solution[j], h2, iteration, j)
                
                    
            # Boundary treatment of transgressing individuals.
            x_minus = np.clip(x_minus, self.problem.lb, self.problem.ub) # Ensure x_minus is within bounds
            x_plus = np.clip(x_plus, self.problem.lb, self.problem.ub) # Ensure x_plus is within bounds
            

            # Convert x_minus and x_plus to Individual objects
            x_minus_individuals = [Individual(np.array(x), self.calculate_fitness(np.array(x))) for x in x_minus]
            x_plus_individuals = [Individual(np.array(x), self.calculate_fitness(np.array(x))) for x in x_plus]

            # Combine x_minus, x_plus, and t1_individual_sorted into one list
            combined_individuals = x_minus_individuals + x_plus_individuals + t1_individual_sorted

            # Sort the combined list by fitness in ascending order
            sorted_combined_individuals = self.sort_fitness(combined_individuals, False)

            # Loop through the sorted_combined_individuals and update the global best for each individual
            for individual in sorted_combined_individuals:
                self.update_global_best(individual)

            # Select the top N individuals from sorted_combined_individuals
            top_N_individuals = sorted_combined_individuals[:N]

            self.pop = top_N_individuals

            self.history.append(top_N_individuals)
            # print(f"Generation {iteration + 1}/{self.max_iteration}, Best Fitness: {self.g_best.fitness}")
    

if __name__ == "__main__":
    # Define the problem's dimension and bounds
    dimension = 4  # Let's use a 2D problem for simplicity

    # #Ackley function
    lb = -32.768 * np.ones(dimension)  # Lower bounds
    ub = 32.768 * np.ones(dimension)   # Upper bounds

    #Griewank function
    # lb = -600 * np.ones(dimension)  # Lower bounds
    # ub = 600 * np.ones(dimension)   # Upper bounds

    # #Michakewicz function
    # lb = 0 * np.ones(dimension)  # Lower bounds
    # ub = np.pi * np.ones(dimension)   # Upper bounds

    # #Mixed solution  function
    # lb = 0 * np.ones(dimension)  # Lower bounds
    # ub = 5 * np.ones(dimension)   # Upper bounds

    #Dixon Price  function
    # lb = -10 * np.ones(dimension)  # Lower bounds
    # ub = 10 * np.ones(dimension)   # Upper bounds

    # Create the optimization problem instance
    problem = OptimizationProblem(lb, ub)

    # Initialize the MRFO optimizer
    # Using smaller values for max_iteration and pop_size for quicker demonstration
    optimizer = NIFMRFO(problem, max_iteration=10000, pop_size=50)

    # Run the optimization process
    optimizer.run()

    # Optionally, to print the best solution found:
    print(f"Best solution found: {optimizer.g_best.solution}")
    print(f"Best fitness achieved: {optimizer.g_best.fitness}")

    # Writing history to CSV after the run completes
    with open('NIFMRFO_optimizer_Ackley4D_trial5.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Individual Index', 'Solution', 'Fitness'])
        for gen_index, generation in enumerate(optimizer.history):
            for ind_index, individual in enumerate(generation):
                writer.writerow([gen_index, ind_index, individual.solution.tolist(), individual.fitness])
        # Writing the best solution and fitness
        writer.writerow([])
        writer.writerow(['Best Solution', optimizer.g_best.solution.tolist()])
        writer.writerow(['Best Fitness', optimizer.g_best.fitness])
    

