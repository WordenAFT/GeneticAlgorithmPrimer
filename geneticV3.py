import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import os
import imageio.v2

#Suppress nuisance warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Weapon:
    def __init__(self, impact_angle=None, velocity=None, mutated=False):
        self.impact_angle = impact_angle if impact_angle is not None else np.random.uniform(35, 90)
        self.velocity = velocity if velocity is not None else np.random.uniform(600, 1500)
        self.mutated = mutated  # Flag to indicate mutation
        # Note: We'll calculate fitness later with optimal_point passed as a parameter
    def calc_fitness(self, optimal_point, fitness="default"):
        
        if fitness == "default":
            dist = np.sqrt((self.impact_angle - optimal_point[0])**2 + (self.velocity - optimal_point[1])**2)
            self.fitness = 1 / (1 + dist)
        elif fitness == "landscape":
            self.fitness = fitness_function(self.impact_angle, self.velocity)

def crossover(parent1, parent2):
    
    jitter1 = abs(parent1.impact_angle - parent2.impact_angle)*.5
    jitter2 = abs(parent1.velocity - parent2.velocity)*.5

 # Generate random values between the two parents' traits for the offspring
    child1_impact_angle = np.random.uniform(min(parent1.impact_angle, parent2.impact_angle)-jitter1, max(parent1.impact_angle, parent2.impact_angle)+jitter1)
    child1_velocity = np.random.uniform(min(parent1.velocity, parent2.velocity)-jitter2, max(parent1.velocity, parent2.velocity)+jitter2)
    
    child2_impact_angle = np.random.uniform(min(parent1.impact_angle, parent2.impact_angle)-jitter1, max(parent1.impact_angle, parent2.impact_angle)+jitter1)
    child2_velocity = np.random.uniform(min(parent1.velocity, parent2.velocity)-jitter2, max(parent1.velocity, parent2.velocity)+jitter2)
    
    child1 = Weapon(impact_angle=child1_impact_angle, velocity=child1_velocity)
    child2 = Weapon(impact_angle=child2_impact_angle, velocity=child2_velocity)
    
    return child1, child2

def mutate(weapon, mutation_rate, optimal_point):
    mutated = False  # Track if mutation occurs
    if np.random.rand() < mutation_rate:
        weapon.impact_angle = np.random.uniform(35, 90)
        mutated = True
    if np.random.rand() < mutation_rate:
        weapon.velocity = np.random.uniform(600, 1500)
        mutated = True
    if mutated:
        weapon.mutated = True  # Mark individual as mutated
        
    weapon.calc_fitness(optimal_point, fitness=fitness)  # Recalculate fitness with the current optimal_point

def selection(population):
    fitness_scores = [individual.fitness for individual in population]
    parents = np.random.choice(population, size=2, replace=False, p=fitness_scores/np.sum(fitness_scores))
    return parents

def create_generation(pop_size, optimal_point, fitness="default"):
    population = []
    for _ in range(pop_size):  # Use '_' for unused loop variable
        if np.random.rand() < 0.5:
            impact_angle_range = (35, 45)
            velocity_range = (1300, 1500)
        else:
            impact_angle_range = (80, 90)
            velocity_range = (600, 800)

        impact_angle = np.random.uniform(*impact_angle_range)
        velocity = np.random.uniform(*velocity_range)
        new_weapon = Weapon(impact_angle=impact_angle, velocity=velocity)
        population.append(new_weapon)  # Correctly append the new instance

    for weapon in population:
        weapon.calc_fitness(optimal_point, fitness=fitness)
        
    return population


def evolve(population, generations, mutation_rate, optimal_point, fitness="landscape"):
    for generation in range(generations):
        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, mutation_rate, optimal_point)
            mutate(child2, mutation_rate, optimal_point)
            new_population += [child1, child2]
        population = new_population[:len(population)]  # Ensure population size stays constant
        yield population

#Helper visualization function
def create_gif(image_folder, output_path, duration=1.0):
    images = []
    for file_name in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if file_name.endswith('.png'):
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(output_path, images, duration=duration, loop=0)

    # #Delete PNG files
    # for file_name in os.listdir(image_folder):
    #     if file_name.endswith('.png'):
    #         os.remove(os.path.join(image_folder, file_name))

    # Display the GIF
    display(HTML(f'<img src="{output_path}" />'))  

#Helper visualization code
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def get_color_for_value(value, min_val, max_val):
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.coolwarm)
    # Inverse the logic for the color scale
    return mapper.to_rgba(max_val - value + min_val)

def plot_population(population, generation, images_dir, optimal_point, fitness="default"):
    plt.figure(figsize=(10, 6))

    # Calculate averages
    avg_impact_angle = np.mean([weapon.impact_angle for weapon in population])
    avg_velocity = np.mean([weapon.velocity for weapon in population])

    if fitness == "default":
        # Plot the optimal point
        plt.scatter(optimal_point[0], optimal_point[1], color='red', s=100, edgecolors='k', label='Optimal Point')

    elif fitness == "landscape":
         # Generate x and y values for the fitness landscape
        x_values = np.linspace(35, 90, 100)
        y_values = np.linspace(600, 1500, 100)
        X, Y = np.meshgrid(x_values, y_values)
        Z = fitness_function(X, Y)
        
        # Increase the number of contour levels to get more color divisions
        num_levels = 100  # Adjust this value to increase/decrease the number of color divisions
        levels = np.linspace(0, 1, num_levels + 1)  # Ensure the levels cover the range of fitness values (0 to 1)

        # Plot the faded heatmap of the fitness landscape
        plt.contourf(X, Y, Z, cmap='nipy_spectral', alpha=0.3, levels=levels)
    
    # Plot each weapon in the population
    for weapon in population:
        color = 'green' if weapon.mutated else 'blue'
        plt.scatter(weapon.impact_angle, weapon.velocity, color=color, alpha=0.3)
    
    
    # Dummy plots for legend
    plt.scatter([], [], color='blue', label='Weapon')
    plt.scatter([], [], color='green', label='Mutated Weapon')
    
    plt.xlim(35, 90)
    plt.ylim(600, 1500)
    plt.xlabel('Impact Angle (degrees)')
    plt.ylabel('Velocity (fps)')
    plt.title(f'Generation {generation + 1}')
    
    # Display the averages inside the plot area
    plt.text(0.01, 0.05, f'Avg. Impact Angle: {avg_impact_angle:.2f}', transform=plt.gca().transAxes, 
             fontsize=9, ha='left', va='bottom')
    plt.text(0.01, 0.01, f'Avg. Velocity: {avg_velocity:.2f}', transform=plt.gca().transAxes, 
             fontsize=9, ha='left', va='bottom')
    
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(images_dir, f"generation_{generation + 1}.png"))
    plt.close()


#Generate gaussian peaks (Creates multiple maximums with a hill-like representation as might be seen in JWS)
def fitness_function(x, y):
    # Parameters for the Gaussian peaks
    peak1_x = 70
    peak1_y = 1300
    peak1_sigma_x = 5
    peak1_sigma_y = 100

    peak2_x = 50
    peak2_y = 800
    peak2_sigma_x = 7
    peak2_sigma_y = 100

    # Calculate Gaussian peaks
    peak1 = np.exp(-((x - peak1_x) ** 2 / (2 * peak1_sigma_x ** 2) + (y - peak1_y) ** 2 / (2 * peak1_sigma_y ** 2)))
    peak2 = np.exp(-((x - peak2_x) ** 2 / (2 * peak2_sigma_x ** 2) + (y - peak2_y) ** 2 / (2 * peak2_sigma_y ** 2)))

    # Combine the peaks to create the fitness landscape
    landscape = 0.6 * peak1 + 0.4 * peak2  # Adjust weights to control relative importance of peaks

    if np.isscalar(x) and np.isscalar(y):
        # If inputs were scalars, return the single landscape value
        return landscape
    else:
        # If inputs were arrays, return the array of landscape values
        return landscape

def run_genetic_algorithm(generation_limit, mutation_rate, pop_size, fitness="default", optimal_point=[85,900], output_type="gif"):
    """Run the genetic algorithm given the desired input parameters"""

    population = create_generation(pop_size=pop_size, optimal_point=optimal_point, fitness=fitness)

    #Directory to save images
    images_dir = "ga_plots"
    os.makedirs(images_dir, exist_ok=True)

    for gen_number, gen_population in enumerate(evolve(population, generation_limit, mutation_rate, optimal_point)):
        plot_population(gen_population, gen_number, images_dir, optimal_point, fitness=fitness)

    from IPython.display import display, HTML

    if output_type == "image" or output_type == "all":
        #Display generation 1, 2, 5, 10, 20, and 50:
        def display_image_square(images):
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(images[i])
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        images = [imageio.imread("ga_plots/generation_1.png"),imageio.imread("ga_plots/generation_2.png"),
                imageio.imread("ga_plots/generation_5.png"),imageio.imread("ga_plots/generation_10.png"),
                imageio.imread("ga_plots/generation_20.png"),imageio.imread("ga_plots/generation_50.png")]

        display_image_square(images)

    elif output_type == "gif" or output_type == "all":    
        # Call this function after generating all images
        gif_path = create_gif(images_dir, 'ga_evolution.gif', duration=0.5)

        
if __name__ == "__main__":
    generation_limit = 50
    mutation_rate = .01
    pop_size = 50
    fitness = "landscape"
    optimal_point = [85,900]
    output_type = "image"

    run_genetic_algorithm(generation_limit, mutation_rate, 
                        pop_size, fitness=fitness, 
                        optimal_point=optimal_point, output_type=output_type)
