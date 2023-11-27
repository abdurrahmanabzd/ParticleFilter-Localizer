import numpy as np
import matplotlib.pyplot as plt

def mapping(file_path):
    # Import Map file and save it in an array
    with open(file_path, 'r') as file:
        map_data = file.read()

    rows = map_data.strip().split("\n")
    mapArray = np.array([[int(grid) for grid in row.split()] for row in rows])

    mapInverted = 1 - mapArray

    plt.imshow(mapInverted, cmap='gray')
    plt.show()
# Example usage:
# file_path = 'map2023.dat'
# mapping(file_path)


def motion_update_odometry(particles, delta_rot1, delta_trans, delta_rot2):
    """
    Update particle positions based on odometry motion model.

    Parameters:
    - particles: numpy array of shape (num_particles, 3) representing particles [x, y, theta]
    - delta_rot1: array of delta rotation1 values for each particle
    - delta_trans: array of delta translation values for each particle
    - delta_rot2: array of delta rotation2 values for each particle

    Returns:
    - Updated particles array after odometry motion update.
    """
    particles[:, 0] += delta_trans * np.cos(particles[:, 2] + delta_rot1)
    particles[:, 1] += delta_trans * np.sin(particles[:, 2] + delta_rot1)
    particles[:, 2] += delta_rot1 + delta_rot2

    return particles

# Load robot data
robot_data = np.loadtxt('Robotdata2023.log')

# Particle filter parameters
num_particles = 1000
initial_particles = np.random.rand(num_particles, 3)  # Each particle is [x, y, theta]

# Initialize particles within a specific region
initial_particles[:, 0] = np.random.uniform(72, 85, num_particles)  # x position
initial_particles[:, 1] = np.random.uniform(445, 465, num_particles)  # y position
initial_particles[:, 2] = np.random.uniform(-0.1745, 0.1745, num_particles)  # theta

# Particle filter main loop
particles = initial_particles.copy()

for t in range(1, len(robot_data)):
    # Motion update based on odometry data
    if robot_data[t, 0] == 'O':
        delta_rot1 = robot_data[t, 3]
        delta_trans = robot_data[t, 4]
        delta_rot2 = robot_data[t, 5]
        
        particles = motion_update_odometry(particles, delta_rot1, delta_trans, delta_rot2)

# Plot the final particle positions
plt.scatter(particles[:, 1], particles[:, 0], s=1, c='b', marker='.')
plt.show()



# Initialize Particles
# - Particle Number
# - Initial Particles Position

# Import Robot Log
# if "L": 
# 1- Motion Model (odometry data)***

# 2- Sensor Model (laser data)***

# 3- Resampling (Low  Variance Algorithm) ***

# if "O":
# 1- Motion Model
  

# Plotting

# Video Creation

