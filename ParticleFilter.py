import numpy as np
import matplotlib.pyplot as plt

def process_map(file_path):
    # Import Map file and save it in an array
    with open(file_path, 'r') as file:
        map_data = file.read()

    rows = map_data.strip().split("\n")
    mapArray = np.array([[int(grid) for grid in row.split()] for row in rows])

    return mapArray
    

def draw_map(mapArray):
    mapInverted = 1 - mapArray

    plt.imshow(mapInverted, cmap='gray')
    plt.show()


def motion_update_odometry(particles, delta_rot1, delta_trans, delta_rot2):
    """
    Update particle positions based on odometry motion model.

    Parameters:
    - particles: numpy array of shape (num_particles, 4) representing particles [x, y, theta, weight]
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


def sensor_model(particles, lidar_data, map):
    """
    Update particles' weight based on lidar scan data.

    Parameters:
    - particles: numpy array of shape (num_particles, 4) representing particles [x, y, theta, weight]
    - lidar_data: lidar sensor scan data (180 beams)
    - map: grid map array

    Returns:
    - Updated particles array after odometry motion update.
    """
    z_max = 8000  #Maximum sensor range in cm (Zmax threshold)
    grid_resolution = 4 

    # Sensor model parameters:
    alpha_1 = 1e-6  #(rad^2/rad^2)
    alpha_2 = 1e-6  #(rad^2/cm^2)
    alpha_3 = 1e-6  #(cm^2/cm^2)
    alpha_4 = 1e-6  #(cm^2/rad^2)
    sigma = 5  #(cm)
    lmbda = 0.1  #(cm^-1)
    z_short = 0.09
    z_hit = 0.9
    z_max_prob = 0.01
    z_rand = 0.001

    for particle in particles:
        x_robot, y_robot, theta_robot, weight = particle

        laser_angles = np.deg2rad(np.arange(theta_robot - 89.5, theta_robot + 90, 1))
        ranges = np.array(lidar_data[3:183]) 

        x_laser = x_robot + 25 * np.cos(np.deg2rad(theta_robot))  #x_laser in global coordinates
        y_laser = y_robot + 25 * np.sin(np.deg2rad(theta_robot))  #y_laser in global coordinates

        x_endpoints = x_laser + ranges * np.cos(laser_angles)  #x-coordinates of endpoints
        y_endpoints = y_laser + ranges * np.sin(laser_angles)  #y-coordinates of endpoints

        updated_weight = 1.0  #Default weight

        for i in range(len(ranges)):
            if ranges[i] < z_max:
                # Convert endpoints to grid cell coordinates
                x_endpoint_cell = int(np.round(x_endpoints[i] / grid_resolution))
                y_endpoint_cell = int(np.round(y_endpoints[i] / grid_resolution))

                # Update probabilities within the map boundaries
                if 0 <= x_endpoint_cell < map.shape[0] and 0 <= y_endpoint_cell < map.shape[1]:
                    x_diff = x_endpoint_cell - x_robot / grid_resolution
                    y_diff = y_endpoint_cell - y_robot / grid_resolution
                    distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

                    # Sensor model calculations
                    if ranges[i] == z_max:
                        p_hit = z_max_prob
                    else:
                        p_hit = z_hit * np.exp(-0.5 * (distance ** 2) / (sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

                    p_short = z_short * (1 / (1 - np.exp(-lmbda * ranges[i]))) if ranges[i] < z_max else 0
                    p_rand = z_rand / z_max

                    prob = (p_hit + p_short + p_rand) / (p_hit + p_short + p_rand + z_max_prob)

                    # Update particle's weight based on sensor model
                    updated_weight *= prob

        # Update particle's weight in the particles array
        particle[3] = updated_weight

    return particles


#Processing map data
map_file = 'map2023.dat'
grid_map = process_map(map_file) #Grid-Map array
draw_map(grid_map)

# Load robot data
robot_data = np.loadtxt('Robotdata2023.log')

# Particle filter parameters
num_particles = 1000
initial_particles = np.random.rand(num_particles, 4)  # Each particle is [x, y, theta, weight]

# Initialize particles within a specific region
initial_particles[:, 0] = np.random.uniform(72, 85, num_particles)  # x position
initial_particles[:, 1] = np.random.uniform(445, 465, num_particles)  # y position
initial_particles[:, 2] = np.random.uniform(-0.1745, 0.1745, num_particles)  # theta
initial_particles[:, 3] = 1 #initial weight = 1

# Particle filter main loop
particles = initial_particles.copy()

for t in range(1, len(robot_data)):
    # Motion update based on odometry data
    if robot_data[t, 0] == 'O':
        delta_rot1 = robot_data[t, 3]
        delta_trans = robot_data[t, 4]
        delta_rot2 = robot_data[t, 5]
        
        particles = motion_update_odometry(particles, delta_rot1, delta_trans, delta_rot2)
    
    if robot_data[t, 0] == 'L':
    #Motion model:
        delta_rot1 = robot_data[t, 3]
        delta_trans = robot_data[t, 4]
        delta_rot2 = robot_data[t, 5]
        particles = motion_update_odometry(particles, delta_rot1, delta_trans, delta_rot2)
    #Sensor model:
        sensor_data = robot_data[4:186]
        sensor_model(particles,sensor_data,grid_map)


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

