import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import imageio



# Particle filter parameters
num_particles = 1000
initial_particles = np.random.rand(num_particles, 4)  # Each particle is [x, y, theta, weight]

# Initialize particles within a specific region
initial_particles[:, 0] = np.random.uniform(445, 465, num_particles)  # x position
initial_particles[:, 1] = np.random.uniform(72, 85, num_particles)  # y position
initial_particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)  # theta
initial_particles[:, 3] = 1 #initial weight = 1


#region Functions

#Creating a map array out of map data file. 
def process_map(file_path):
    # Import Map file and save it in an array
    with open(file_path, 'r') as file:
        map_data = file.read()

    rows = map_data.strip().split("\n")
    mapArray = np.array([[int(grid) for grid in row.split()] for row in rows])

    return mapArray

#Drawing a map of the environment (inverted, occupied grids will be shown in black)
def show_particles(mapArray,particles):
    mapInverted = 1 - mapArray
    # Plot the final particle positions on the map
    plt.imshow(mapInverted, cmap='gray')  # Display the map
    # Scatter plot for particles
    plt.scatter(particles[:, 0], particles[:, 1], s=1, c='b', marker='.')
    plt.show()

def motion_model(particles, delta_rot1, delta_trans, delta_rot2):
    """
    Update particle positions based on motion model.

    Parameters:
    - particles: numpy array of shape (num_particles, 4) representing particles [x, y, theta, weight]
    - delta_rot1: change in orientation before motion (rad)
    - delta_trans: translational change (cm)
    - delta_rot2: change in orientation after motion (rad)

    Returns:
    - Updated particles array with adjusted positions based on motion model.
    """
    alpha_1 = 1e-6
    alpha_2 = 1e-6
    alpha_3 = 1e-6
    alpha_4 = 1e-6

    # Add noise to motion model
    for i in range(len(particles)):
        # Extract particle's pose
        x, y, theta = particles[i, 0], particles[i, 1], particles[i, 2]

        # Add noise to each motion parameter
        delta_rot1_hat = delta_rot1 - np.random.normal(0, np.sqrt(alpha_1 * abs(delta_rot1) + alpha_2 * delta_trans))
        delta_trans_hat = delta_trans - np.random.normal(0, np.sqrt(alpha_3 * delta_trans + alpha_4 * (abs(delta_rot1) + abs(delta_rot2))))
        delta_rot2_hat = delta_rot2 - np.random.normal(0, np.sqrt(alpha_1 * abs(delta_rot2) + alpha_2 * delta_trans))

        # Update particle pose
        x_new = x + delta_trans_hat * np.cos(theta + delta_rot1_hat)
        y_new = y + delta_trans_hat * np.sin(theta + delta_rot1_hat)
        theta_new = (theta + delta_rot1_hat + delta_rot2_hat) % (2 * np.pi)

        # Update particle position in the array
        particles[i, 0] = x_new
        particles[i, 1] = y_new
        particles[i, 2] = theta_new

    return particles


def sensor_model(particles, sensor_data, grid_map):
    """
    Update particle weights based on sensor (laser) data.

    Parameters:
    - particles: numpy array of shape (num_particles, 4) representing particles [x, y, theta, weight]
    - sensor_data: list of 180 laser range scanner measurements (180 elements)
    - grid_map: 2D numpy array representing the occupancy grid map

    Returns:
    - Updated particles array with modified weights based on sensor data likelihood.
    """
    sigma = 5  # Laser measurement error (cm)
    z_max = 8000  # Maximum sensor range

    # Sensor model parameters
    z_short = 0.09
    z_hit = 0.9
    z_max_prob = 0.01
    z_rand = 0.001

    # Iterate through each particle
    for i in range(len(particles)):
        particle_x, particle_y, particle_theta = particles[i, 0], particles[i, 1], particles[i, 2]

        # Predicted laser scan from the particle's position
        predicted_scan = []
        for angle in range(-89, 91):  # Angles from -89 to 90 degrees
            angle_rad = np.radians(angle) + particle_theta
            x_end = int(particle_x + z_max * np.cos(angle_rad))
            y_end = int(particle_y + z_max * np.sin(angle_rad))

            # Ray casting to simulate laser measurement
            hit = False
            for r in range(1, z_max + 1):
                x_check = int(particle_x + r * np.cos(angle_rad))
                y_check = int(particle_y + r * np.sin(angle_rad))

                if x_check == x_end and y_check == y_end:
                    # Reached the maximum sensor range
                    break
                if x_check < 0 or y_check < 0 or x_check >= grid_map.shape[0] or y_check >= grid_map.shape[1]:
                    # Ray went outside the map
                    break
                elif grid_map[x_check, y_check] == 1:  # Hit an occupied cell
                    hit = True
                    predicted_scan.append(r)
                    break

            if not hit:
                predicted_scan.append(z_max)

        # Calculate likelihood based on predicted and actual sensor data
        likelihood = 1.0
        for j in range(len(sensor_data)):
            if sensor_data[j] < z_max:
                p_hit = np.exp(-0.5 * ((sensor_data[j] - predicted_scan[j]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                # Set p_hit to zero if the ray goes outside the map
                if predicted_scan[j] == z_max:
                    p_hit = 0
                likelihood *= z_hit * p_hit + z_rand * (1.0 / z_max)
            else:
                if sensor_data[j] == z_max:
                    likelihood *= z_max_prob
                else:
                    likelihood *= z_short

        # Update particle weight
        particles[i, 3] *= likelihood

    # Normalize weights
    particles[:, 3] /= np.sum(particles[:, 3])

    return particles


def resample(particles):
    num_particles = len(particles)
    weights = particles[:, 3]  # Extract weights from particles

    # Resampling wheel
    indices = np.zeros(num_particles, dtype=int)
    cumulative_sum = np.cumsum(weights)
    step_size = cumulative_sum[-1] / num_particles
    beta = 0.0

    for i in range(num_particles):
        beta += step_size
        while beta > cumulative_sum[indices[i]]:
            indices[i] += 1

    # Resample particles based on selected indices
    resampled_particles = particles[indices]
    resampled_particles[:, 3] = 1.0 / num_particles  # Reset weights to 1

    return resampled_particles


#endregion

#Drawing the map of the environment
map_file = 'map2023.dat'
grid_map = process_map(map_file) #Grid-Map array

#region Main Loop

# Read the odometry data from the file
with open('Robotdata2023.log', 'r') as file:
    # Particle filter main loop
    particles = initial_particles.copy()

    fig, ax = plt.subplots() # Initialize figure and axes for the animation
    frames = [] # Create an empty list to store frames
    
    i=0

    # Read the file line by line
    for line in file:
        # Split the line into individual elements
        elements = line.split()

        # Extract relevant odometry data and timestamp
        x_robot_cm, y_robot_cm, theta_robot = map(float, elements[1:4])
        # Convert cm to grid numbers
        x_robot = math.floor(x_robot_cm / 4)
        y_robot = math.floor(y_robot_cm / 4)

        timestamp = elements.pop()

        # Calculate delta_rot1, delta_trans, and delta_rot2
        delta_trans = np.sqrt((x_robot - particles[i, 0])**2 + (y_robot - particles[i, 1])**2)
        delta_rot1 = np.arctan2(y_robot - particles[i, 1], x_robot - particles[i, 0]) - particles[i, 2]
        delta_rot2 = theta_robot - particles[i, 2]

        #if(delta_trans!=0 and delta_rot1!=0)
        show_particles(grid_map, particles) # Show the map at the beginning of each loop

        # Check if the line starts with 'O' (indicating odometry data)
        if line.startswith('O'):
            # Update particle positions based on odometry data
            particles = motion_model(particles, delta_rot1, delta_trans, delta_rot2)
        
        # Check if the line starts with 'L' (indicating laser data)
        if line.startswith('L'):
            #Motion model:
            particles = motion_model(particles, delta_rot1, delta_trans, delta_rot2)
            #Sensor model:
            sensor_data_cm = [float(element) for element in elements[7:187]]
            sensor_data = [min(math.floor(sensor / 4), 8000 // 4) for sensor in sensor_data_cm]  # Convert to grid numbers
    
            particles = sensor_model(particles,sensor_data,grid_map)
            #Resampling
            particles = resample(particles)
    
        # Show and update particle positions after each step
        ax.clear()
        ax.imshow(1 - grid_map, cmap='gray')
        ax.scatter(particles[:, 0], particles[:, 1], s=1, c='b', marker='.')
        #plt.pause(0.001)  # Pause for a short time to update the plot
        
        # Convert the plot to an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
 
        # Add the current frame to the list
        frames.append(image)
        #frames.append(ax.imshow(1 - grid_map, cmap='gray')) # Save the current frame
        i+=1

    # ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True) # Create a video from saved frames
    # ani.save('particle_filter_animation.mp4', writer='ffmpeg') # Save the animation as an .mp4 file
    
     # Save frames as a video using imageio
    imageio.mimsave('particle_filter_animation.mp4', frames, fps=10)  # Adjust fps as needed


    plt.show()  # Display the final plot

#endregion


