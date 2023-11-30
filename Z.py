import numpy as np
import matplotlib.pyplot as plt

# Particle filter parameters
num_particles = 1000
initial_particles = np.random.rand(num_particles, 3)  # Each particle is [x, y, theta]

# Initialize particles within a specific region
initial_particles[:, 0] = np.random.uniform(72, 85, num_particles)  # x position
initial_particles[:, 1] = np.random.uniform(445, 465, num_particles)  # y position
initial_particles[:, 2] = np.random.uniform(-np.pi, np.pi, num_particles)  # theta

# Particle filter odometry motion update function
def particle_filter_odometry_update(particles, delta_rot1, delta_trans, delta_rot2):
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

# Read the odometry data from the file
with open('abdurrahmanabzd/ParticleFilter-Localizer/Robotdata2023.log', 'r') as file:
    # Particle filter main loop
    particles = initial_particles.copy()

    # Read the file line by line
    for line in file:
        # Check if the line starts with 'O' (indicating odometry data)
        if line.startswith('O'):
            # Split the line into individual elements
            elements = line.split()

            # Extract relevant odometry data
            x_robot, y_robot, theta_robot, timestamp = map(float, elements[1:])

            # Calculate delta_rot1, delta_trans, and delta_rot2
            # Assuming delta_trans is the Euclidean distance from the previous position
            delta_trans = np.sqrt((x_robot - particles[0, 0])**2 + (y_robot - particles[0, 1])**2)
            delta_rot1 = np.arctan2(y_robot - particles[0, 1], x_robot - particles[0, 0]) - particles[0, 2]
            delta_rot2 = theta_robot - particles[0, 2]

            # Update particle positions based on odometry data
            particles = particle_filter_odometry_update(particles, delta_rot1, delta_trans, delta_rot2)

# Plot the final particle positions
plt.scatter(particles[:, 1], particles[:, 0], s=1, c='b', marker='.')
plt.show()
