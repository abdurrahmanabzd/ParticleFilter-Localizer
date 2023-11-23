import numpy as np
import matplotlib.pyplot as plt

# Import Map file and save it in an array
file_path = 'map2023.dat'
with open(file_path, 'r') as file:
    map_data = file.read()

rows = map_data.strip().split("\n")
mapArray = np.array([[int(grid) for grid in row.split()] for row in rows])

mapInverted = 1 - mapArray

plt.imshow(mapInverted, cmap='gray')
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

