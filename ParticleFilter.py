import numpy as np
import matplotlib.pyplot as plt

file_path = 'map2023.dat'
with open(file_path, 'r') as file:
    map_data = file.read()

rows = map_data.strip().split("\n")
mapArray = np.array([[int(grid) for grid in row.split()] for row in rows])

mapInverted = 1 - mapArray

plt.imshow(mapInverted, cmap='gray')
plt.show()