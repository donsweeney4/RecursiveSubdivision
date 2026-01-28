import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load input data
df = pd.read_csv("InputData.csv")
df = df.dropna(subset=['longitude', 'latitude', 'temperature'])

# Extract coordinates and values
xs = df['longitude'].values
ys = df['latitude'].values
temps = df['temperature'].values

# Define interpolation grid
grid_x = np.linspace(xs.min(), xs.max(), 300)
grid_y = np.linspace(ys.min(), ys.max(), 300)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

# Interpolate using linear method
zi = griddata((xs, ys), temps, (grid_xx, grid_yy), method='linear')

cmap = plt.cm.get_cmap("coolwarm")

# 🔹 Plot 1: Contour with data points
fig1, ax1 = plt.subplots(figsize=(10, 8))
contour1 = ax1.contourf(grid_xx, grid_yy, zi, levels=20, cmap=cmap)
plt.colorbar(contour1, ax=ax1, label="Temperature (°F)")
ax1.scatter(xs, ys, c=temps, cmap=cmap, edgecolors='k', s=20)
ax1.set_title("Temperature Contour Map (With Data Points)")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_aspect("equal", adjustable="box")

# 🔹 Plot 2: Contour only (no data points)
fig2, ax2 = plt.subplots(figsize=(10, 8))
contour2 = ax2.contourf(grid_xx, grid_yy, zi, levels=20, cmap=cmap)
plt.colorbar(contour2, ax=ax2, label="Temperature (°F)")
ax2.set_title("Temperature Contour Map (No Data Points)")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_aspect("equal", adjustable="box")

plt.show()
