#!/usr/bin/env python3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import csv
import sys

with open(sys.argv[1], 'r') as fid:
    csv_reader = csv.reader(fid, delimiter=',')
    rows = [r for r in csv_reader]

x_dim = int(rows[0][0])
y_dim = int(rows[1][0])
data = np.array(rows[2][:-1], dtype=np.float)

x = np.arange(x_dim)
y = np.arange(y_dim)
X, Y = np.meshgrid(x, y)
Z = data.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature')
ax.set_zlim(-1, 1)

plt.savefig("plot.png")
