import matplotlib.pyplot as plt
import numpy as np

d2 = np.array([258.711, 687.11, 781.245, 941.46, 1013.38])
d4 = np.array([425.091, 1062.53, 1227.23, 1434.69,1589.55])
d8 = np.array([747.688, 1616.65, 1847.65, 2215.07, 2522.87])

points = [2**16, 2**18, 2**20, 2**22, 2**24]
points = [p/10**6 for p in points]

plt.figure()
plt.title('Bandwidth, GPU shared, 2000 iterations')
plt.loglog(points, d2, "o--", label='order 2')
plt.loglog(points, d4, "o--", label='order 4')
plt.loglog(points, d8, "o--", label='order 8')
plt.legend()
plt.xlabel('Number of gridpoints (millions)')
plt.ylabel('Bandwidth (GB/sec)')
plt.savefig('plot_bandwidth_shared.png')