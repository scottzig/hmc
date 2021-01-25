"""
# This code implements a random-walk M-H algorithm in 2 dimensions
# assuming a symmetric proposal (Gaussian)

# Scott Ziegler
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#import turtle

# define the multimodal Gaussian distribution we wish to sample from   
mu_1 = -3
mu_2 = 3        
def gauss_mm(x_1, x_2):
    return 0.5*(1/(2*np.pi))*np.exp(-0.5*((x_1-mu_1)**2+(x_2-mu_2)**2)) + 0.5*(1/(2*np.pi))*np.exp(-0.5*((x_1-mu_2)**2+(x_2-mu_1)**2))

# plot this distribution
x = np.arange(-6.0,6.0,0.2)
y = np.arange(-6.0,6.0,0.2)
X,Y = np.meshgrid(x, y) # grid of points
Z = gauss_mm(X, Y) # evaluation of the function on the grid
#fig, ax = plt.subplots()
#CS = ax.contour(X, Y, Z)
#ax.set_title('Target Distribution')

# define the random walk mh algorithm
def rwmh(target,burn_in,length):
    start_1, start_2 = 5., 5.
    x_1, x_2 = start_1, start_2
    theta = np.zeros((length,2))
    acc = 0 # acceptance counter
    for i in range(burn_in):
        x_p1, x_p2 = [x_1, x_2] + np.random.normal(size=2)
        if np.random.rand()<target(x_p1,x_p2)/target(x_1,x_2):
            x_1, x_2 = x_p1, x_p2
    for i in range(length):
        x_p1, x_p2 = [x_1, x_2] + np.random.normal(size=2)
        if np.random.rand()<target(x_p1,x_p2)/target(x_1,x_2):
            x_1, x_2 = x_p1, x_p2
            acc=acc+1
        theta[i]=np.array([x_1,x_2])
    return theta, acc

# run the random walk mh algorithm on the density defined above and plot
burn_in_size=0
sample_size=500
samples, numacc = rwmh(gauss_mm,burn_in_size,sample_size)
accrate=numacc/sample_size
ar = str(accrate)
ar = ar[:5]
#plt.plot(samples[:,0],samples[:,1],'o', markersize=4, alpha=1.4)
#print('Acceptance Rate:',accrate)

# animate the above chain
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
ax.set_title('Random Walk ' +ar)
plt.xlabel('x_1')
plt.ylabel('x_2')
graph, = plt.plot([], [], 'o', markersize=4, alpha=1.4)
samples_x = samples[:,0]
samples_y = samples[:,1]

def animate(i):
    graph.set_data(samples_x[:i+1], samples_y[:i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=sample_size, interval=1)
plt.show()
#ani.save('../../files/random_walk_mm.gif', writer='imagemagick', fps=60)