"""
# This code implements a random-walk M-H algorithm in 2 dimensions
# assuming a symmetric proposal (Gaussian)

# Scott Ziegler
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#import turtle

# define the Gaussian distribution we wish to sample from   
mu_1 = -1.5
mu_2 = 1.5          
def gauss(x_1, x_2):
    return 1/(2*np.pi)*np.exp((-1/2)*(x_1**2+x_2**2))

# plot this distribution
x = np.arange(-4.0,4.0,0.2)
y = np.arange(-4.0,4.0,0.2)
X,Y = np.meshgrid(x, y) # grid of points
Z = gauss(X, Y) # evaluation of the function on the grid
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
sample_size=300
samples, numacc = rwmh(gauss,burn_in_size,sample_size)
accrate=numacc/sample_size
ar = str(accrate)
ar = ar[:5]
#plt.scatter(samples[:,0],samples[:,1])
#print('Acceptance Rate:',accrate)

# animate the above chain
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
ax.set_title('Random Walk ' +ar)
plt.xlabel('x_1')
plt.ylabel('x_2')
graph, = plt.plot([], [], 'o', markersize=4, alpha=1.4)
samples_x = samples[:,0]
samples_y = samples[:,1]

def animate(i):
    graph.set_data(samples_x[:i+1], samples_y[:i+1])
    return graph

ani = FuncAnimation(fig, animate, frames=sample_size, interval=70)
plt.show()
#plt.rcParams['animation.convert_path'] = '/Users/scottziegler/image_magick/magick.exe'
#ani.save('/Users/scottziegler/random_walk_g.gif', writer='imagemagick', fps=100)