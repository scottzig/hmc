"""
# This code implements a HMC algorithm in 2 dimensions on a simple
# Gaussian distribution

# Scott Ziegler
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define the Gaussian distribution we wish to sample from           
def gauss(x_1, x_2):
    return 1/(2*np.pi)*np.exp((-1/2)*(x_1**2+x_2**2))

def grad_U(x_1, x_2):
    par_x_1 = x_1
    par_x_2 = x_2
    return par_x_1, par_x_2

def Ham(x_1,x_2,p_1,p_2):
    return -np.log(gauss(x_1,x_2))+(1/2)*np.dot([p_1,p_2],[p_1,p_2])

# plot this distribution
x = np.arange(-4.0,4.0,0.2)
y = np.arange(-4.0,4.0,0.2)
X,Y = np.meshgrid(x, y) # grid of points
Z = gauss(X, Y) # evaluation of the function on the grid
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
#ax.set_title('Target Distribution')

# define the Hamiltonian MC algorithm
def hmc(target,grad,length,ep,L):
    start_q1, start_q2 = 4., 4. # starting q value of chain
    q_1,q_2 = start_q1, start_q2
    theta = np.zeros((length,2))
    acc = 0 # acceptance counter
    for i in range(length):
        p_1,p_2 = np.random.normal(size=2) # resample momentum from standard normal
        q_1n,q_2n = q_1, q_2
        p_1n,p_2n = p_1,p_2
        [p_1n,p_2n] = [p_1n,p_2n]-(ep/2)*np.array(grad(q_1n,q_2n)) # leapfrog steps
        for j in range(L):
            [q_1n,q_2n] = [q_1n,q_2n]+ep*np.dot(np.identity(2),np.array([p_1n,p_2n]))
            [p_1n,p_2n] = [p_1n,p_2n]-ep*np.array(grad(q_1n,q_2n))
        [p_1n,p_2n] = [p_1n,p_2n]-(ep/2)*np.array(grad(q_1n,q_2n))
        [p_1n,p_2n] = [-p_1n,-p_2n]
        if np.random.rand()<np.exp(Ham(q_1n,q_2n,p_1n,p_2n)-Ham(q_1,q_2,p_1,p_2)): #MH correction
            q_1, q_2 = q_1n, q_2n
            acc=acc+1
        theta[i]=np.array([q_1,q_2])
    return theta, acc   
        
#run the HMC algorithm on the density defined above and plot
sample_size=20
e=0.5
L_1=10
samples, numacc = hmc(gauss,grad_U,sample_size,e,L_1)
accrate=numacc/sample_size
ar = str(accrate)
ar = ar[:5]
plt.scatter(samples[:,0],samples[:,1])
print('Acceptance Rate:',accrate)

# animate the above chain
# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
# ax.set_title('HMC ' +ar)
# plt.xlabel('x_1')
# plt.ylabel('x_2')
# graph, = plt.plot([], [], 'o', markersize=4, alpha=1.4)
# samples_x = samples[:,0]
# samples_y = samples[:,1]

# def animate(i):
#     graph.set_data(samples_x[:i+1], samples_y[:i+1])
#     return graph

# ani = FuncAnimation(fig, animate, frames=sample_size, interval=10)
# plt.show()
# #ani.save('../../files/hmc_g.gif', writer='imagemagick', fps=60)