"""
# This code implements a HMC algorithm in 2 dimensions on a multimodal
# distribution

# Scott Ziegler
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define the multimodal Gaussian distribution we wish to sample from   
mu_1 = -3
mu_2 = 3      
def gauss_mm(x_1, x_2):
    return 0.5*(1/(2*np.pi))*np.exp(-0.5*((x_1-mu_1)**2+(x_2-mu_2)**2)) \
             + 0.5*(1/(2*np.pi))*np.exp(-0.5*((x_1-mu_2)**2+(x_2-mu_1)**2))

def grad_U(x_1, x_2):
    par_x_1 = ((x_1-mu_1)*np.exp(x_2)+np.exp(x_1)*(x_1-mu_2))/(np.exp(x_1)+np.exp(x_2))
    par_x_2 = ((x_2-mu_1)*np.exp(x_1)+np.exp(x_2)*(x_2-mu_2))/(np.exp(x_1)+np.exp(x_2))
    return par_x_1, par_x_2

def Ham(x_1,x_2,p_1,p_2):
    return -np.log(gauss_mm(x_1,x_2))+(1/2)*np.dot([p_1,p_2],[p_1,p_2])

# plot this distribution
x = np.arange(-6.0,6.0,0.2)
y = np.arange(-6.0,6.0,0.2)
X,Y = np.meshgrid(x, y) # grid of points
Z = gauss_mm(X, Y) # evaluation of the function on the grid
#fig, ax = plt.subplots()
#CS = ax.contour(X, Y, Z)
#ax.set_title('Target Distribution')

# define the Hamiltonian MC algorithm
def hmc(target,grad,burn_in,length,ep,L):
    start_q1, start_q2 = 4., 4. # starting q value of chain
    start_p1, start_p2 = 0., 0. # starting p value of chain
    q_1,q_2 = start_q1, start_q2
    p_1,p_2 = start_p1, start_p2
    theta = np.zeros((length,2))
    acc = 0 # acceptance counter
    for i in range(burn_in):
        q_1p, q_2p = [q_1, q_2] + np.random.normal(size=2)
        if np.random.rand()<target(q_1p,q_2p)/target(q_1,q_2):
            q_1, q_2 = q_1p, q_2p
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
burn = 0
sample_size=10000
e=0.1
L_1=6
samples, numacc = hmc(gauss_mm,grad_U,burn,sample_size,e,L_1)
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
ax.set_title('HMC ' +ar)
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
#ani.save('../../files/hmc_mm.gif', writer='imagemagick', fps=60)