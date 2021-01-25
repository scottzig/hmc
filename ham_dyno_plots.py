"""
# This code runs a leapfrog integrator to simulate Hamiltonian
# dynamics in 1D for the purpose of creating plots in phase space

# Scott Ziegler
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define the multimodal Gaussian distribution we wish to sample from   
def gauss(x):
    return 1/(2*np.pi)*np.exp((-1/2)*x**2)

def grad_U(x):
    return x

def Ham(q,p):
    return -np.log(gauss(q))+(1/2)*p**2

# plot level sets of the Hamiltonian
x = np.arange(-5.0,5.0,0.2)
y = np.arange(-5.0,5.0,0.2)
X,Y = np.meshgrid(x, y) # grid of points
Z = Ham(X, Y) # evaluation of the function on the grid
#fig, ax = plt.subplots()
#CS = ax.contour(X, Y, Z)
#ax.set_title('Level Sets of Hamiltonian')

def leapfrog(target,ep,L):
    theta=np.zeros((L,2))
    q_1n = 2.
    p_1n = 2.
    for i in range(L):
        theta[i]=np.array([q_1n,p_1n])
        p_1n = p_1n-(ep/2)*target(q_1n)
        q_1n = q_1n+ep*p_1n
        p_1n = p_1n-(ep/2)*target(q_1n)
    return theta

#run the HMC algorithm on the density defined above and plot
e=0.3
L_1=4
result = leapfrog(grad_U,e,L_1)

# animate the above chain
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
#plt.plot(result[:,0],result[:,1],'-o')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
ax.set_title('Results of Leapfrog Integrator in Phase Space')
plt.xlabel('q')
plt.ylabel('p')
graph, = plt.plot([], [], '-o')
result_x = result[:,0]
result_y = result[:,1]

def animate_1(i):
    graph.set_data(result_x[:i+1], result_y[:i+1])
    return graph

ani = FuncAnimation(fig, animate_1, frames=L_1, interval=500)
plt.show()