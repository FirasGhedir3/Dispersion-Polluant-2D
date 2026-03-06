import numpy as np
from copy import deepcopy
from scipy.linalg import lu
from matplotlib import pyplot as plt
import matplotlib.animation as animation



def LU(A):
    U=deepcopy(A)   #U=A
    N=U.shape       #taille de U
    L=np.eye(N[0])
    #k: boucle sur les n-1 étapes de la factorisation LU
    #A chaque étape on met des zéros sous la diagonale de la colonne courante k
    for k in range(0,N[0]-1):
        for j in range(k+1,N[0]):
            alpha=U[j,k]/U[k,k]
            U[j,:]=U[j,:]-alpha*U[k,:]
            L[j,k]=alpha
    return (L,U)


def Descente(L,b):
    N=L.shape
    x=np.zeros((N[0],1));
    x[0]=b[0]/L[0,0]
    for i in range(1,N[0]):
        x[i]=(b[i]-np.dot(L[i,0:i],x[0:i]))/L[i,i]
    return x



def Remonte(U,b):
    N=U.shape
    x=np.zeros((N[0],1))
    n=N[0]
    x[n-1]=b[n-1]/U[n-1,n-1]
    for i in range(n-2,-1,-1):
        x[i]=(b[i]-np.dot(U[i,i+1:n],x[i+1:n]))/U[i,i]
    return x


#Test matrice TD2
print('\n**********************************************')
print('Test matrice TD2 \n')

b=np.array([[1.5],[4],[-14],[-6.5]])
A=np.array([[-2., 1.,-1., 1.],[2., 0., 4., -3.],[-4., -1., -12., 9.],[-2., 1., 1., -4.]])


(L,U)=LU(A)

print('L = \n',L)
print('U = \n',U)


#Verif:

print('Verif : L.U = \n', np.dot(L,U))
Pverif, Lverif, Uverif = lu(A)
print("P_verif = \n", Pverif, "\nL_verif = \n", Lverif, "\nU_verif = \n", Uverif)


y=Descente(L,b)

print('Après descente, y =\n',y)

x=Remonte(U,y)

print('Après remontée, x =\n',x)
print('**********************************************\n')




def Jacobi(A,b,x0,eps,Nmax):
    k=0
    x=x0
    N=-np.triu(A,1)-np.tril(A,-1)
    invM=np.diag(1.0/np.diag(A))
    normb=np.linalg.norm(b)
    residu=[]
    residu.append(np.linalg.norm(A*x-b)/normb)
    while(np.linalg.norm(A*x-b)/normb>eps and k<Nmax):
        x=invM*(N*x+b)
        residu.append(np.linalg.norm(A*x-b)/normb)
        k=k+1
    return (x,k,residu)




def GaussSeidel(A,b,x0,eps,Nmax):
    k=0
    x=x0
    N=-np.triu(A,1)
    M=np.tril(A)
    normb=np.linalg.norm(b)
    residu=[]
    residu.append(np.linalg.norm(A*x-b)/normb)
    while(np.linalg.norm(A*x-b)/normb>eps and k<Nmax):
        x=Descente(M,(np.dot(N,x)+b))  #M=D-E triangular inferior
        residu.append(np.linalg.norm(A*x-b)/normb)
        k=k+1
    return (x,k,residu)




#Matrice TD3
Cexo = np.array([[1, -2, -2], [-1, 1, -1], [2, -2, 1]])
mat_iter_C = np.array([[0., -2., 2.], [-1., 0., -1.], [-2., -2., 0.]])
eigen_val_mat_iter_C, eigen_vectors_mat_iter_C = np.linalg.eig(mat_iter_C)
print('Rayon spectral de la matrice d"iteration de Jacobi', max(abs(eigen_val_mat_iter_C)))
mat_iter_C_GS = np.dot(np.linalg.inv(np.array([[1., 0., 0.], [1., 1., 0.], [2., 2., 1.]])),np.array([[0., -2., 2.], [0., 0., -1.], [0., 0., 0.]]))
eigen_val_mat_iter_C_GS, eigen_vectors_mat_iter_C_GS = np.linalg.eig(mat_iter_C_GS)
print('Rayon spectral de la matrice d"iteration de Gauss-Seidel', max(abs(eigen_val_mat_iter_C_GS)))
print('Jacobi converge, Gauss Seidel ne converge pas')



x0=np.matrix(np.zeros((3,1)))
b=np.matrix(np.ones((3,1)))
(x,k,erreur_J)=Jacobi(Cexo,b,x0,1e-7,100)
print(erreur_J)
(x_GS,k_GS,erreur_GS)=GaussSeidel(Cexo,b,x0,1e-7,100)
plt.figure()
#plt.yscale('log')
plt.plot(np.arange(0,k+1),erreur_J,np.arange(0,k_GS+1),erreur_GS)
plt.legend(['Jacobi','Gauss-Seidel'])
plt.title('Convergence rates')
plt.xlabel('Number of iterations')
plt.ylabel('Precision')





def AssembleMatrix(nx, ny, dt, kappa, dx, dy, v, X, Y):
    A = np.zeros((nx * ny, nx * ny))
    
    # Inside nodes
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = i + (j - 1) * nx
            A[k, k] = 1 / (2 * dt) + kappa / (dx ** 2) + kappa / (dy ** 2)
            A[k, k + 1] = v[0] / (4 * dx) - kappa / (2 * dx ** 2)
            A[k, k - 1] = -v[0] / (4 * dx) - kappa / (2 * dx ** 2)
            A[k, k + nx] = v[1] / (4 * dy) - kappa / (2 * dy ** 2)
            A[k, k - nx] = -v[1] / (4 * dy) - kappa / (2 * dy ** 2)
    
    # Boundary conditions
    j = 0
    for i in range(nx):
        k = i + j * nx
        A[k, k] = 1
    
    i = 0
    for j in range(ny):
        k = i + j * nx
        A[k, k] = 1
    
    # Neumann condition - North
    j = ny - 1
    for i in range(1, nx - 1):
        k = i + j * nx
        A[k, k] = 1 / (2 * dt) + kappa / (dx ** 2) + kappa / (dy ** 2)
        A[k, k + 1] = v[0] / (4 * dx) - kappa / (2 * dx ** 2)
        A[k, k - 1] = -v[0] / (4 * dx) - kappa / (2 * dx ** 2)
        A[k, k - nx] = -kappa / (dy ** 2)
    
    # Neumann condition - East
    i = nx - 1
    for j in range(1, ny - 1):
        k = i + j * nx
        A[k, k] = 1 / (2 * dt) + kappa / (dx ** 2) + kappa / (dy ** 2)
        A[k, k - 1] = -kappa / (dx ** 2)
        A[k, k + nx] = v[1] / (4 * dy) - kappa / (2 * dy ** 2)
        A[k, k - nx] = -v[1] / (4 * dy) - kappa / (2 * dy ** 2)
    
    # Corner (x_nx, y_ny)
    i = nx - 1
    j = ny - 1
    k = i + j * nx
    A[k, k] = 1
    
    return A


def CondInitiale(X, Y, sigma, x0, y0):
    nx = X.shape[0]
    ny = X.shape[1]
    C0 = np.zeros(nx * ny)
    
    for i in range(nx):
        for j in range(ny):
            k = i + j * nx
            C0[k] = np.exp(-((X[i, j] - x0) / sigma) ** 2) * np.exp(-((Y[i, j] - y0) / sigma) ** 2)
    
    C0 = np.transpose(C0)
    return C0


def Rhs(nx, ny, dt, kappa, dx, dy, v, un, X, Y):
    A = np.zeros((nx * ny, nx * ny))
    b = np.zeros(nx * ny)
    
    # Inside nodes
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = i + (j - 1) * nx
            A[k, k] = 1 / dt - kappa / (dx ** 2) - kappa / (dy ** 2)
            A[k, k + 1] = -v[0] / (4 * dx) + kappa / (2 * dx ** 2)
            A[k, k - 1] = v[0] / (4 * dx) + kappa / (2 * dx ** 2)
            A[k, k + nx] = -v[1] / (4 * dy) + kappa / (2 * dy ** 2)
            A[k, k - nx] = v[1] / (4 * dy) + kappa / (2 * dy ** 2)
    
    # Neumann condition - North
    j = ny - 1
    for i in range(1, nx - 1):
        k = i + j * nx
        A[k, k] = 1 / dt - kappa / (dx ** 2) - kappa / (dy ** 2)
        A[k, k + 1] = -v[0] / (4 * dx) + kappa / (2 * dx ** 2)
        A[k, k - 1] = v[0] / (4 * dx) + kappa / (2 * dx ** 2)
        A[k, k - nx] = kappa / (dy ** 2)
    
    # Neumann condition - East
    i = nx - 1
    for j in range(1, ny - 1):
        k = i + j * nx
        A[k, k] = 1 / dt - kappa / (dx ** 2) - kappa / (dy ** 2)
        A[k, k - 1] = kappa / (dx ** 2)
        A[k, k + nx] = -v[1] / (4 * dy) + kappa / (2 * dy ** 2)
        A[k, k - nx] = v[1] / (4 * dy) + kappa / (2 * dy ** 2)
    
    # Corner (x_nx, y_ny)
    i = nx - 1
    j = ny - 1
    k = i + j * nx
    A[k, k] = 1
    
    b = np.dot(A, un)
    
    return b


# Parameters
nx = 30
ny = 30
kappa = 0.9
xmax = 100
ymax = 100
dx = xmax / (nx - 1)
dy = ymax / (ny - 1)
v = np.array([0.5, 0.5])  # Velocity
dt = 0.5

# Grid
x = np.linspace(0, xmax, nx)
y = np.linspace(0, ymax, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
x0 = 10.0
y0 = 10.0
sigma = 0.5
C = CondInitiale(X, Y, sigma, x0, y0)
print("C = \n",C)
Z = C.reshape(nx, ny)

# Enable interactive mode for real-time plotting
plt.ion()

# Create figure and initial plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Concentration')
ax.set_title('Pollutant Dispersion Simulation - Initial')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.pause(0.01)

# Handle window close event
window_closed = False
def on_close(event):
    global window_closed
    window_closed = True

fig.canvas.mpl_connect('close_event', on_close)

# Collect all frames for animation saving
Z_list = [Z.copy()]

A = AssembleMatrix(nx, ny, dt, kappa, dx, dy, v, X, Y)
print("A = \n", A)

# Iterative loop with real-time plotting
for t in range(200):  # 200 steps to make it 201 total including initial
    if window_closed:
        print("Plot window closed. Stopping simulation.")
        break
    print('t =', (t+1) * dt)
    b = Rhs(nx, ny, dt, kappa, dx, dy, v, C, X, Y)
    C, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    Z = C.reshape(nx, ny)
    Z_list.append(Z.copy())
    
    # Update the plot in real-time
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Concentration')
    ax.set_title(f'Pollutant Dispersion at t = {(t+1) * dt:.1f}')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.pause(0.01)

# Create animation from collected frames
if not window_closed:
    anim = animation.FuncAnimation(fig, lambda frame: (
        ax.clear(),
        ax.set_xlabel('X'),
        ax.set_ylabel('Y'),
        ax.set_zlabel('Concentration'),
        ax.set_title(f'Pollutant Dispersion at t = {frame * dt:.1f}'),
        ax.plot_surface(X, Y, Z_list[frame], cmap='viridis')
    )[-1], frames=len(Z_list), interval=100, blit=False)

    # Save as GIF
    anim.save('media/dispersion_simulation.gif', writer='pillow', fps=10)

    print("Animation saved as media/dispersion_simulation.gif")
else:
    print("Simulation stopped early due to window close. Animation not saved.")

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open