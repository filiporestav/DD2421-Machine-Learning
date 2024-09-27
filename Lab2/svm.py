import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random

# Define the kernel function (linear kernel)
def linear_kernel(x, y):
    return np.dot(np.transpose(x), y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(np.transpose(x), y)) ** p

def rbf_kernel(x, y, sigma=0.5):
    return np.exp(-(np.linalg.norm(x-y) ** 2) / (2 * sigma ** 2))

# Objective function for the optimization
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(alpha, P)) - np.sum(alpha)

# Compute P matrix for the dual problem
def P_matrix(X, Y, kernel):
    n_samples, n_features = X.shape
    P = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            P[i, j] = Y[i] * Y[j] * kernel(X[i], X[j])
    return P

# Zerofun constraint for the optimization problem
def zerofun(alpha):
    return np.dot(alpha, targets)

def indicator(x, y):
    ind = -b
    for i in range(len(sv)):
        ind += sv[i][2] * sv[i][1] * KERNEL([x, y], sv[i][0])
    return ind

def get_data(x1a=1.5, y1a=0.5, x2a=-1.5, y2a=0.5, x1b=0.0, y1b=-0.5, n1a=10, n2a=10, nb=20, std=0.3):
    np.random.seed(100) # Ensure same data every time
    classA = np.concatenate((np.random.randn(n1a, 2) * std + [x1a, y1a], np.random.randn(n2a, 2) * std + [x2a, y2a]))
    classB = np.random.randn(nb, 2) * std + [x1b, y1b]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0] # Number of rows (samples)

    permute = list(range(N))
    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB, N

def get_support_vectors():
    upper_bound = None
    if C:
        upper_bound = C
    ret = minimize(objective, np.zeros(N), bounds=[(0, upper_bound) for _ in range(N)], constraints={'type':'eq', 'fun':zerofun})
    alpha = ret['x']
    success = ret['success']
    if not success:
        print("Optimization failed")
        return None
    # Collect support vectors (those with non-zero alphas)
    sv = []
    threshold = 10e-5
    for i in range(len(alpha)):
        if alpha[i] > threshold:
            sv.append((inputs[i], targets[i], alpha[i]))
    if len(sv) == 0: 
        print("No support vectors found")
        return None
    return sv
        
def compute_bias(sv):
    b = 0
    for i in range(len(sv)):
        b += sv[i][2] * sv[i][1] * KERNEL(sv[0][0], sv[i][0])
    b -= sv[0][1]
    return b

def plot(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'r.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'b.')

    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace(-4, 4)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('blue', 'black', 'red'), linewidths=(1, 3, 1))

    plt.axis('equal') # Force same scale on both axes
    plt.savefig('svmplot.png') # Save a copy in a file
    return plt

# Helper function to format input parameters for display
def format_params(index, params):
    return f"x1a={params['x1a'][index]}, y1a={params['y1a'][index]}, x2a={params['x2a'][index]}, y2a={params['y2a'][index]}, " \
           f"x1b={params['x1b'][index]}, y1b={params['y1b'][index]}, n1a={params['n1a'][index]}, n2a={params['n2a'][index]}, nb={params['nb'][index]}, std={params['std'][index]}"

def explore_kernel():
    global inputs, targets, classA, classB, N
    inputs, targets, classA, classB, N = get_data(
        1.5,
        0.5,
        -1.5,
        0.5,
        0,
        -0.5,
        10,
        10,
        20,
        0.3
    )
    global P
    P = P_matrix(inputs, targets, KERNEL)
    global sv
    sv = get_support_vectors()
    if sv is not None:
        global b
        b = compute_bias(sv)
        plotViz = plot(classA, classB)
        plotViz.show()

C = 100
KERNEL = linear_kernel
explore_kernel()