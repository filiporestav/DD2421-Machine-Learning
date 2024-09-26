import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the kernel function (linear kernel)
def linear_kernel(x, y):
    return np.dot(np.transpose(x), y)

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
        ind += sv[i][2] * sv[i][1] * linear_kernel([x, y], sv[i][0])
    return ind

def get_data(x1a=1.5, y1a=0.5, x2a=-1.5, y2a=0.5, x1b=0.0, y1b=-0.5, n1a=10, n2a=10, nb=20):
    np.random.seed(100) # Ensure same data every time
    classA = np.concatenate((np.random.randn(n1a, 2) * 0.2 + [x1a, y1a], np.random.randn(n2a, 2) * 0.2 + [x2a, y2a]))
    classB = np.random.randn(nb, 2) * 0.2 + [x1b, y1b]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0] # Number of rows (samples)

    permute = list(range(N))
    np.random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets, classA, classB, N

def get_support_vectors():
    ret = minimize(objective, np.zeros(N), bounds=[(0, None) for _ in range(N)], constraints={'type':'eq', 'fun':zerofun})
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
    return sv
        
def compute_bias(sv):
    b = 0
    for i in range(len(sv)):
        b += sv[i][2] * sv[i][1] * linear_kernel(sv[0][0], sv[i][0])
    b -= sv[0][1]
    return b

def plot(classA, classB, show_input=False):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace(-4, 4)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.axis('equal') # Force same scale on both axes
    plt.savefig('svmplot.png') # Save a copy in a file
    return plt



# List of input variables to test (first is default values)
x1a = [1.5, 1, 2, 3, 4, 5]
y1a = [0.5, 1, 2, 3, 4, 5]
x2a = [-1.5, -1, -2, -3, -4, -5]
y2a = [0.5, -1, -2, -3, -4, -5]
x1b = [0.0, 1, 2, 3, 4, 5]
y1b = [-0.5, -1, -2, -3, -4, -5]

n1a = [10, 10, 20, 30, 40, 50]
n2a = [10, 10, 20, 30, 40, 50]
nb = [20, 20, 40, 60, 80, 100]

# Test the function with different input values
for i in range(6):
    inputs, targets, classA, classB, N =  get_data(x1a[i], y1a[i], x2a[i], y2a[i], x1b[i], y1b[i], n1a[i], n2a[i], nb[i])
    P = P_matrix(inputs, targets, linear_kernel)
    sv = get_support_vectors()
    b = compute_bias(sv)
    plotViz = plot(classA, classB)
    # Add the data variables (x1a, y1a, x2a, y2a, x1b, y1b, n1a, n2a, nb) as text to the plot, located below the x-axis
    plotViz.text(-5, -5, f"x1a={x1a[i]}, y1a={y1a[i]}, x2a={x2a[i]}, y2a={y2a[i]}, x1b={x1b[i]}, y1b={y1b[i]}, n1a={n1a[i]}, n2a={n2a[i]}, nb={nb[i]}", fontsize=8)
    plotViz.show()