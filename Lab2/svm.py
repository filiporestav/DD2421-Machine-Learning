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

# Helper function to format input parameters for display
def format_params(index, params):
    return f"x1a={params['x1a'][index]}, y1a={params['y1a'][index]}, x2a={params['x2a'][index]}, y2a={params['y2a'][index]}, " \
           f"x1b={params['x1b'][index]}, y1b={params['y1b'][index]}, n1a={params['n1a'][index]}, n2a={params['n2a'][index]}, nb={params['nb'][index]}"





# Define the input variables to test
test_params = {
    'x1a': random.sample(range(-5, 5), 6),
    'y1a': random.sample(range(-5, 5), 6),
    'x2a': random.sample(range(-5, 5), 6),
    'y2a': random.sample(range(-5, 5), 6),
    'x1b': random.sample(range(-5, 5), 6),
    'y1b': random.sample(range(-5, 5), 6),
    'n1a': random.sample(range(1, 20), 6),
    'n2a': random.sample(range(1, 20), 6),
    'nb': random.sample(range(1, 20), 6),
    'std': random.sample(range(1, 10), 6),
}

# Define kernel functions and parameters to test
kernel_functions = [
    {'name': 'Linear Kernel', 'func': linear_kernel},
    {'name': 'Polynomial Kernel', 'func': polynomial_kernel, 'params': {'p': [2, 3, 5]}},
    {'name': 'RBF Kernel', 'func': rbf_kernel, 'params': {'sigma': [0.1, 0.5, 1.0]}}
]

for kernel in kernel_functions:
        KERNEL = kernel['func']  # Set the kernel
        kernel_params = kernel.get('params', None)
        print(f"Testing with {kernel['name']}")
        
        # Test with different random data samples
        for i in range(6):
            inputs, targets, classA, classB, N = get_data(
                test_params['x1a'][i], test_params['y1a'][i],
                test_params['x2a'][i], test_params['y2a'][i],
                test_params['x1b'][i], test_params['y1b'][i],
                test_params['n1a'][i], test_params['n2a'][i],
                test_params['nb'][i], test_params['std'][i]
            )

            P = P_matrix(inputs, targets, KERNEL)
            sv = get_support_vectors()
            
            if sv is not None:
                
                b = compute_bias(sv)
                plotViz = plot(classA, classB)
            
                # Add parameter info to the plot
                plotViz.text(-5, -5, format_params(i, test_params), fontsize=8)
                plotViz.show()

                # If kernel has additional parameters, test with those too
                if kernel_params:
                    for param_name, param_values in kernel_params.items():
                        for param_value in param_values:
                            print(f"Testing {kernel['name']} with {param_name}={param_value}")
                            
                            # Re-define the kernel with new parameters (e.g., polynomial degree, sigma for RBF)
                            if kernel['name'] == 'Polynomial Kernel':
                                KERNEL = lambda x, y: polynomial_kernel(x, y, p=param_value)
                            elif kernel['name'] == 'RBF Kernel':
                                KERNEL = lambda x, y: rbf_kernel(x, y, sigma=param_value)
                            
                            # Run the SVM again with the new kernel configuration
                            P = P_matrix(inputs, targets, KERNEL)
                            sv = get_support_vectors()
                            
                            if sv is None:
                                print(f"Optimization failed for {kernel['name']} with {param_name}={param_value}")
                                continue
                            
                            b = compute_bias(sv)
                            plotViz = plot(classA, classB)
                            
                            # Add kernel parameter info to the plot
                            plotViz.text(-5, -6, f"{param_name}={param_value}", fontsize=8)
                            plotViz.show()
            else:
                print(f"Optimization failed with parameters: {format_params(i, test_params)}")