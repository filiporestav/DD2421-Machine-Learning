import monkdata as m
import dtree as d
import random
import drawtree_qt5 as draw
import matplotlib.pyplot as plt
import numpy as np

class Tree():
    def __init__(self, root):
        self.root = root

    def build_tree(self, dataset, attributes):
        pass

class TreeNode():
    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'


class Lab1():

    # Assignment 1
    def monk_entropies(self, monk):
        return d.entropy(monk)

    """
    Assignment 2:
    Uniform distribution: maximum (high) entropy because the uncertainty is high (e.g. choosing the correct letter in the alphabet randomly)
    Skewed/heavily undistributed distribution: low entropy, high certainty (e.g. guessing the name of a child, most common names have high frequency)
    """

    # Assignment 3
    def information_gain(self, monk):
        """
        Calculates the information gain for each attribute in the monk datasets
        """
        for attr in m.attributes:
            print("Information gain for attribute %s: %.15f" % (attr, d.averageGain(monk, attr)))

    """
    Assignment 4:

    The entropy of the subsets should be as low as possible (0), 
    to maximize the information gain.

    Using the information gain as a heuristic for picking an
    attribute for splitting is logical because if determines
    how efficient we can split the attributes and how much
    certainty we have on the new subtree.

    If we have a subset which is a large proportion of the set,
    and the distribution within that subset is uneven,
    we gain a large amount of information by incorporating it
    in the decision tree.
    """


    def build_tree(self, monk, root):
        # Select attribute A5 as the root node for monk1
        print(f"Root node: {root}")
        print(f"Splitting on {root}:")

        initial_entropy = d.entropy(monk)
        print(f"Initial entropy: {initial_entropy}")

        # Print the values of the root node
        values_dict = {}
        for v in root.values: # For each value of the root node (1-4 for A5)
            #print(f"Value {v}:")
            list_of_values = []
            for x in d.select(monk, root, v):
                #print(f"    {x}")
                list_of_values.append(x)
            values_dict[v] = list_of_values
        
        weighted_entropy_dict = {}
        for key, value in values_dict.items():
            weighted_entropy_dict[key] = d.entropy(value) * (len(value) / len(monk))
        
        print("Weighted entropy for each attribute:")
        for key, value in weighted_entropy_dict.items():
            print(f"  A{key}: {value}")
        
        print("Information gain for each attribute:")
        for key, value in weighted_entropy_dict.items():
            print(f"  A{key}: {initial_entropy - value}")
    
    def partition(self, data, fraction):
        """
        Split the data into two sets, determined by the fraction parameter.
        The first set is the fraction of the data, the second set is the rest.
        """
        ldata = list(data)
        random.shuffle(ldata)
        breakPoint = int(len(ldata) * fraction)
        return ldata[:breakPoint], ldata[breakPoint:]
    
    def test_fractions(self, monkdata, n, fractions):
        error_avg_arr = np.array([])
        error_std_arr = np.array([])

        for fraction in fractions:
            fraction_error = np.array([])
            for i in range(n):
                train, val = self.partition(monkdata, fraction)
                monk1tree = d.buildTree(train, m.attributes)
                # Get all the possible pruned trees
                monk1pruned_trees = d.allPruned(monk1tree)
                # Pick the tree with the best performance on validation set
                best_tree = max(monk1pruned_trees, key=lambda t: d.check(t, val))
                fraction_error = np.append(fraction_error, 1 - d.check(best_tree, val))

            error_avg_arr = np.append(error_avg_arr, fraction_error.mean())
            error_std_arr = np.append(error_std_arr, fraction_error.std())

        return error_avg_arr, error_std_arr

def __main__():
    lab = Lab1()
    for i, monk in enumerate([m.monk1, m.monk2, m.monk3], 1):
        print(f"Entropy for monk {i} dataset: %.15f" % lab.monk_entropies(monk))
    print()
    for i, monk in enumerate([m.monk1, m.monk2, m.monk3], 1):
        print(f"Information gain for monk {i} dataset:")
        lab.information_gain(monk)
        print()

    #lab.build_tree(m.monk1, m.attributes[4])

    monk1tree = d.buildTree(m.monk1, m.attributes)
    print("Monk 1 tree:")
    print("Performance on training data", d.check(monk1tree, m.monk1))
    print("Error rate on training data", 1 - d.check(monk1tree, m.monk1))
    print("Performance on test data", d.check(monk1tree, m.monk1test))
    print("Error rate on test data", 1 - d.check(monk1tree, m.monk1test))
    print()

    monk2tree = d.buildTree(m.monk2, m.attributes)
    print("Monk 2 tree:")
    print("Performance on training data", d.check(monk2tree, m.monk2))
    print("Error rate on training data", 1 - d.check(monk2tree, m.monk2))
    print("Performance on test data", d.check(monk2tree, m.monk2test))
    print("Error rate on test data", 1 - d.check(monk2tree, m.monk2test))
    print()

    monk3tree = d.buildTree(m.monk3, m.attributes)
    print("Monk 3 tree:")
    print("Performance on training data", d.check(monk3tree, m.monk3))
    print("Error rate on training data", 1 - d.check(monk3tree, m.monk3))
    print("Performance on test data", d.check(monk3tree, m.monk3test))
    print("Error rate on test data", 1 - d.check(monk3tree, m.monk3test))
    print()

    # draw.drawTree(monk3tree)

    """
    Evaluate the effect pruning has on the test error for
    the monk1 and monk3 datasets, in particular determine the optimal
    partition into training and pruning by optimizing the parameter
    fraction. Plot the classification error on the test sets as a function
    of the parameter fraction ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.

    Note that the split of the data is random. We therefore need to
    compute the statistics over several runs of the split to be able to draw
    any conclusions. Reasonable statistics includes mean and a measure
    of the spread. Do remember to print axes labels, legends and data
    points as you will not pass without them
    """
    n = 10000
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    error_avg_arr_monk1, error_std_arr_monk1 = lab.test_fractions(m.monk1, n, fractions)
    error_avg_arr_monk3, error_std_arr_monk3 = lab.test_fractions(m.monk3, n, fractions)

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for Monk 1
    axs[0].errorbar(fractions, error_avg_arr_monk1, yerr=error_std_arr_monk1, fmt='o', color='blue', capsize=5, linestyle='-', marker='s')
    axs[0].set_title('Monk 1', fontsize=14)
    axs[0].set_xlabel('Fraction', fontsize=12)
    axs[0].set_ylabel('Error rate (Test set)', fontsize=12)
    axs[0].grid(True)

    # Plot for Monk 3
    axs[1].errorbar(fractions, error_avg_arr_monk3, yerr=error_std_arr_monk3, fmt='o', color='green', capsize=5, linestyle='-', marker='^')
    axs[1].set_title('Monk 3', fontsize=14)
    axs[1].set_xlabel('Fraction', fontsize=12)
    axs[1].grid(True)

    # Adjust the layout to prevent overlap and improve readability
    plt.tight_layout()

    # Show the plots
    plt.show()


__main__()