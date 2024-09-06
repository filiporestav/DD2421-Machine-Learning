import monkdata as m
import dtree as d
import random

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
        for v in root.values:
            #print(f"  {v}:")
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

    """
    Write code which performs the complete pruning by repeatedly calling
    allPruned and picking the tree which gives the best classification performance on the validation dataset. You should stop pruning when all the
    pruned trees perform worse than the current candidate.
    """
    possible_trees = d.allPruned(monk1tree)
    print("Number of possible trees for monk1:", len(possible_trees))
    # Find the best tree for monk1
    best_tree = None
    for i in range(len(possible_trees)):
        if best_tree is None or d.check(possible_trees[i], m.monk1) > d.check(best_tree, m.monk1):
            best_tree = possible_trees[i]     
    print("Best tree for monk1:", best_tree)


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
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n=1000
    fractions_errors = {}
    for fraction in fractions:
        val_errors = 0
        deviations_sqrd_sum = 0
        for i in range(n):
            # Split the data into training and validation sets
            monk1train, monk1val = lab.partition(m.monk1, fraction)
            
            # Calculate the error rate for the training and validation sets
            tree = d.buildTree(monk1train, m.attributes)
            val_errors += 1 - d.check(tree, monk1val)

            # Calculate the standard deviation
            deviations_sqrd_sum += (1 - d.check(tree, monk1val))**2

        val_errors /= n
        deviations_sqrd_sum /= n
        std = (deviations_sqrd_sum - val_errors**2)**0.5
        fractions_errors[fraction] = (val_errors, std)
    
    print("Validation errors for different fractions:")
    for key, value in fractions_errors.items():
        print(f"  {key}: {value}")


__main__()