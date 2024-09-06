import monkdata as m
from dtree import entropy, averageGain, select, mostCommon, buildTree

class Lab1():

    """
    Assignment 0:

    """

    # Assignment 1
    def monk_entropies():
        print("Entropy of monk 1 (training set): ", entropy(m.monk1))
        print("Entropy of monk 2 (training set): ", entropy(m.monk2))
        print("Entropy of monk 3 (training set): ", entropy(m.monk3))

    """
    Assignment 2:

    Uniform distribution: maximum (high) entropy because the uncertainty is high (e.g. choosing the correct letter in the alphabet randomly)
    Skewed/heavily undistributed distribution: low entropy, high certainty (e.g. guessing the name of a child, most common names have high frequency)
    """

    # Assignment 3
    def information_gain():
        dict = {}
        for attribute in m.attributes:
            dict[f"Monk1 {attribute}"] = averageGain(m.monk1, attribute)
            dict[f"Monk2 {attribute}"] = averageGain(m.monk2, attribute)
            dict[f"Monk3 {attribute}"] = averageGain(m.monk3, attribute)
            
        print("{:<10} {:<10}".format('ATTRIBUTE', 'INFORMATION GAIN'))
        # print each data item.
        for key, value in sorted(dict.items(), key=lambda item: item[1]):
            inf_gain = value
            print("{:<10} {:<10}".format(key, inf_gain))
        print(f"We should use {max(dict, key=dict.get)} for splitting the examples at the root node")

    information_gain()

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

    # Assignment 5
    def build_decision_trees():
        # Select attribute A1 for monk 1
        subtree = select(m.monk1, m.attributes[4], 1) # for monk-1, a5=1  is a "true concept"
        subtree_entropy = entropy(subtree)
        
        #print("%.15f" % max_info_gain)
        #print(max_info_attr)


    build_decision_trees()