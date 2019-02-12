import numpy as np
import math

#Author: Nora Myer
#Date: 2/5/19
#Description: This is a node class which represents one node in the binary search tree
#which splits based on the data at the node based on one of two different gain calculation methods
class Node:

    #Node constructor
    def __init__(self, data, labels, impurity_method, level):
        #Initialize attributes in node based on input
        self.data_idx = data
        self.data_idx_labels = labels
        self.impurity_method = impurity_method
        self.nlevels = level
        self.nfeatures = len(data[0])

        #Calc impurity depending on the input method
        self.impurity  = self.calculate_IP(data, labels)

        #initialize other variables as Null for now, they will be set later
        self.left_child = None
        self.right_child = None
        self.class_label = None
        self.dfeature = None

    #This method will be called on a node to begin splitting the decision_tree and return root node
    def build_decision_tree(self, data, labels, impurity_method, nl, p):
        #Initialize new root node
        decision_tree = Node(data, labels, impurity_method, 0)

        #Begin splitting
        decision_tree.split_node(nl, p)

        return decision_tree

    #Split node is called on each node to split on a single feature based on the feature with maxmimum gain
    def split_node(self, nl, p):
        #If we havent reached max depth or minimum impurity
        if self.nlevels < nl and self.impurity > p:
            max_gain = -1.0
            splitFeature = None

            #for each feature, make two new sets and calc the impurity of splitting the node on this feature
            for f in range(self.nfeatures):
                data_left = []
                data_right = []
                labels_left = []
                labels_right = []

                for idx, row in enumerate(self.data_idx):
                    if row[f] == 1:
                        data_right.append(row)
                        labels_right.append(self.data_idx_labels[idx])
                    else:
                        data_left.append(row)
                        labels_left.append(self.data_idx_labels[idx])

                #Based on this split, calc the new impurity
                p_left = self.calculate_IP(data_left, labels_left)
                p_right = self.calculate_IP(data_right, labels_right)

                #Calc the gain of this split
                M = (float(len(data_left))/len(self.data_idx) * p_left) + (float(len(data_right))/len(self.data_idx) * p_right)
                gain = self.impurity - M

                #If this split is maximum, save it
                if gain > max_gain:
                    max_gain = gain
                    splitFeature = f

            #set node feature split equal to the split the max gain
            self.dfeature = splitFeature

            #now, split the data based on the max gain feature
            data_idx_left = []
            data_idx_right = []
            labels_left = []
            labels_right = []

            #splitting data sets
            for idx, row in enumerate(self.data_idx):
                if row[splitFeature] == 1:
                    data_idx_right.append(row)
                    labels_right.append(self.data_idx_labels[idx])

                else:
                    data_idx_left.append(row)
                    labels_left.append(self.data_idx_labels[idx])

            #now, create the two children nodes based on the split data
            self.left_child = Node(data_idx_left, labels_left, self.impurity_method, self.nlevels + 1)
            self.right_child = Node(data_idx_right, labels_right, self.impurity_method, self.nlevels + 1)

            #recursive call to the two children nodes
            self.left_child.split_node(nl, p)
            self.right_child.split_node(nl, p)

        #If youve reached the bounds of the tree or there is low impurity, stop splitting and set the class equal
        #to the most common class occurance in current data set
        else:
            self.class_label = max(self.data_idx_labels,key=self.data_idx_labels.count)

        return

    #Based on the impurity method input, calc either gini or entropy
    def calculate_IP(self, data, labels):
        p = 0.0
        if self.impurity_method == "gini":
            p = self.calculate_gini(data, labels)
        elif self.impurity_method == "entropy":
            p = self.calculate_entropy(data, labels)
        return p

    #Called on a node, will return the class given a test row of data based on its features
    def classify(self, row):
            #if we reached a split node, return the classification
            if self.left_child == None and self.right_child == None:
                return self.class_label
            #otherwise, split based on the feature value of the data
            elif row[self.dfeature] == 0:
                return self.left_child.classify(row)
            else:
                return self.right_child.classify(row)

    #Calc gini on data set based on class labels
    def calculate_gini(self, data, labels):
        size = len(data)
        sum = 0.0

        #for each class, calc the prob in class given the current split
        for l in set(labels):
            prob = (labels == l).sum() / float(len(data))
            sum += math.pow(prob, 2)
        return 1.0 - sum

    def calculate_entropy(self, data, labels):
        size = len(data)
        sum = 0.0

        #for each class, calc the prob in class given the current split
        for l in set(labels):
            prob = (labels == l).sum()  / len(data)

            #get the log prob
            if prob != 1.0:
                sum += prob * math.log(prob, 2)

        return -1.0 * sum
