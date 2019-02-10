import numpy as np
import math

class Node:

    def __init__(self, data, labels, impurity_method, level):
        self.data_idx = data
        self.data_idx_labels = labels
        self.impurity_method = impurity_method
        self.nlevels = level
        self.nfeatures = len(data[0])

        #calc impurity
        self.impurity  = self.calculate_IP(data, labels)

        #initialize other variables
        self.left_child = None
        self.right_child = None
        self.class_label = None
        self.dfeature = None

    def build_decision_tree(self, data, labels, impurity_method, nl, p):
        decision_tree = Node(data, labels, impurity_method, 0)
        decision_tree.split_node(nl, p)

        return decision_tree

    def split_node(self, nl, p):
        if self.impurity == 0.0:
            print(set(self.data_idx_labels))
            print(len(set(self.data_idx_labels)))

        if(len(set(self.data_idx_labels)) == 1):
            self.class_label = self.data_idx_labels[0]
            return

        if self.nlevels < nl and self.impurity > float(p):
            max_gain = -1.0
            splitFeature = None

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

                p_left = self.calculate_IP(data_left, labels_left)
                p_right = self.calculate_IP(data_right, labels_right)

                M = (float(len(data_left))/len(self.data_idx) * p_left) + (float(len(data_right))/len(self.data_idx) * p_right)
                gain = self.impurity - M

                if gain > max_gain:
                    max_gain = gain
                    splitFeature = f

            self.dfeature = splitFeature

            data_idx_left = []
            data_idx_right = []
            labels_left = []
            labels_right = []

            for idx, row in enumerate(self.data_idx):
                if row[splitFeature] == 1:
                    data_idx_right.append(row)
                    labels_right.append(self.data_idx_labels[idx])

                else:
                    data_idx_left.append(row)
                    labels_left.append(self.data_idx_labels[idx])


            self.left_child = Node(data_idx_left, labels_left, self.impurity_method, self.nlevels + 1)
            self.right_child = Node(data_idx_right, labels_right, self.impurity_method, self.nlevels + 1)
            self.left_child.split_node(nl, p)
            self.right_child.split_node(nl, p)

        return

    def calculate_IP(self, data, labels):
        p = 0.0
        if self.impurity_method == "gini":
            p = self.calculate_gini(data, labels)
        elif self.impurity_method == "entropy":
            p = self.calculate_entropy(data, labels)
        return p

    def classify(self, row):
            if self.left_child == None and self.right_child == None:
                return self.class_label
            elif row[self.dfeature] == 0:
                return self.left_child.classify(row)
            else:
                return self.right_child.classify(row)

    def calculate_gini(self, data, labels):
        size = len(data)
        sum = 0.0
        for l in set(labels):
            prob = (labels == l).sum() / float(len(data))
            sum += math.pow(prob, 2)
        return 1.0 - sum

    def calculate_entropy(self, data, labels):
        size = len(data)
        sum = 0.0
        for l in set(labels):
            prob = (labels == l).sum()  / len(data)

            if prob != 1.0:
                sum += prob * math.log(prob, 2)

        return -1.0 * sum
