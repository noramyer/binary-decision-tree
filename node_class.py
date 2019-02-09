class Node:

    def __init__(self, data, impurity_method, level):
        self.data_idx = data
        self.impurity_method = impurity_method
        self.nlevels = level
        self.impurity = None
        self.dfeature = None
        self.nfeatures = None
        self.class = None
        self.left_child = None
        self.right_child = None

        #calc impurity
        #initialize other variables

    def build_decision_tree(self, data, label, impurity_method, nl, p):
        decision_tree = Node(data, impurity_method, 0)
        decision_tree.splitNode(nl, p)

        return decision_tree

    def split_node(self, nl, p):

        if self.nLevels < nl and self.impurity > p:
            max_gain = -1.0
            splitFeature = None

            for f in range(self.nfeatures):
                data_left = []
                data_right = []

                for row in self.data_idx:
                    if row[f] == 1:
                        data_right.append(row)
                    else:
                        data_left.append(row)

                p_left = calculate_IP(data_left)
                p_right = calculate_IP(data_right)

                M = (len(data_left)/len(self.data_idx) * p_left) + (len(data_right)/len(self.data_idx) * data_right)
                gain = self.impurity - M

                if gain > max_gain:
                    max_gain = gain
                    splitFeature = f

            self.dfeature = splitFeature

            data_idx_left = []
            data_idx_right = []

            for row in self.data_idx:
                if row[splitFeature] == 1:
                    data_idx_right.append(row)
                else:
                    data_idx_left.append(row)

            self.left_child = Node(data_idx_left, impurity_method, self.nLevels + 1)
            self.right_child = Node(data_idx_right, impurity_method, self.nLevels + 1)

            self.left_child.split_node(nl, p)
            self.right_child.split_node(nl, p)

    def calculate_IP(self, data, impurity_method):
        p = 0.0
        if impurity_method == "gini":
            p = calculate_gini(data_idx)
        elif impurity_method == "entropy":
            p = calculate_entropy(data_idx)
        return p

    def calculate_gini(data):
        size = len(data)
        return None

    def calculate_entropy(data):
        return None
