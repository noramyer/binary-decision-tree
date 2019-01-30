class Node:

    def __init__(self, data, impurity_method, level):
        self.data_idx = data
        self.impurity_method = impurity_method
        self.nLevels = level
        self.impurity = None
        self.dfeature = None
        self.nfeatures = None



    def build_decision_tree(self, data, label, impurity_method, nl, p):
        decision_tree = Node(data, impurity_method, 0)
        decision_tree.splitNode(nl, p)

        return decision_tree

    def split_node(self, nl, p):

        if self.nLevels < nl and self.impurity > p:
            max_gain = -1.0
            splitFeature = None

            for f in range(self.nfeatures):
                p_left = calculate_IP(data_left)
                p_right = calculate_IP(data_right)

                M = impurity(data)
                gain = self.impurity - M

                if gain > max_gain:
                    max_gain = gain
                    splitFeature = f

            self.dfeature = splitFeature



    def calculate_IP(data, impurity_method):
        p = 0.0
        if impurity_method == "gini":
            p = calculate_gini(data_idx)
        elif impurity_method == "entropy":
            p = calculate_entropy(data_idx)

        return p
