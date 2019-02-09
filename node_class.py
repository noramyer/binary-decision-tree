class Node:

    def __init__(self, data, labels, impurity_method, level):
        self.data_idx = data
        self.data_idx_labels = labels
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

    def build_decision_tree(data, labels, impurity_method, nl, p):
        decision_tree = Node(data, labels, impurity_method, 0)
        decision_tree.splitNode(nl, p)

        return decision_tree

    def split_node(self, nl, p):

        if self.nLevels < nl and self.impurity > p:
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

                p_left = calculate_IP(data_left, self.impurity_method, labels_left)
                p_right = calculate_IP(data_right, self.impurity_method, labels_right)

                M = (len(data_left)/len(self.data_idx) * p_left) + (len(data_right)/len(self.data_idx) * data_right)
                gain = self.impurity - M

                if gain > max_gain:
                    max_gain = gain
                    splitFeature = f

            self.dfeature = splitFeature

            data_idx_left = []
            data_idx_right = []
            labels_left = []
            labels_right = []

            for row in self.data_idx:
                if row[splitFeature] == 1:
                    data_idx_right.append(row)
                    labels_right.append(self.data_idx_labels[idx])

                else:
                    data_idx_left.append(row)
                    labels_left.append(self.data_idx_labels[idx])

            self.left_child = Node(data_idx_left, labels_left, impurity_method, self.nLevels + 1)
            self.right_child = Node(data_idx_right, labels_right, impurity_method, self.nLevels + 1)

            self.left_child.split_node(nl, p)
            self.right_child.split_node(nl, p)

    def calculate_IP(data, impurity_method, labels):
        p = 0.0
        if impurity_method == "gini":
            p = calculate_gini(data_idx, labels)
        elif impurity_method == "entropy":
            p = calculate_entropy(data_idx, labels)
        return p

    def calculate_gini(data, labels):
        size = len(data)
        for class in set(labels):
            prob = labels.count(class) / len(data)
            sum += np.square(prob)
        return 1.0 - sum

    def calculate_entropy(data, labels):
        size = len(data)
        sum = 0.0
        for class in set(labels):
            prob = labels.count(class) / len(data)

            if prob != 1.0:
                sum += prob * math.log(prob, 2)

        return -1.0 * sum
