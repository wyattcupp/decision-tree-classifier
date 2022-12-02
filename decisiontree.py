'''
Decision tree algorithm implementation using the Gini Impurity
splitting criteria.

Author: Wyatt Cupp <wyattcupp@gmail.com>
'''

import numpy as np


class Node:
    '''
    A class representing a Node in a Decision Tree.
    '''

    def __init__(self, gini, num_samples, pred_class):
        self.gini = gini  # total gini index score
        self.pred_class = pred_class
        self.num_samples = num_samples

        self.feat_index = None
        self.split_threshold = None
        self.left = None
        self.right = None


class DecisionTree:
    '''
    A class representing a Decision Tree Classifier, which evaluates split costs
    by using a Gini impurity calculation.
    '''

    def __init__(self, max_depth=None, verbose=False):
        self.max_depth = max_depth
        self.verbose = verbose
        self.total_classes = 0
        self.total_feats = 0
        self.tree = None

    def fit(self, X, y):  # TODO: detect and convert Pandas DataFrames as input to numpy arrays
        '''
        Fits the input data to the model (grows the tree).
        '''
        self.total_classes = len(set(y))
        self.total_feats = X.shape[1]
        if self.max_depth:
            self.tree = self._grow(X, y, depth=0)
        else:
            self.tree = self._grow(X, y)

    def predict(self, X):
        '''
        Predicts all X test samples using the Decision Tree.
        A call to fit() is necessary prior to prediction.
        '''
        preds = []

        for row in X:
            curr = self.tree
            while curr.left:
                if row[curr.feat_index] < curr.split_threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            preds.append(curr.pred_class)
        return preds

    def info(self):  # TODO
        '''
        Print the details of the model.
        '''
        print()

    def _grow(self, X, y, depth=None):
        '''
        Recursively grows this DecisionTree and returns the root Node.
        '''
        if self.verbose:
            if depth is not None:
                print('Current depth: {}'.format(depth))
        node = Node(gini=self._calc_gini(y), num_samples=y.size, pred_class=np.argmax(
            [np.sum(y == c) for c in range(len(set(y)))]))

        if depth and depth >= self.max_depth:
            return node

        gini, split_idx, split_threshold = self._split(X, y)

        if gini < node.gini:  # recursively call _grow for left and right child nodes
            node.feat_index = split_idx
            node.split_threshold = split_threshold

            # left indices, negate for right indices
            left_cond = X[:, split_idx] < split_threshold
            node.left = self._grow(
                X[left_cond], y[left_cond], depth+1 if depth is not None else None)
            node.right = self._grow(
                X[~left_cond], y[~left_cond], depth+1 if depth is not None else None)

        return node

    def _calc_gini(self, y):
        '''
        Calculates the unweighted gini score for the given data.
        '''
        class_counts = [np.sum(y == c) for c in range(len(set(y)))]
        return 1 - np.sum([(count/len(y))**2 for count in class_counts])

    def _split(self, X, y):
        '''
        Calculates the optimal split of the given dataset based on the lowest gini index
        and returns the following:

        - gini: Minimum gini score for the given data (out of all the feats)
        - split_idx: The split index (the column index) of the feature selected as a split
        - split_threshold: The numerical threshold value where the split occurs in the data
        '''
        assert(len(X) == len(y))  # ensure data has same length
        if y.size <= 1:
            return 999999, None, None

        num_feats = X.shape[1]

        gini, split_idx, split_threshold = 999999, None, None
        for feat in range(num_feats):
            vals, labels = zip(*sorted(zip(X[:, feat], y)))
            vals = np.array(vals)
            labels = np.array(labels)

            for index in range(1, len(y)):
                if vals[index-1] == vals[index]:  # avoids attempting a split on identical values
                    continue

                # adjacent mean
                curr_threshold = (vals[index-1] + vals[index]) / 2

                # calculate left and right gini scores
                left = self._calc_gini(labels[:index])
                right = self._calc_gini(labels[index:])

                # calculate total (weighted) gini impurity for left and right gini scores
                curr_gini = ((len(vals[:index]) / y.size)*left) + \
                    ((len(vals[index:]) / y.size)*right)
                if curr_gini < gini:
                    gini, split_idx, split_threshold = curr_gini, feat, curr_threshold

        if self.verbose:
            print("Best Gini: {}\nFeature Index: {}\nThreshold Value: {}\n".format(
                gini, split_idx, split_threshold))
        return gini, split_idx, split_threshold
