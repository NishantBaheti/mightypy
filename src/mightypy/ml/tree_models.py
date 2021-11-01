from __future__ import annotations
from typing import Union, Tuple, List
import warnings
import numpy as np


class Question:
    """Question is a thershold/matching concept for splitting the node of the Decision Tree

    Args:
        column_index (int): Column index to be chosen from the array passed at the matching time.
        value (Union[int, str, float, np.int64, np.float64]): Threshold value/ matching value.
        header (str): column/header name.
    """

    def __init__(self, column_index: int, value: Union[int, str, float, np.int64, np.float64], header: str):
        """Constructor
        """
        self.column_index = column_index
        self.value = value
        self.header = header

    def match(self, example: Union[list, np.ndarray]) -> bool:
        """Matching function to decide based on example whether result is true or false.

        Args:
            example (Union[list, np.ndarray]): Example to compare with question parameters.
        Returns:
            bool: if the example is in threshold or value matches then results true or false.
        """
        if isinstance(example, list):
            example = np.array(example, dtype="O")
        val = example[self.column_index]

        # adding numpy int and float data types as well
        if isinstance(val, (int, float, np.int64, np.float64)):
            # a condition for question to return True or False for numeric value
            return float(val) >= float(self.value)
        else:
            return str(val) == str(self.value)  # categorical data comparison

    def __repr__(self):
        condition = "=="
        if isinstance(self.value, (int, float, np.int64, np.float64)):
            condition = ">="
        return f"Is {self.header} {condition} {self.value} ?"


class Node:
    """A Tree node either Decision Node or Leaf Node

    Args:
        question (Question, optional): question object. Defaults to None.
        true_branch (Node, optional): connection to node at true side of the branch. Defaults to None.
        false_branch (Node, optional): connection to node at false side of the branch. Defaults to None.
        uncertainty (float, optional): Uncertainty value like gini,entropy,variance etc. Defaults to None.
        leaf_value (Union[dict,int,float], optional): Leaf node/final node's value. Defaults to None.
    """

    def __init__(self, question: Question = None, true_branch: Node = None, false_branch: Node = None, uncertainty: float = None, *, leaf_value: Union[dict, int, float] = None):
        """Constructor
        """
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.uncertainty = uncertainty
        self.leaf_value = leaf_value

    @property
    def _is_leaf_node(self) -> bool:
        """Check if this node is leaf node or not.
        Returns:
            bool: True if leaf node else false.
        """
        return self.leaf_value is not None


class DecisionTreeClassifier:
    """Decision Tree Based Classification Model

    Args:
        max_depth (int, optional): max depth of the tree. Defaults to 100.
        min_samples_split (int, optional): min size of the sample at the time of split. Defaults to 2.
        criteria (str, optional): what criteria to use for information. Defaults to 'gini'. available 'gini','entropy'.
    """

    def __init__(self, max_depth: int = 100, min_samples_split: int = 2, criteria: str = 'gini'):
        """Constructor
        """
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _count_dict(self, a: np.ndarray) -> dict:
        """Count class frequecies and get a dictionary from it

        Args:
            a (np.ndarray): input array. shape should be (m,1) for m samples.

        Returns:
            dict: categories/classes freq dictionary.
        """
        unique_values = np.unique(a, return_counts=True)
        zipped = zip(*unique_values)
        dict_obj = dict(zipped)
        return dict_obj

    def _gini_impurity(self, arr: np.ndarray) -> float:
        """Calculate Gini Impurity

        Args:
            arr (np.ndarray): input array.

        Returns:
            float: gini impurity value.
        """
        classes, counts = np.unique(arr, return_counts=True)
        gini_score = 1 - np.square(counts / arr.shape[0]).sum(axis=0)
        return gini_score

    def _entropy(self, arr: np.ndarray) -> float:
        """Calculate Entropy

        Args:
            arr (np.ndarray): input array.

        Returns:
            float: entropy result.
        """
        classes, counts = np.unique(arr, return_counts=True)
        p = counts / arr.shape[0]
        entropy_score = (-p * np.log2(p)).sum(axis=0)
        return entropy_score

    def _uncertainty(self, a: np.ndarray) -> float:
        """calcualte uncertainty

        Args:
            a (np.ndarray): input array

        Returns:
            float: uncertainty value
        """

        if self.criteria == "entropy":
            value = self._entropy(a)
        elif self.criteria == "gini":
            value = self._gini_impurity(a)
        else:
            warnings.warn(f"{self.criteria} is not coded yet. returning to gini.")
            value = self._gini_impurity(a)
        return value

    def _partition(self, rows: np.ndarray, question: Union[Question, None]) -> Tuple[list, list]:
        """partition the rows based on the question

        Args:
            rows (np.ndarray): input array to split.
            question (Question): question object containing spltting concept.

        Returns:
            Tuple[list,list]: true idxs and false idxs.
        """
        true_idx, false_idx = [], []
        for idx, row in enumerate(rows):
            if question.match(row):
                true_idx.append(idx)
            else:
                false_idx.append(idx)
        return true_idx, false_idx

    def _info_gain(self, left: np.ndarray, right: np.ndarray, parent_uncertainty: float) -> float:
        """Calculate information gain after splitting

        Args:
            left (np.ndarray): left side array.
            right (np.ndarray): right side array.
            parent_uncertainty (float): parent node Uncertainity.

        Returns:
            flaot: information gain value.
        """

        # calculating portion/ partition/ weightage
        pr = left.shape[0] / (left.shape[0] + right.shape[0])

        # calcualte child uncertainity
        child_uncertainty = pr * \
            self._uncertainty(left) - (1 - pr) * self._uncertainty(right)
        # calculate information gain
        info_gain_value = parent_uncertainty - child_uncertainty
        return info_gain_value

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Union[Question, None], float]:
        """method to find best split possible for the sample
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): target matrix.
        
        Returns:
            Tuple[float,Union[Question,None],float]: maximum gain from the split, best question of it, and parent node uncertainty.
        """

        max_gain = -1
        best_split_question = None

        parent_uncertainty = self._uncertainty(y)

        m_samples, n_labels = X.shape

        for col_index in range(n_labels):  # iterate over feature columns
            # get unique values from the feature
            unique_values = np.unique(X[:, col_index])
            for val in unique_values:  # check for every value and find maximum info gain

                ques = Question(
                    column_index=col_index,
                    value=val,
                    header=self._feature_names[col_index]
                )

                t_idx, f_idx = self._partition(X, ques)
                # if it does not split the data
                # skip it
                if len(t_idx) == 0 or len(f_idx) == 0:
                    continue

                true_y = y[t_idx, :]
                false_y = y[f_idx, :]

                # get information gain
                gain = self._info_gain(true_y, false_y, parent_uncertainty)
                if gain > max_gain:
                    max_gain, best_split_question = gain, ques

        return max_gain, best_split_question, parent_uncertainty

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursive funtion to build tree.

        Args:
            X (np.ndarray): input features matrix.   
            y (np.ndarray): target matrix.
            depth (int, optional): depth count of the recursion. Defaults to 0.

        Returns:
            Node: either leaf node or decision node
        """
        m_samples, n_labels = X.shape

        # if depth is greater than max depth defined or labels/features are left to 1
        # or number of samples are less than the minimum size of samples to split then
        # stop recursion and return a node
        if (depth > self.max_depth or n_labels == 1 or m_samples < self.min_samples_split):
            return Node(leaf_value=self._count_dict(y))

        gain, ques, uncertainty = self._find_best_split(X, y)

        # if gain is zero
        # then no point grinding further here
        if gain < 0:
            return Node(leaf_value=self._count_dict(y))

        t_idx, f_idx = self._partition(X, ques)  # get partition idxs
        true_branch = self._build_tree(
            X[t_idx, :], y[t_idx, :], depth + 1)  # recog true branch samples
        false_branch = self._build_tree(
            X[f_idx, :], y[f_idx, :], depth + 1)  # recog false branch samples
        return Node(
            question=ques,
            true_branch=true_branch,
            false_branch=false_branch,
            uncertainty=uncertainty
        )

    def train(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], feature_name: list = None, target_name: list = None) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix.
            y (Union[np.ndarray,list]): target matrix.
            feature_name (list, optional): feature names list. Defaults to None.
            target_name (list, optional): target name list. Defaults to None.
        """

        X = np.array(X, dtype='O') if not isinstance(
            X, (np.ndarray)) else X  # converting to numpy array
        y = np.array(y, dtype='O') if not isinstance(
            y, (np.ndarray)) else y  # converting to numpy array

        # reshaping to vectors
        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # creating feature names if not mentioned
        self._feature_names = feature_name or [
            f"C_{i}" for i in range(self._X.shape[1])]

        # creating target name if not mentioned
        self._target_name = target_name or ['target']

        # BOOOM
        # building the tree
        self._tree = self._build_tree(
            X=self._X,
            y=self._y
        )

    def print_tree(self, node: Union[Node, None] = None, spacing: str = "|-") -> None:
        """print the tree

        Args:
            node (Union[Node,None], optional): starting node. Defaults to None. then it will go to the root node of the tree.
            spacing (str, optional): printing separater. Defaults to "|-".
        """

        node = node or self._tree

        if node._is_leaf_node:
            print(spacing, " Predict :", node.leaf_value)
            return

        # Print the question at this node
        print(spacing + str(node.question) +
              " | " + self.criteria + " :" + str(node.uncertainty))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, "  " + spacing + "-")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, "  " + spacing + "-")

    def _classification(self, row: np.ndarray, node: Union[Node, None]) -> Union[dict]:
        """Classification recursive function

        Args:
            row (np.ndarray): input matrix.
            node (Union[Node,None]): node to start with. mostly root node. rest will be handled by recursion.

        Returns:
            Union[dict]: leaf value. classification result.
        """
        if node._is_leaf_node:
            return node.leaf_value

        if node.question.match(row):
            return self._classification(row, node.true_branch)
        else:
            return self._classification(row, node.false_branch)

    def _leaf_probabilities(self, results: dict) -> dict:
        """get probabilties

        Args:
            results (dict): results from _classification.

        Returns:
            dict: dictionary with categorical probabilities.
        """
        total = sum(results.values())
        probs = {}
        for key in results:
            probs[key] = (results[key] / total) * 100
        return probs

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """predict classification results

        Args:
            X (Union[np.ndarray,list]): testing matrix.

        Raises:
            ValueError: input X can only be a list or numpy array.

        Returns:
            np.ndarray: results of classification.
        """
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                max_result = 0
                result_dict = self._classification(row=X, node=self._tree)
                result = None
                for key in result_dict:
                    if result_dict[key] > max_result:
                        result = key
                return np.array([[result]], dtype='O')
            else:
                leaf_value = []
                # get maximum caterigorical value from all catergories
                for row in X:
                    max_result = 0
                    result_dict = self._classification(row=row, node=self._tree)
                    result = None
                    for key in result_dict:
                        if result_dict[key] > max_result:
                            result = key
                    leaf_value.append([result])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")

    def predict_probability(self, X: Union[np.ndarray, list]) -> Union[np.ndarray, dict]:
        """predict classfication probabilities

        Args:
            X (Union[np.ndarray,list]): testing matrix.

        Raises:
            ValueError: input X can only be a list or numpy array.

        Returns:
            Union[np.ndarray, dict]: probabity results of classification.
        """
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                return self._leaf_probabilities(self._classification(row=X, node=self._tree))
            else:
                leaf_value = []
                for row in X:
                    leaf_value.append([self._leaf_probabilities(
                        self._classification(row=row, node=self._tree))])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")


class DecisionTreeRegressor:
    """Decision Tree Based Regression Model

    Args:
        max_depth (int, optional): maximum depth of the tree. Defaults to 10.
        min_samples_split (int, optional): minimum number of samples while splitting. Defaults to 3.
        criteria (str, optional): criteria for best info gain. Defaults to 'variance'.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 3, criteria: str = 'variance'):
        """constructor
        """
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _mean_leaf_value(self, a: np.ndarray) -> float:
        """leaf values mean

        Args:
            a (np.ndarray): input array.

        Returns:
            float: mean value
        """

        return float(np.mean(a))

    def _partition(self, rows: np.ndarray, question: Union[Question, None]) -> Tuple[list, list]:
        """partition the rows based on the question

        Args:
            rows (np.ndarray): input array to split.
            question (Question): question object containing spltting concept.

        Returns:
            Tuple[list,list]: true idxs and false idxs.
        """
        true_idx, false_idx = [], []
        for idx, row in enumerate(rows):
            if question.match(row):
                true_idx.append(idx)
            else:
                false_idx.append(idx)
        return true_idx, false_idx

    def _uncertainty(self, a: np.ndarray) -> float:
        """calcualte uncertainty

        Args:
            a (np.ndarray): input array

        Returns:
            float: uncertainty value
        """
        if self.criteria == "variance":
            value = np.var(a)
        else:
            warnings.warn(f"{self.criteria} is not coded yet. returning to variance.")
            value = np.var(a)
        return float(value)

    def _info_gain(self, left: np.ndarray, right: np.ndarray, parent_uncertainty: float) -> float:
        """Calculate information gain after splitting

        Args:
            left (np.ndarray): left side array.
            right (np.ndarray): right side array.
            parent_uncertainty (float): parent node Uncertainity.

        Returns:
            flaot: information gain value.
        """

        pr = left.shape[0] / (left.shape[0] + right.shape[0])

        child_uncertainty = pr * \
            self._uncertainty(left) - (1 - pr) * self._uncertainty(right)
        info_gain_value = parent_uncertainty - child_uncertainty
        return info_gain_value

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, Union[Question, None], float]:
        """method to find best split possible for the sample

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): target matrix.

        Returns:
            Tuple[float,Union[Question,None],float]: maximum gain from the split, best question of it, and parent node uncertainty
        """

        max_gain = -1
        best_split_question = None

        parent_uncertainty = self._uncertainty(y)

        m_samples, n_labels = X.shape

        for col_index in range(n_labels):  # iterate over feature columns
            # get unique values from the feature
            unique_values = np.unique(X[:, col_index])
            for val in unique_values:  # check for every value and find maximum info gain

                ques = Question(
                    column_index=col_index,
                    value=val,
                    header=self._feature_names[col_index]
                )

                t_idx, f_idx = self._partition(X, ques)
                # if it does not split the data
                # skip it
                if len(t_idx) == 0 or len(f_idx) == 0:
                    continue

                true_y = y[t_idx, :]
                false_y = y[f_idx, :]

                gain = self._info_gain(true_y, false_y, parent_uncertainty)
                if gain > max_gain:
                    max_gain, best_split_question = gain, ques

        return max_gain, best_split_question, parent_uncertainty

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursive funtion to build tree

        Args:
            X (np.ndarray): input features matrix.   
            y (np.ndarray): target matrix.
            depth (int, optional): depth count of the recursion. Defaults to 0.

        Returns:
            Node: either leaf node or decision node
        """
        m_samples, n_labels = X.shape

        # if depth is greater than max depth defined or labels/features are left to 1
        # or number of samples are less than the minimum size of samples to split then
        # stop recursion and return a node
        if (depth > self.max_depth or n_labels == 1 or m_samples < self.min_samples_split):
            return Node(leaf_value=y)

        gain, ques, uncertainty = self._find_best_split(X, y)

        # if gain is zero no point in going further
        if gain < 0:
            return Node(leaf_value=y)

        t_idx, f_idx = self._partition(X, ques)
        true_branch = self._build_tree(
            X[t_idx, :], y[t_idx, :], depth + 1)  # get true samples
        false_branch = self._build_tree(
            X[f_idx, :], y[f_idx, :], depth + 1)  # get false samples
        return Node(
            question=ques,
            true_branch=true_branch,
            false_branch=false_branch,
            uncertainty=uncertainty
        )

    def train(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], feature_name: list = None, target_name: list = None) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix.
            y (Union[np.ndarray,list]): target matrix.
            feature_name (list, optional): feature names list. Defaults to None.
            target_name (list, optional): target name list. Defaults to None.
        """

        X = np.array(X, dtype='O') if not isinstance(
            X, (np.ndarray)) else X  # converting to numpy array
        y = np.array(y, dtype='O') if not isinstance(
            y, (np.ndarray)) else y  # converting to numpy array

        # reshaping to vectors
        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # creating feature names if not mentioned
        self._feature_names = feature_name or [
            f"C_{i}" for i in range(self._X.shape[1])]

        # creating target name if not mentioned
        self._target_name = target_name or ['target']

        # BOOOM
        # building the tree
        self._tree = self._build_tree(
            X=self._X,
            y=self._y
        )

    def print_tree(self, node: Union[Node, None] = None, spacing: str = "|-", mean_preds: bool = True) -> None:
        """print the tree

        Args:
            node (Union[Node,None], optional): starting node. Defaults to None. then it will go to the root node of the tree.
            spacing (str, optional): printing separater. Defaults to "|-".
            mean_preds (bool): do the mean of prediction values. Defaults to True.
        """

        node = node or self._tree

        if node._is_leaf_node:
            if mean_preds:
                print(spacing, " Predict :", self._mean_leaf_value(node.leaf_value))
            else:
                print(spacing, " Predict :", node.leaf_value[...,-1])
            return

        # Print the question at this node
        print(spacing + str(node.question) +
              " | " + self.criteria + " :" + str(node.uncertainty))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, "  " + spacing + "-", mean_preds)

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, "  " + spacing + "-", mean_preds)

    def _regression(self, row: np.ndarray, node: Union[Node, None], mean_preds: bool) -> float:
        """regression recursive method

        Args:
            row (np.ndarray): input matrix.
            node (Union[Node,None]): node to start with. mostly root node. rest will be handled by recursion.
            mean_preds (bool): do the mean of prediction values.
            
        Returns:
            float: regression result.
        """

        if node._is_leaf_node:
            if mean_preds:
                return self._mean_leaf_value(node.leaf_value)
            else:
                return node.leaf_value[...,-1]

        if node.question.match(row):
            return self._regression(row, node.true_branch, mean_preds)
        else:
            return self._regression(row, node.false_branch, mean_preds)

    def predict(self, X: np.ndarray, mean_preds: bool = True) -> np.ndarray:
        """predict regresssion
        Args:
            X (np.ndarray): testing matrix.
            mean_preds (bool): do the mean of prediction values. Defaults to True.
        Raises:
            ValueError: X should be list or numpy array
        Returns:
            np.ndarray: regression prediction.
        """
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            if len(X.shape) == 1:
                result = self._regression(row=X, node=self._tree, mean_preds=mean_preds)
                return np.array([[result]], dtype='O')
            else:
                leaf_value = []
                for row in X:
                    result = self._regression(row=row, node=self._tree, mean_preds=mean_preds)
                    leaf_value.append([result])
                return np.array(leaf_value, dtype='O')
        else:
            raise ValueError("X should be list or numpy array")


