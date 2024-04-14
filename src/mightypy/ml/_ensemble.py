"""
Ensemble methods for Machine Learning
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional
import numpy as np
from mightypy.ml._tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """Ensemble method for classification

    using a bunch of Decision Tree's to do to the classification.

    Args:
        num_of_trees (int, optional): number of trees in ensemble. Defaults to 50.
        min_features (int, optional): minimum number of features to use in every tree. Defaults to None.
        max_depth (int, optional): max depth of the every tree. Defaults to 100.
        min_samples_split (int, optional): minimum size ofsampels to split. Defaults to 2.
        criteria (str, optional): criteria to calcualte information gain. Defaults to 'gini'.
    """

    def __init__(
        self,
        num_of_trees: int = 25,
        min_features: Optional[int] = None,
        max_depth: int = 50,
        min_samples_split: int = 2,
        criteria: str = "gini",
    ) -> None:
        """constructor"""
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._trees = []
        self.num_of_trees = num_of_trees
        self.min_features = min_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """sampling function

        Returns:
            Tuple[np.ndarray, np.ndarray]: sampling idxs for rows nad columns for feature and target matrix.
        """
        m, n = self._X.shape  # type: ignore

        # sampling with replacement
        # means rows with repeat in the data
        # statitistically it gives an edge for data prediction
        idxs = np.random.randint(low=0, high=m, size=m)

        # feature sampling to decrease correlation between trees
        if self.min_features is None:
            size = n
        else:
            size = n if self.min_features > n else self.min_features

        feat_idxs = np.random.choice(n, size=size, replace=False)
        return idxs, feat_idxs

    def train(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        feature_name: Optional[list] = None,
        target_name: Optional[list] = None,
    ) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix
            y (Union[np.ndarray,list]): target matrix
            feature_name (str, optional): feature names. Defaults to None.
            target_name (str, optional): target names. Defaults to None.
        """

        X = (
            np.array(X, dtype="O") if not isinstance(X, (np.ndarray)) else X
        )  # converting to numpy array
        y = (
            np.array(y, dtype="O") if not isinstance(y, (np.ndarray)) else y
        )  # converting to numpy array
        # reshaping to vectors
        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # creating feature names if not mentioned
        self._feature_names = feature_name or [
            f"C_{i}" for i in range(self._X.shape[1])
        ]

        # creating target name if not mentioned
        self._target_name = target_name or ["target"]

        for _ in range(self.num_of_trees):
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criteria=self.criteria,
            )
            idxs, feat_idxs = self._sampling()
            X_sampled = self._X[idxs, :][:, feat_idxs]
            y_sampled = self._y[idxs]

            clf.train(
                X=X_sampled,
                y=y_sampled,
                feature_name=[self._feature_names[i] for i in feat_idxs],
                target_name=self._target_name,
            )

            self._trees.append([clf, feat_idxs])

    @staticmethod
    def _get_max_result(a: np.ndarray) -> Union[str, int, None]:
        """get max result from the bunch of classification results

        Args:
            a (np.ndarray): input array for category

        Returns:
            Union[str,int,None]: max count class/category
        """
        unique_values = np.unique(a, return_counts=True)
        zipped = zip(*unique_values)
        max_count = 0
        result = None
        for i in zipped:
            if i[1] > max_count:
                result = i[0]
                max_count = i[1]
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict results

        Args:
            X (np.ndarray): test matrix.

        Raises:
            ValueError: X should be list or numpy array.

        Returns:
            np.ndarray: prediction results.
        """

        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype="O") if not isinstance(X, (np.ndarray)) else X

            results = []
            for clf, feat_idxs in self._trees:
                result = clf.predict(X[:, feat_idxs])
                results.append(result)

            all_tree_results = np.concatenate(np.array(results, dtype="O"), axis=1)
            final_results = np.apply_along_axis(
                func1d=self._get_max_result, axis=1, arr=all_tree_results
            ).reshape(
                -1, 1
            )  # type: ignore
            return final_results
        else:
            raise ValueError("X should be list or numpy array")


class RandomForestRegressor:
    """Ensemble method for regression

    using a bunch of Decision Tree's to do to the regression.

    Args:
        num_of_trees (int, optional): number of trees in ensemble. Defaults to 50.
        min_features (int, optional): minimum number of features to use in every tree. Defaults to None.
        max_depth (int, optional): max depth of the every tree. Defaults to 100.
        min_samples_split (int, optional): minimum size ofsampels to split. Defaults to 2.
        criteria (str, optional): criteria to calcualte information gain. Defaults to 'gini'.
    """

    def __init__(
        self,
        num_of_trees: int = 25,
        min_features: Optional[int] = None,
        max_depth: int = 30,
        min_samples_split: int = 3,
        criteria: str = "variance",
    ) -> None:
        self._X = None
        self._y = None
        self._feature_names = None
        self._target_name = None
        self._trees = []
        self.num_of_trees = num_of_trees
        self.min_features = min_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criteria = criteria

    def _sampling(self) -> Tuple[np.ndarray, np.ndarray]:
        """sampling function

        Returns:
            Tuple[np.ndarray, np.ndarray]: sampling idxs for rows nad columns for feature and target matrix.
        """
        m, n = self._X.shape  # type: ignore

        # sampling with replacement
        # means rows with repeat in the data
        # statitistically it gives an edge for data prediction
        idxs = np.random.randint(low=0, high=m, size=m)

        # feature sampling to decrease correlation between trees
        if self.min_features is None:
            size = n
        else:
            size = n if self.min_features > n else self.min_features

        feat_idxs = np.random.choice(n, size=size, replace=False)
        return idxs, feat_idxs

    def train(
        self,
        X: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        feature_name: Optional[list] = None,
        target_name: Optional[list] = None,
    ) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix.
            y (Union[np.ndarray,list]): target matrix.
            feature_name (list, optional): feature names. Defaults to None.
            target_name (list, optional): target name. Defaults to None.
        """

        X = (
            np.array(X, dtype="O") if not isinstance(X, (np.ndarray)) else X
        )  # converting to numpy array
        y = (
            np.array(y, dtype="O") if not isinstance(y, (np.ndarray)) else y
        )  # converting to numpy array
        # reshaping to vectors
        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # creating feature names if not mentioned
        self._feature_names = feature_name or [
            f"C_{i}" for i in range(self._X.shape[1])
        ]

        # creating target name if not mentioned
        self._target_name = target_name or ["target"]

        for _ in range(self.num_of_trees):
            reg = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criteria=self.criteria,
            )
            idxs, feat_idxs = self._sampling()  # get sampling idxs
            X_sampled = self._X[idxs, :][:, feat_idxs]
            y_sampled = self._y[idxs]

            reg.train(
                X=X_sampled,
                y=y_sampled,
                feature_name=[self._feature_names[i] for i in feat_idxs],
                target_name=self._target_name,
            )

            self._trees.append([reg, feat_idxs])

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """predict regression result

        Args:
            X (Union[np.ndarray, list]): test matrix.

        Raises:
            ValueError: X should be list or numpy array.

        Returns:
            np.ndarray: regression results.
        """
        if isinstance(X, (np.ndarray, list)):
            X = np.array(X, dtype="O") if not isinstance(X, (np.ndarray)) else X

            results = []
            for reg, feat_idxs in self._trees:
                result = reg.predict(X[:, feat_idxs])
                results.append(result)

            all_tree_results = np.concatenate(np.array(results, dtype="O"), axis=1)
            final_results = np.mean(all_tree_results, axis=1).reshape(-1, 1)
            return final_results
        else:
            raise ValueError("X should be list or numpy array")
