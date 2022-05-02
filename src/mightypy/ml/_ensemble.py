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

    def __init__(self, num_of_trees: int = 25, min_features: Optional[int] = None, max_depth: int = 50, min_samples_split: int = 2, criteria: str = 'gini') -> None:
        """constructor
        """
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

    def _sampling(self) -> Tuple[List[int], List[int]]:
        """sampling function

        Returns:
            Tuple[List[int],List[int]]: sampling idxs for rows nad columns for feature and target matrix.
        """
        m, n = self._X.shape

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

    def train(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], feature_name: list = None, target_name: list = None) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix
            y (Union[np.ndarray,list]): target matrix
            feature_name (str, optional): feature names. Defaults to None.
            target_name (str, optional): target names. Defaults to None.
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

        for _ in range(self.num_of_trees):
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criteria=self.criteria
            )
            idxs, feat_idxs = self._sampling()
            X_sampled = self._X[idxs, :][:, feat_idxs]
            y_sampled = self._y[idxs]

            clf.train(
                X=X_sampled,
                y=y_sampled,
                feature_name=[self._feature_names[i] for i in feat_idxs],
                target_name=self._target_name
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
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            results = []
            for clf, feat_idxs in self._trees:
                result = clf.predict(X[:, feat_idxs])
                results.append(result)

            all_tree_results = np.concatenate(np.array(results, dtype='O'), axis=1)
            final_results = np.apply_along_axis(
                func1d=self._get_max_result, axis=1, arr=all_tree_results).reshape(-1, 1)
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

    def __init__(self, num_of_trees: int = 25, min_features: int = None, max_depth: int = 30, min_samples_split: int = 3, criteria: str = 'variance') -> None:
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

    def _sampling(self) -> Tuple[List[int], List[int]]:
        """sampling function

        Returns:
            Tuple[List[int],List[int]]: sampling idxs for rows nad columns for feature and target matrix.
        """
        m, n = self._X.shape

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

    def train(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list], feature_name: list = None, target_name: list = None) -> None:
        """Train the model

        Args:
            X (Union[np.ndarray,list]): feature matrix.
            y (Union[np.ndarray,list]): target matrix.
            feature_name (list, optional): feature names. Defaults to None.
            target_name (list, optional): target name. Defaults to None.
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

        for _ in range(self.num_of_trees):

            reg = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criteria=self.criteria
            )
            idxs, feat_idxs = self._sampling()  # get sampling idxs
            X_sampled = self._X[idxs, :][:, feat_idxs]
            y_sampled = self._y[idxs]

            reg.train(
                X=X_sampled,
                y=y_sampled,
                feature_name=[self._feature_names[i] for i in feat_idxs],
                target_name=self._target_name
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
            X = np.array(X, dtype='O') if not isinstance(X, (np.ndarray)) else X

            results = []
            for reg, feat_idxs in self._trees:
                result = reg.predict(X[:, feat_idxs])
                results.append(result)

            all_tree_results = np.concatenate(np.array(results, dtype='O'), axis=1)
            final_results = np.mean(all_tree_results, axis=1).reshape(-1, 1)
            return final_results
        else:
            raise ValueError("X should be list or numpy array")


class AdaboostClassifier:
    """
    Adaboost Classification Model.
    
    It is still under construction. DO NOT USE IT.

    .. _Adaboost: https://machinelearningexploration.readthedocs.io/en/latest/EnsembleMethods/ExploreBoosting.html#Adaboost-Classfication
    """
    def __init__(self, n_stumps:int, stump_depth:int=0):
        print("DO not use this method. It is not tested.")
        self.stump_depth = stump_depth
        self._X= None
        self._y= None
        self._feature_names = None
        self._stumps:list = []
        self.n_stumps = n_stumps

    def amount_of_say(self, total_err:Union[np.ndarray, float, int])->Union[np.ndarray, float, int]:
        """
        Amount of say.

        .. math::
            \\text{amount of say} = \\frac{1}{2}(log(\\frac{1 - \\text{TE}}{\\text{TE}}))

        Args:
            total_err (Union[np.ndarray, float, int]): Total error from tree.

        Returns:
            Union[np.ndarray, float, int]: amount of say.
        """
        return np.log((1 - total_err)/ total_err) / 2.0

    def normalize(self, x:np.ndarray)->np.ndarray:
        """
        Nornmalization.

        Args:
            x (np.ndarray): input array.

        Returns:
            np.ndarray: normalized values.
        """
        return x / x.sum()

    def _update_sample_weight(self, sample_weights:Union[np.ndarray,float], aos:Union[int,float], is_correct:bool)->np.ndarray:
        """
        Update sample weight for new tree.

        Args:
            sample_weights (Union[np.ndarray,float]): sample weight.
            aos (Union[int,float]): amount of say.
            is_correct (bool): is correctly classified records.

        Returns:
            Union[np.ndarray,float]: New updated weights based on amount of say.
        """
        if is_correct: # correctly classified records
            return sample_weights * np.exp(aos)
        else:
            return sample_weights * np.exp(-aos)

    def _get_sample_idxs(self, size:int, probs:Union[list,np.ndarray])->np.ndarray:
        """
        get indexes based on probabilities from a range.

        Args:
            size (int): size of array.
            probs (Union[list,np.ndarray]): probabities of indexes.

        Returns:
            np.ndarray: sampling indexes.
        """
        idxs = np.random.choice(range(size),size=(size,), p=probs)
        return idxs 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X (np.ndarray): input array.

        Returns:
            np.ndarray: predictions.
        """
        results = np.ones((X.shape[0], 1)) * self.leaf_value
        for stump in self._stumps:
            results = results + (stump['aos'] * stump['model'].predict(X))

        return results

    def train(self, X:np.ndarray, y:np.ndarray, feature_names:list=None, target_name:list=None):
        """
        Train the model.

        Args:
            X (np.ndarray): input features.
            y (np.ndarray): target.
            feature_names (list, optional): feature names. Defaults to None.
            target_name (list, optional): target name. Defaults to None.
        """
        X = np.array(X, dtype='O') if not isinstance(
            X, (np.ndarray)) else X  # converting to numpy array
        y = np.array(y, dtype='O') if not isinstance(
            y, (np.ndarray)) else y  # converting to numpy array
        # reshaping to vectors
        self._X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        self._y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        # creating feature names if not mentioned
        self._feature_names = feature_names or [
            f"C_{i}" for i in range(self._X.shape[1])]

        # creating target name if not mentioned
        self._target_name = target_name or ['target']

        self.leaf_value = 1 / X.shape[0]
        sample_weights = np.ones((X.shape[0], 1)) * self.leaf_value

        sample_X = self._X
        sample_y = self._y
        self._n_samples = sample_X.shape[0]
        aos = -1

        for i_stump in range(self.n_stumps):
            print(i_stump," number of stump")
            
            # building a decision tree stump
            stump = DecisionTreeClassifier(max_depth=self.stump_depth)
            stump.train(
                X=sample_X,
                y=sample_y,
                feature_name=self._feature_names,
                target_name=self._target_name
            )

            stump_preds = self.predict(sample_X) + stump.predict(sample_X)
            
            total_err = ((stump_preds != sample_y) * sample_weights).sum()

            if total_err <= 0:
                print("early stopping as total error is <= 0", total_err)
                break

            aos = self.amount_of_say(total_err)
            
            if aos <= 0.0:
                print("early stopping as amount of say is <= 0", aos)
                break

            # storing in bag of stumps
            self._stumps.append({
                "idx": i_stump,
                "model": stump,
                "aos": aos
            })

            # preparation for next stump
            wrong_class_weights = (stump_preds != sample_y) * sample_weights
            right_class_weights = (stump_preds == sample_y) * sample_weights


            new_wrong_class_weights = self._update_sample_weight(
                sample_weights=wrong_class_weights,
                aos = aos,
                is_correct= False
            )

            new_right_class_weights = self._update_sample_weight(
                sample_weights=right_class_weights,
                aos = aos,
                is_correct=True
            )
            
            # new sample weights
            new_sample_weights = new_right_class_weights + new_wrong_class_weights
            sample_weights = self.normalize(new_sample_weights)

            # new samples
            sample_idxs = self._get_sample_idxs(size=self._n_samples,probs=sample_weights[...,-1])
            sample_X = sample_X[sample_idxs]
            sample_y = sample_y[sample_idxs]


    
    

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X,y = make_classification(n_classes=2,n_features=4,n_samples=100)

    model = AdaboostClassifier(n_stumps=100)
    model.train(
        X=X,
        y=y
    )
    print(model.predict(X))
