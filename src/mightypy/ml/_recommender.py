"""
Recommender Systems
--------------------
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

_TQDM_BAR_FORMAT = "{desc:<5.5} : {percentage:3.0f}%| {bar:30} {r_bar}"


class ALS:
    """Alternating Least Squares"""

    def __init__(self, dim_factors, n_iter, lambda_=1.0) -> None:
        self._lambda = lambda_
        self._dim_factors = dim_factors
        self._n_iter = n_iter
        # self._user_ids = None
        # self._item_ids = None
        # self._user_idx_map = None
        # self._idx_user_map = None
        # self._item_idx_map = None
        # self._idx_item_map = None
        # self._mask = None
        # self._iter_losses = None
        # self._user_emb = None
        # self._item_emb = None

    def _loss(
        self,
        ratings: np.ndarray,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        mask: np.ndarray,
    ):
        # use by mask M to ignore NaNs
        return np.sqrt(np.square(ratings - (user_emb @ item_emb), where=mask).sum())

    def data_preparation(
        self, dataframe: pd.DataFrame, user_col: str, item_col: str, score_col: str
    ):
        self._user_col = user_col
        self._item_col = item_col
        self.score_col = score_col
        matrix_df = dataframe.pivot_table(
            index=user_col, columns=item_col, values=score_col, aggfunc="sum"
        ).astype("float32")

        self._user_ids = list(matrix_df.index)
        self._item_ids = list(matrix_df.columns)
        self._user_idx_map = dict(zip(self._user_ids, range(len(self._user_ids))))
        self._idx_user_map = dict(zip(range(len(self._user_ids)), self._user_ids))
        self._item_idx_map = dict(zip(self._item_ids, range(len(self._item_ids))))
        self._idx_item_map = dict(zip(range(len(self._item_ids)), self._item_ids))

        ratings = matrix_df.values
        return ratings

    def _fit_user_emb(self, user_idxs: int, ratings: np.ndarray):
        user_idxs = (
            [user_idxs] if not isinstance(user_idxs, (list, tuple)) else user_idxs
        )
        for user_idx in user_idxs:
            mask_idx = self._mask[user_idx]
            self._user_emb[user_idx] = np.array(
                (self._item_emb[:, mask_idx] @ ratings[user_idx, mask_idx]).T
                @ np.linalg.inv(
                    (self._item_emb @ self._item_emb.T)
                    + (self._lambda * np.eye(self._dim_factors))
                ),
            )

    def _fit_item_emb(self, item_idxs: int, ratings: np.ndarray):
        item_idxs = (
            [item_idxs] if not isinstance(item_idxs, (list, tuple)) else item_idxs
        )
        for item_idx in item_idxs:
            mask_idx = self._mask[:, item_idx]
            self._item_emb[:, item_idx] = np.array(
                (self._user_emb.T[:, mask_idx] @ ratings[mask_idx, item_idx])
                @ np.linalg.inv(
                    (self._user_emb.T @ self._user_emb)
                    + (self._lambda * np.eye(self._dim_factors))
                ),
            )

    def fit(
        self,
        dataframe: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        score_col: str = "ratings",
    ):
        ratings = self.data_preparation(dataframe, user_col, item_col, score_col)

        n_users, n_items = ratings.shape

        self._user_emb = np.random.rand(n_users, self._dim_factors) * 1e-3
        self._item_emb = np.random.rand(self._dim_factors, n_items) * 1e-3

        self._mask = ~np.isnan(ratings)

        self._iter_losses = []

        pbar = tqdm(total=self._n_iter, desc="Training ", bar_format=_TQDM_BAR_FORMAT)

        for iter_ in range(self._n_iter):

            user_pbar = tqdm(total=n_users, desc="Users ", bar_format=_TQDM_BAR_FORMAT)
            for user_idx in range(n_users):
                self._fit_user_emb(user_idx, ratings)
                user_pbar.update(1)

            item_pbar = tqdm(total=n_items, desc="Items ", bar_format=_TQDM_BAR_FORMAT)
            for item_idx in range(n_items):
                self._fit_item_emb(item_idx, ratings)
                item_pbar.update(1)

            loss = self._loss(ratings, self._user_emb, self._item_emb, self._mask)

            print("Loss over iteration : ", iter_, loss)
            self._iter_losses.append(loss)

            pbar.update(1)
        return self._iter_losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = pd.read_csv("/workspaces/mightypy/datasets/ratings.csv", sep="\t")

    model = ALS(dim_factors=500, n_iter=10, lambda_=1.0)
    losses = model.fit(
        dataframe=df,
        user_col="userId",
        item_col="movieId",
        score_col="rating"
    )

    plt.plot(losses)
    plt.savefig("/workspaces/mightypy/plots/loss_plot.png")
