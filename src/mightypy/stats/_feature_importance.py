""" 
Feature Importance Methods
-----------------------------

"""

from typing import Callable, Tuple, Optional
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class WOE_IV:
    """
    Weight of Evidence and Information Value.

    References:
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    Args:
        event (str): event name. Generally label true/1.
        non_event (str): non event name. Generally label false/0.
        target_col (str): Target column name.
        value_col (str): Value column name to aggregate(count). Defaults to None.
        bucket_col (str): bucketing column name.
        agg_func (Callable, optional): Aggregation function name. Defaults to np.count_nonzero.
        bucket_col_type (str, optional): Bucketing columns value type. If discrete buckets will not be created else buckets will be created.
                                            Defaults to 'continuous'.
        n_buckets (int, optional): If bucket column has continuous values then create aritificial buckets. Defaults to 10.

    Examples:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from mightypy.stats import WOE_IV

        >>> dataset = load_breast_cancer(as_frame=True)
        >>> df = dataset.frame[['mean radius', 'target']]
        >>> target_map = {0: 'False', 1: 'True'}
        >>> df['label'] = df['target'].map(target_map)

        >>> obj = WOE_IV(event='True', non_event='False', target_col='label',
        >>>              bucket_col='mean radius')

        >>> cal_df, iv = obj.values(df)
        >>> fig = obj.plot()
        >>> fig.tight_layout()
        >>> fig.show()

        or directly

        >>> fig, ax = obj.plot(df)
        >>> fig.show()
    """

    def __init__(
        self,
        event: str,
        non_event: str,
        target_col: str,
        bucket_col: str,
        value_col: Optional[str] = None,
        agg_func: Callable = np.count_nonzero,
        bucket_col_type: str = "continuous",
        n_buckets: int = 10,
    ):
        self._event = event
        self._non_event = non_event
        self._target_col = target_col
        self._bucket_col = bucket_col
        self._bucket_col_name = f"buckets_{bucket_col}"
        self._value_col = value_col
        self._agg_func = agg_func
        self._bucket_col_type = bucket_col_type
        self._n_buckets = n_buckets
        self._perc_event_col_name: str = f"%_event_{event}"
        self._perc_non_event_col_name: str = f"%_non_event_{non_event}"
        self._df: pd.DataFrame = None  # type: ignore
        self._cal_df: pd.DataFrame = None  # type: ignore
        self._iv: float = None  # type: ignore

    def _calculate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Calculations for weight of evidence and information value.

        Args:
            df (pd.DataFrame): input dataframe with said values.

        Raises:
           NotImplementedError: If bucket column type is not 'continuous' or 'discrete'.

        Returns:
            Tuple[pd.DataFrame, float]: calculated dataframe and information value.
        """
        if self._value_col is None:
            self._value_col = "values"
            self._df = df[[self._target_col, self._bucket_col]].copy()
            self._df.insert(loc=0, column=self._value_col, value="x")
        else:
            self._df = df[[self._target_col, self._bucket_col, self._value_col]].copy()

        if self._bucket_col_type == "continuous":
            quantiles = np.linspace(0, 1, self._n_buckets + 1)
            self._df.insert(
                loc=0,
                column=self._bucket_col_name,
                value=pd.qcut(
                    self._df[self._bucket_col].values,
                    q=quantiles,
                    duplicates="raise",
                    retbins=False,
                ),
            )
        elif self._bucket_col_type == "discrete":
            self._df.insert(
                loc=0, column=self._bucket_col_name, value=df[self._bucket_col]
            )
        else:
            raise NotImplementedError

        self._cal_df = pd.pivot_table(
            self._df,
            index=[self._bucket_col_name],
            columns=[self._target_col],
            values=self._value_col,
            aggfunc=self._agg_func,
        )

        self._cal_df.fillna(value=0, inplace=True)

        self._cal_df[["adj_event", "adj_non_event"]] = self._cal_df[
            [self._event, self._non_event]
        ].apply(lambda x: (x + 0.5) if (x[0] == 0 or x[1] == 0) else x, axis=1)

        event_sum = self._cal_df[self._event].sum()
        self._cal_df[self._perc_event_col_name] = self._cal_df["adj_event"] / event_sum

        non_event_sum = self._cal_df[self._non_event].sum()
        self._cal_df[self._perc_non_event_col_name] = (
            self._cal_df["adj_non_event"] / non_event_sum
        )

        self._cal_df["woe"] = np.log(
            self._cal_df[self._perc_non_event_col_name]
            / self._cal_df[self._perc_event_col_name]
        )

        self._cal_df["iv"] = (
            self._cal_df[self._perc_non_event_col_name]
            - self._cal_df[self._perc_event_col_name]
        ) * self._cal_df["woe"]

        self._iv: float = self._cal_df["iv"].sum()
        return self._cal_df, self._iv

    def values(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, float]:
        """
        Returns weight of evidence and information value for given dataframe.

        Args:
            df (Optional[pd.DataFrame], optional): Input dataframe. Defaults to None.

        Raises:
            ValueError: If input dataframe does not exist either in the model or in method
                        input args.

        Returns:
            Tuple[pd.DataFrame, float]: calculated dataframe and information value.
        """
        if self._iv is None or self._cal_df is None:
            if df is None:
                if self._df is None:
                    raise ValueError(
                        "dataframe doesn't exist. Please insert dataframe."
                    )
                else:
                    self._calculate(self._df)
            else:
                self._calculate(df)
        return self._cal_df, self._iv

    def plot(self, df: Optional[pd.DataFrame] = None, figsize=(10, 5)) -> plt.Figure:  # type: ignore
        """
        Plot weight of evidence and subsequent plots.

        Args:
            df (Optional[pd.DataFrame], optional): Input dataframe. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (10, 5).

        Raises:
            ValueError: If dataframe doesn't exist either in the model or in method args.

        Returns:
            plt.Figure: matplotlib figure.
        """

        if self._iv is None or self._cal_df is None:
            if df is None:
                if self._df is None:
                    raise ValueError(
                        "dataframe doesn't exist. Please insert dataframe."
                    )
                else:
                    self._calculate(self._df)
            else:
                self._calculate(df)

        idxs = self._cal_df.index.astype(str)
        ranges = np.arange(0, self._n_buckets, step=1)

        fig, _ax = plt.subplots(1, 2, figsize=figsize)
        _ax[0].set_xlim(
            left=self._cal_df["woe"].min() - 2,  # type: ignore
            right=self._cal_df["woe"].max() + 2,
        )
        _ax[0].barh(
            y=idxs, width=self._cal_df["woe"], color="blue", alpha=0.6  # type: ignore
        )  # type: ignore
        for i in _ax[0].containers:  # type: ignore
            _ax[0].bar_label(i, fmt="%.3f", padding=5)  # type: ignore
        _ax[0].grid(alpha=0.2)  # type: ignore
        _ax[0].set_xlabel(None)  # type: ignore
        _ax[0].set_ylabel(None)  # type: ignore
        _ax[0].set_title("Weight Of Evidence")  # type: ignore

        _ax[1].barh(
            y=ranges - 0.2,
            width=self._cal_df[self._event],  # type: ignore
            color="red",
            alpha=0.6,
            label=self._event,
            height=0.4,
        )
        _ax[1].barh(
            y=ranges + 0.2,
            width=self._cal_df[self._non_event],  # type: ignore
            color="green",
            alpha=0.6,
            label=self._non_event,
            height=0.4,
        )
        _ax[1].set_yticks(ranges)  # type: ignore
        _ax[1].set_yticklabels(idxs)  # type: ignore
        for i in _ax[1].containers:  # type: ignore
            _ax[1].bar_label(i, fmt="%.0f", padding=5)  # type: ignore
        _ax[1].grid(alpha=0.2)  # type: ignore
        _ax[1].set_ylabel(None)  # type: ignore
        _ax[1].set_title("Deciles")  # type: ignore
        _ax[1].legend(bbox_to_anchor=(1.5, 1), loc="upper right")  # type: ignore

        fig.suptitle(
            f"""
                        {self._bucket_col}
                =======================================
                  Information Value  : {self._iv:.3f}
                ---------------------------------------
        """,
            fontsize=12,
        )
        return fig


if __name__ == "__main__":
    # from sklearn.datasets import load_breast_cancer

    # plt.style.use('seaborn')
    # dataset = load_breast_cancer(as_frame=True)
    # df = dataset.frame[['mean radius', 'target']]
    # target_map = {0: 'False', 1: 'True'}
    # df['label'] = df['target'].map(target_map)

    # model = WOE_IV(event='True', non_event='False',
    #                target_col='label', bucket_col='mean radius')

    # fig = model.plot(df)
    # fig.tight_layout()
    # plt.show()

    # print(model.values())

    pass
