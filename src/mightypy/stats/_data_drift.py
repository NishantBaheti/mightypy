"""
data drift module
==================

"""
from typing import Union
import numpy as np
import pandas as pd


def population_stability_index(expected: Union[list, np.ndarray], actual: Union[list, np.ndarray], data_type: str) -> pd.DataFrame:
    """
    Populaion Stability Index.

    References:
        https://www.listendata.com/2015/05/population-stability-index.html

    Args:
        expected (Union[list, np.ndarray]): Expected values.
        actual (Union[list, np.ndarray]): Actual values.
        data_type (str): Type of data. Helps in bucketing.

    Returns:
        pd.DataFrame: calculated dataframe.

    Examples:
        >>> import numpy as np
        >>> from mightypy.stats import population_stability_index
        
        continuous data
            >>> expected_continuous = np.random.normal(size=(500,))
            >>> actual_continuous = np.random.normal(size=(500,))
            >>> psi_df = population_stability_index(expected_continuous, actual_continuous, data_type='continuous')
            >>> psi_df.psi.sum()

        discrete data
            >>> expected_discrete = np.random.randint(0,10, size=(500,))
            >>> actual_discrete = np.random.randint(0,10, size=(500,))
            >>> psi_df = population_stability_index(expected_discrete, actual_discrete, data_type='discrete')
            >>> psi_df.psi.sum()
    """
    if data_type == 'continuous':
        max_val: Union[int, float] = np.max(expected)
        min_val: Union[int, float] = np.min(expected)

        ranges = np.linspace(min_val, max_val, 11)[1:-1]
        bins = [-np.inf, *ranges, np.inf]
        labels = [
            f"{idx+1} | {i[0]:.2f} to {i[1]:.2f}" for idx, i in enumerate(zip(bins[:-1], bins[1:]))
        ]
        expected_cuts = pd.cut(expected, bins=bins, labels=labels).value_counts()
        actual_cuts = pd.cut(actual, bins=bins, labels=labels).value_counts()

    elif data_type == 'discrete':
        expected_cuts = pd.Series(expected).value_counts()
        actual_cuts = pd.Series(actual).value_counts()

    else:
        raise NotImplementedError(f"Method {data_type} is not implemented, or correct one. Try continuous, discrete.")

    calc_df = pd.concat([expected_cuts, actual_cuts], axis=1,
                        keys=['expected', 'actual']).sort_index()
    calc_df[['expected %', 'actual %']] = (
        calc_df[['expected', 'actual']]/calc_df[['expected', 'actual']].sum(axis=0))
    calc_df['diff'] = calc_df['actual %'] - calc_df['expected %']
    calc_df['log(actual %/ expected %)'] = np.log(calc_df['actual %'] /
                                                  calc_df['expected %'])
    calc_df['psi'] = calc_df['diff'] * calc_df['log(actual %/ expected %)']
    return calc_df


if __name__ == "__main__":

    expected_continuous = np.random.normal(size=(500,))
    actual_continuous = np.random.normal(size=(500,))

    expected_discrete = np.random.randint(0,10, size=(500,))
    actual_discrete = np.random.randint(0,10, size=(500,))

    print(population_stability_index(expected_continuous, actual_continuous, data_type='continuous'))

    print(population_stability_index(expected_discrete, actual_discrete, data_type='discrete'))
