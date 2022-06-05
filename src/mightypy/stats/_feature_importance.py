from typing import Callable, Tuple, Optional
import numpy as np
import pandas as pd


def woe_and_iv(df: pd.DataFrame, event: str, non_event: str, target_col: str, bucket_col: str,
               value_col: Optional[str] = None, agg_func: Callable = np.count_nonzero, bucket_col_type: str = 'continuous',
               n_buckets: int = 10) -> Tuple[pd.DataFrame, float]:
    """
    Weight of Evidence and Information Value.

    References:
    -----------
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html


    Args:
        df (pd.DataFrame): Input pandas dataframe.
        event (str): event name. Generally label true/1.
        non_event (str): non event name. Generally label false/0.
        target_col (str): Target column name.
        value_col (str): Value column name to aggregate(count). Defaults to None.
        bucket_col (str): bucketing column name.
        agg_func (Callable, optional): Aggregation function name. Defaults to np.count_nonzero.
        bucket_col_type (str, optional): Bucketing columns value type. If discrete buckets will not be created else buckets will be created.
                                            Defaults to 'continuous'.
        n_buckets (int, optional): If bucket column has continuous values then create aritificial buckets. Defaults to 10.

    Raises:
        NotImplementedError: If bucket column type is not 'continuous' or 'discrete'.

    Returns:
        Tuple[ pd.DataFrame, float]: calculated dataframe with weight of evidence, Information Value.

    Examples:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from mightypy.stats import woe_and_iv 
        >>> dataset = load_breast_cancer(as_frame=True)
        >>> df = dataset.frame[['mean radius', 'target']]
        >>> target_map = {0: 'False', 1: 'True'}
        >>> df['label'] = df['target'].map(target_map)
        >>> cal_df, iv = woe_and_iv(df, event='True', non_event='False', target_col='label',
        >>>                         bucket_col='mean radius')
    """
    if value_col is None:
        value_col = 'values'
        df.insert(loc=0, column=value_col, value='x')
    df = df[[target_col, value_col, bucket_col]].copy()
    bucket_col_name: str = f'buckets_{bucket_col}'
    perc_event_col_name: str = f'%_event_{event}'
    perc_non_event_col_name: str = f'%_non_event_{non_event}'

    if bucket_col_type == 'continuous':
        quantiles = np.linspace(0, 1, n_buckets+1)
        df.insert(loc=0, column=bucket_col_name,
                  value=pd.qcut(df[bucket_col].values, q=quantiles, duplicates='raise', retbins=False))
    elif bucket_col_type == 'discrete':
        df.insert(loc=0, column=bucket_col_name, value=df[bucket_col])
    else:
        raise NotImplementedError

    cal_df = pd.pivot_table(df, index=[bucket_col_name], columns=[target_col],
                            values=value_col, aggfunc=agg_func)

    cal_df.fillna(value=0, inplace=True)

    cal_df[['adj_event', 'adj_non_event']] = cal_df[[event,non_event]]\
        .apply(lambda x: (x+0.5) if (x[0] == 0 or x[1] == 0) else x, axis=1)

    event_sum = cal_df[event].sum()
    cal_df[perc_event_col_name] = cal_df['adj_event'] / event_sum

    non_event_sum = cal_df[non_event].sum()
    cal_df[perc_non_event_col_name] = cal_df['adj_non_event'] / non_event_sum

    cal_df['woe'] = np.log(
        cal_df[perc_non_event_col_name] / cal_df[perc_event_col_name]
    )

    cal_df['iv'] = (
        cal_df[perc_non_event_col_name] -
        cal_df[perc_event_col_name]
    ) * cal_df['woe']

    iv: float = cal_df['iv'].sum()
    return cal_df, iv


if __name__ == "__main__":
    pass
