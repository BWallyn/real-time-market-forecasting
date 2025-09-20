# =================
# ==== IMPORTS ====
# =================

import logging

import numpy as np
import pandas as pd

# Options
logger = logging.getLogger(__name__)


# ===================
# ==== FUNCTIONS ====
# ===================


def reduce_memory_usage(df: pd.DataFrame, float16_as32: bool = True) -> pd.DataFrame:
    """Reduce the memory usage of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        float16_as32 (bool): Whether to convert float 16 as float 32.

    Returns:
        df (pd.DataFrame): Output DataFrame with memory usage reduced.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    # Loop over columns
    for col in df.columns:
        col_type = df[col].dtype
        if col_type is not object and str(col_type) != 'category':
            # The column is not a string column
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(
        'Memory usage after optimization is: {:.2f} MB'\
        'Decreased by {:.1f}%'.format(end_mem, 100 * (start_mem - end_mem) / start_mem)
    )
    return df
