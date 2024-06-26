from jude.stat_maker import gen_stats
import pandas as pd
import pickle
import numpy as np
from config import LTA_PATH, OUT_PATH, HEADER, T_COL, S_COL
from config import S_MIN, S_MAX, RS_WINDOWS, DATA_START, DATA_END


def get_time_and_signal(
        lta,
        header,
        t_col,
        s_col,
        s_min,
        s_max,
        d_start,
        d_stop):
    """ Method to get time and one signal from an lta file.

    Args:
        lta (str): Full path to LTA file
        header (int): Number of rows in the LTA before the header
        t_col (str): String name of the time column.
        s_col (str): String name of the relevant signal column.
        s_min (float): Minimum signal to include.
        s_max (float): Maximum signal to include.
        d_start (int): Index of first data point to include in
            the processed data. This is useful if you want to
            discard data before a certain point (e.g., if the
            first few minutes of your data correspond to an
            irrelevant warmup process).
        d_stop (int): Index of the last data point to include in
            the processed data.
    Return:
        time, sig (np.array, np.array): arrays with the times and
            signal (units depend on the data).
    """
    df = pd.read_csv(lta, sep='\t', header=header)
    cols = df.columns.to_list()
    for col in cols:
        if col not in [t_col, s_col]:
            df = df.drop(col, axis=1)
    df = df.dropna()
    df = df[s_min < df[s_col]]
    df = df[df[s_col] < s_max]
    time = df[t_col].values
    sig = df[s_col].values
    return time[d_start: d_stop], sig[d_start: d_stop]


if __name__ == '__main__':
    time, wlns = get_time_and_signal(
        LTA_PATH,
        HEADER,
        T_COL,
        S_COL,
        S_MIN,
        S_MAX,
        DATA_START,
        DATA_END)
    dt = np.mean(np.diff(time)) / 1000  # Mean difference between points (s)
    windows = np.array(RS_WINDOWS)
    windows = (windows / dt).astype(int)
    bins = np.linspace(min(wlns), max(wlns), 300)
    data = gen_stats(wlns, rs_windows=windows, hi_windows=None, bins='auto')
    data['time'] = time
    data['wlns'] = wlns
    with open(OUT_PATH, 'wb+') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
