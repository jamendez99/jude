import numpy as np


def mean_offset(data):
    """ Get mean-offset data.

    Args:
        data (np.array): an array-like of data to be processed.

    Return:
        mean, offset: the mean of the data is returned, as well
            as an np.array of the data minus the mean.
    """
    avg = np.mean(data)
    return avg, data - avg


def smooth(data, window):
    """ Get rolling-window smoothed data.

    Args:
        data (np.array): an array-like of data to be processed.
        window (int): size of the window to do the smoothing. This
            is the number of adjacent data point that will be
            averaged together to get a single number.
    Return:
        smooth_data: np.array of smoothed data. This array will be
            smaller than the input array. It's length will be
            len(data) - window + 1
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')
