import numpy as np
import bottleneck as bn


CALCS = ['bs', 'rs', 'hi']


def basic_stats(data):
    """ Get basic statistics on one-dimensional data.

    Args:
        data (np.array): an array-like of data to be processed.

    Return:
        A dictionary containing the following entries:
        - 'avg': the average of the data
        - 'std': the standard deviation of the data
        - 'max': maximum of the data
        - 'min': minimum of the data
        - 'med': median of the data
    """
    bs = dict()
    bs['avg'] = np.mean(data)
    bs['std'] = np.std(data)
    bs['max'] = np.max(data)
    bs['min'] = np.min(data)
    bs['med'] = np.median(data)
    return bs


def rolling_stats(data, windows):
    """ Get rollinwg window statistics on one-dimensional data.

    Args:
        data (np.array): an array-like of data to be processed.

    Return:
        A dictionary mapping a window size to a dictionary of
        rolling statistics. Each statistics dictionary will
        contain the following.
        - 'avg': rolling average of the data
        - 'std': rolling standard deviation of the data
        - 'max': rolling maximum of the data
        - 'min': rolling minimum of the data
        - 'med': rolling median of the data
    """
    rs = dict()
    for w in windows:
        rs[w] = dict()
        rs[w]['avg'] = bn.move_mean(data, w)[w - 1:]
        rs[w]['std'] = bn.move_std(data, w)[w - 1:]
        rs[w]['max'] = bn.move_max(data, w)[w - 1:]
        rs[w]['min'] = bn.move_min(data, w)[w - 1:]
        rs[w]['med'] = bn.move_median(data, w)[w - 1:]
    return rs


def histograms(data, windows=None, bins='auto'):
    if windows is None:
        hist, b = np.histogram(data, bins=bins, density=True)
        return {'bins': b, 'hist': hist}
    else:
        pass  # WIP
        assert not isinstance(bins, str), \
            'Cannot make rolling histogram with auto generated bins.'
        assert hasattr(bins, "__len__"), \
            '"bins" must be a sequence of scalars.'
        res = dict()
        for window in windows:
            hist, b = np.histogram(data, bins=bins, density=True)
            res[window] = {'bins': b, 'hist': hist}
        return res


def gen_stats(data, windows=[2], calcs=CALCS):
    stats = dict()
    if 'bs' in calcs:
        stats['bs'] = basic_stats(data)
    if 'rs' in calcs:
        stats['rs'] = rolling_stats(data, windows)
    return stats
