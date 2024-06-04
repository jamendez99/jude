# from pynufft import NUFFT
import numpy as np


def nufft(times, signal):
    # TODO: implement someone's version of a Non-Uniform FFT
    pass


def fft(times, signal):
    """ Calculate the FFT of the given signal.

    This method assumes that the input signal is quasi-evenly
    spaced. This is the case with many experimentally acquired
    values. This method simply wraps the fft methods in numpy.

    Args:
        times (np.array): a list of timestamps for the data.
        signal (np.array): the signal to be Fourier transformed.

    Return:
        fft, freqs: the fourier transformed function and the
            corresponding frequencies. The units of the returned
            frequencies will be the inverse of the units of the
            input frequencies.
    """
    signal_fft = np.fft.fft(signal)
    n = signal_fft.size
    dt = np.mean(np.diff(times))
    freqs = np.fft.fftfreq(n, d=dt)
    return signal_fft, freqs
