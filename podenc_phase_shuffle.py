import math

import numpy as np


def phase_shuffle(input_signal):
    # Returns a vector of the same size and amplitude spectrum but with shuffled
    # phase information.

    N = input_signal.size
    if N % 2:
        h = input_signal[-1]
        input_signal = input_signal[:-1]

    F = abs(np.fft.fft(input_signal))
    t = np.zeros(F.shape)
    t[1:math.
      floor(N /
            2)] = np.random.rand(math.floor(N / 2) - 1) * 2 * math.pi - math.pi
    t[(math.floor(N / 2) + 1):] = -t[math.floor(N / 2) - 1:0:-1]

    output_signal = abs(np.fft.ifft(F * np.exp(1j * t)))

    if N % 2:
        output_signal.append(h)

    return output_signal


if __name__ == '__main__':
    input_vec = np.arange(1, 101)
    output_vec = phase_shuffle(input_vec)
