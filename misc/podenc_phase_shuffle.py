import numpy as np 


def phase_shuffle(input_signal):
    # % Returns a vector of the same size and amplitude spectrum but with shuffled
    # % phase information. 

    input_signal = input_signal.reshape(-1, 1)

    N = len(input_signal)
    if  N % 2:
        h = input_signal[-1] 
        input_signal = input_signal[:-1] 

    F = np.fft.fft(input_signal)
    r = abs(F)
    t = np.zeros_like(F)
    t[0] = 0
    t[1: math.floor(N / 2)] = np.random.rand(math.floor(N / 2) - 1, 1)*2*math.pi - math.pi
    t[(floor(N/2) + 1):] = -t[floor(N/2):1:-1]


    output_signal = np.fft.ifft(r * exp(1j * t))

    if N % 2:
        output_signal.append(h)

    return output_signal


if __name__ == '__main__':
    input_vec = np.random.rand(10)
    output_vec = phase_shuffle(input_vec)
