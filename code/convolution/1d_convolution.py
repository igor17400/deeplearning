# %%
import numpy as np
from numpy.typing import NDArray

# %%
def conv1d(signal: NDArray, kernel: NDArray, stride:int=1, padding:int=0):
    """
    Perform 1D convolution on a signal with a given kernel.

    Parameters:
    - signal (np.ndarray): Input signal.
    - kernal (np.ndarray): Convolution kernel.
    - stride (int): Stride for the convolution operation.
    - padding (int): Padding size for the signal.

    Returns:
    - np.ndarray: Result of the convolution operation.
    """
    # 1) Folding: Flip the kernal
    kernel = np.flip(kernel)

    # Pad the signal (optional step depending on padding value)
    padded_signal = np.pad(signal, (padding, padding), mode='constant', constant_values=0)

    # Calculate the output length
    output_length = (len(padded_signal) - len(kernel)) // stride + 1
    result = np.zeros(output_length)

    for i in range(output_length): # Integral
        # Step 2: Displacement - move the kernel along the signal
        start = i * stride
        end = start + len(kernel)

        # This is the displaced window of the signal
        window = padded_signal[start: end]

        # Step 3: Multiplication - element-wise
        multiplied = window * kernel

        # Step 4: Integration - sum all values
        result[i] = np.sum(multiplied)

    return result

# %%
signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, 0, -1])
print("Convolved output:", conv1d(signal, kernel, stride=1, padding=1))
