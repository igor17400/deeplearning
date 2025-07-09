# %%
import numpy as np
from numpy import ndarray

# %%
def conv2d(input: ndarray, kernel: ndarray, padding='valid',stride=1):
    """
    Performs a 2D convolution operation.
    Args:
        input (np.array): The input 2D array (ex: image)
        kernel (np.array): The 2D convolution kernel (filter)
        padding (str):  'valid' (no padding) or
                        'same' (pads to keep output size same as input).
            Note: 'same' padding here assumes a stride of 1 for simplicity.
    Returns:
        np.array: The convolved 2D array
    """
    # Get dimensions
    in_height, in_width = input.shape
    krnl_height, krnl_width = kernel.shape
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Calculate padding
    padded_in = input # In case padding == 'valid' or something else
    if padding == 'same':
        pad_h = (krnl_height - 1) // 2
        pad_w = (krnl_width - 1) // 2
        padded_in = np.pad(input, ((pad_h, pad_h), (pad_w, pad_w)   ), mode='constant')

    # Get dimensions of the padded input:
    padded_in_height, padded_in_width = padded_in.shape

    # Calculate output dimensions
    out_height = (padded_in_height - krnl_height) // stride + 1
    out_width = (padded_in_width - krnl_width) // stride + 1

    # Initialize the output matrix
    output = np.zeros((out_height, out_width))
    # Convolution operation
    for y in range(0, padded_in_height - krnl_height + 1, stride):
        for x in range(0, padded_in_width - krnl_width + 1, stride):
            # Extract the input patch
            in_patch = padded_in[y:y + krnl_height, x:x + krnl_width]
            # Perform element-wise multiplication and sum
            output[y // stride, x // stride] = np.sum(in_patch * kernel)
    return output

# %%
input = np.array([[0, 1, 2, 3, 1], [0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
print(f"Convolved output: \n {conv2d(input, kernel)}")
