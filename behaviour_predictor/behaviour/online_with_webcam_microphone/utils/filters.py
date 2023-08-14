import numpy as np
import sys
import time

def bilateral_filter(outputs):
    import cv2
    """ smoothing function

    function that applies bilateral filtering along temporal dim of sequence.
    TODO: Fix this for live streaming
    """
    outputs_smooth = np.zeros(outputs.shape)
    for b in range(outputs.shape[0]):
        for f in range(outputs.shape[2]):
            smoothed = np.reshape(cv2.bilateralFilter(
                outputs[b, :, f], 5, 20, 20), (-1))
            outputs_smooth[b, :, f] = smoothed
    return outputs_smooth.astype(np.float32)


def time_hns():
    """Return time in 100 nanoseconds (more precise with >= python 3.7)"""

    # Python 3.7 or newer use nanoseconds
    if (sys.version_info.major == 3 and sys.version_info.minor >= 7) or sys.version_info.major >= 4:
        # 100 nanoseconds / 0.1 microseconds
        time_now = int(time.time_ns() / 100)
    else:
        # timestamp = int(time.time() * 1000)
        # timestamp = timestamp.to_bytes((timestamp.bit_length() + 7) // 8, byteorder='big')

        # match 100 nanoseconds / 0.1 microseconds
        time_now = int(time.time() * 10000000)  # time.time()
        # time_now = time.time()

    return time_now