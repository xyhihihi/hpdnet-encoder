import scipy.io as sio
import numpy as np

for i in range(1, 13):
    try:
        w = sio.loadmat(f'd:/pythonProject/HPDNet-fpn/tmp/customed/w_{i}.mat')[f'w_{i}']
        print(f'w_{i}: shape={w.shape}, dtype={w.dtype}')
    except Exception as e:
        print(f'w_{i}: error - {e}')

# w_1: shape=(64, 56), dtype=complex128
# w_2: shape=(56, 49), dtype=complex128
# w_3: shape=(49, 42), dtype=complex128
# w_4: shape=(42, 36), dtype=complex128
# w_5: shape=(36, 30), dtype=complex128
# w_6: shape=(30, 25), dtype=complex128
# w_7: shape=(25, 20), dtype=complex128
# w_8: shape=(20, 16), dtype=complex128
# w_9: shape=(16, 12), dtype=complex128
# w_10: shape=(12, 9), dtype=complex128
# w_11: shape=(9, 6), dtype=complex128
# w_12: shape=(6, 4), dtype=complex128
