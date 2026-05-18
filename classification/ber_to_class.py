import numpy as np

def ber_to_class(ber_values, lower_thres, upper_thres, logBER=False):
    if logBER:
        eps = 1e-15  # To avoid log(0)
        ber_values = np.log10(np.clip(ber_values, eps, None)).astype(np.float32)

    # Class mapping:
    # 0 -> BER < lower_thres
    # 1 -> lower_thres <= BER <= upper_thres
    # 2 -> BER > upper_thres
    labels = np.ones_like(ber_values, dtype=np.int64)
    labels[ber_values < lower_thres] = 0
    labels[ber_values > upper_thres] = 2
    return labels