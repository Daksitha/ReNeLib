import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

from src.config import VERBOS
from scipy import stats


def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width.
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively.


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin.
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1 / 3)

    if returnas == "width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return result


def prep_histo_data_bins(durations: list, col_name: str):
    x = np.array(durations)
    df = pd.DataFrame({col_name: x})

    # calculate histo bins Freedman–Diaconis rule
    # q25, q75 = np.percentile(x, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    # bins = (x.max() - x.min()) / bin_width
    # bins = stats.iqr(x, rng=(25, 75), scale="raw", nan_policy="omit")
    try:
        bins = freedman_diaconis(data=x, returnas="bins")
    except ValueError as ve:
        if VERBOS:
            print(ve)
            print(f"Data that cause the error {x}")
        bins = 1

    return df, round(bins)


def save_session_histograms(output_folder, label, bc_durations, ss_durations, ls_durations, session_id=0):
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    if bc_durations:
        x_bk, bin_bk = prep_histo_data_bins(bc_durations, col_name='back_channel')
        sns.histplot(data=x_bk, x="back_channel", bins=bin_bk, label=f"bin size: {round(bin_bk, 2)}",
                     color="skyblue", kde=True, ax=axs[0])
        axs[0].legend(loc='upper right')
    else:
        warnings.warn(f"No backchannel data in:  {session_id}")
    if ss_durations:
        x_sp, bin_sp = prep_histo_data_bins(ss_durations, col_name='short_speech')
        sns.histplot(data=x_sp, x="short_speech", bins=bin_sp, color="red", label=f"bin size:{round(bin_sp, 2)}",
                     kde=True, ax=axs[1])
        axs[1].legend(loc='upper right')
    else:
        warnings.warn(f"No short speech data in:  {session_id}")
    if ls_durations:
        x_ls, bin_ls = prep_histo_data_bins(ls_durations, col_name='long_speech')
        sns.histplot(data=x_ls, x="long_speech", bins=bin_ls, color="green", label=f"bin size: {round(bin_ls, 2)}",
                     kde=True, ax=axs[2])
        axs[2].legend(loc='upper right')
    else:
        warnings.warn(f"No long speech data in:  {session_id}")

    img_pth = Path(output_folder) / f"{label}_speech_histograms.png"
    if not img_pth.exists():
        fig.savefig(img_pth)
        if VERBOS:
            print(f"saving histogram image: {img_pth}")
