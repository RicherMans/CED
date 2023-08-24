from typing import Optional, List
from einops import rearrange, repeat
import torchaudio
import torch
from torchaudio.transforms._transforms import _AxisMasking
from torchaudio.functional.functional import _get_mask_param
import utils


def mask_along_axis(
    specgram,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
    values: Optional[torch.Tensor] = None,
    min_values: Optional[torch.Tensor] = None,
):
    r"""Apply a mask along ``axis``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgrams.size(axis) * p))``
    otherwise.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram `(channel, freq, time)`
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrogram of dimensions `(channel, freq, time)`
    """

    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    if values is not None and min_values is not None:
        value = values.to(specgram.device) * mask_param
        min_value = min_values.to(
            specgram.device) * (specgram.size(axis) - value)
    else:
        value = torch.rand(1) * mask_param
        min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(0,
                        specgram.shape[axis],
                        device=specgram.device,
                        dtype=specgram.dtype)
    # Pad dims adds one dimensional paddings for 3d case (Spec is B, T, F) and 2 dims for 4d case (Spec is B, C, T, F).
    pad_dims = ' '.join(['1'] * (specgram.ndim - 2))
    if axis == 2:
        mask = repeat(mask, f'... -> b {pad_dims} ...', b=specgram.shape[0])
    else:
        mask = repeat(mask, f'... -> b ... {pad_dims}', b=specgram.shape[0])
    mask_start = rearrange(mask_start, f'... -> ... 1 1')
    mask_end = rearrange(mask_end, f'... -> ... 1 1')
    mask = (mask >= mask_start) & (mask < mask_end)

    if any(mask_end - mask_start >= mask_param):
        raise ValueError(
            "Number of columns to be masked should be less than mask_param")

    specgram = specgram.masked_fill(mask, mask_value)

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram


class TimeMasking(_AxisMasking):

    def __init__(self,
                 time_mask_param: int,
                 iid_masks: bool = False,
                 p: float = 1.0) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks, p=p)

    def forward(self,
                specgram,
                values: Optional[torch.Tensor] = None,
                min_values: Optional[torch.Tensor] = None,
                mask_value: float = 0.0) -> torch.Tensor:
        return mask_along_axis(specgram,
                               self.mask_param,
                               mask_value,
                               self.axis,
                               p=self.p,
                               values=values,
                               min_values=min_values)


class FrequencyMasking(_AxisMasking):

    def __init__(self, freq_mask_param: int, iid_masks: bool = False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)

    def forward(self,
                specgram,
                values: Optional[torch.Tensor] = None,
                min_values: Optional[torch.Tensor] = None,
                mask_value: float = 0.0) -> torch.Tensor:
        return mask_along_axis(specgram,
                               self.mask_param,
                               mask_value,
                               self.axis,
                               p=self.p,
                               values=values,
                               min_values=min_values)


if __name__ == '__main__':
    x = torch.rand(8, 10, 1000)
    m = TimeMasking(10)
    seeds = torch.arange(len(x))
    f = FrequencyMasking(2)
    print(f"{torch.where(f(x, )==0)}.shape")
    print(f"{torch.where(f(x, seeds=seeds)==0)}.shape")
    print(f"{torch.where(f(x, seeds=seeds)==0)}.shape")
    print(f"{torch.where(f(x, seeds=seeds)==0)}.shape")
