from pathlib import Path
from loguru import logger
import multiprocessing as mp
import math
import re
import numpy as np
from typing import Callable, Optional, Sequence, Tuple
import pandas as pd
from h5py import File
import utils
import torch


class LogitsReader(torch.utils.data.Dataset):

    def __init__(
            self,
            data_frame: pd.DataFrame,
            logits_basepath: Path,
            sample_rate: int = 16000,
            label_type: str = 'last_mean',
            num_classes: int = 527,
            wavtransforms: Optional[Callable] = None,
            max_epochs: int = 0,  # Default infer from data
    ) -> None:
        super().__init__()
        self.fname_to_hdf5 = data_frame.set_index(
            'filename')['hdf5path'].to_dict()
        self.basepath = logits_basepath
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        if max_epochs == 0:
            for p in self.basepath.glob(f'*ep*parquet'):
                match_regx = re.search(r'ep(\d+)', p.stem)
                if match_regx:
                    max_epochs = max(int(match_regx.group(1)), max_epochs)
            if max_epochs == 0:
                raise ValueError("Did not find any data")
        self.max_epochs = max_epochs
        logger.info(
            f"Using a maximum of {self.max_epochs} epochs for augmentation")
        self.hdfs = dict()
        # Setting to the start of epochs,
        self.wavtransforms = wavtransforms
        self._datasetcache = dict()
        if label_type == 'last_mean':
            self.target_transform = self._labeltransform_lm
        if label_type == 'last_half':
            self.target_transform = self._labeltransform_half
        if label_type == 'last_random':
            self.target_transform = self._labeltransform_lastrand
        if label_type == '527':
            self.target_transform = self._labeltransform_527
        elif label_type == 'zero':
            self.target_transform = self._labeltransform_zero
        # At last
        self.epoch = mp.Value('i', 1)
        self._set_epoch(1)

    def _labeltransform_lm(self, idxs: torch.Tensor,
                           vals: torch.Tensor) -> torch.Tensor:

        # Last-mean transform, i.e., calculate the mean over the left over probability after topk
        fill_val = (vals.min() / (self.num_classes - vals.shape[-1])).item()
        # print(f"{vals.shape=}, {self.num_classes=} {fill_val=} {vals.dtype=}")
        targets = torch.full((self.num_classes, ),
                             dtype=vals.dtype,
                             fill_value=fill_val)
        targets.scatter_(0, idxs, vals)
        return targets

    def _labeltransform_527(self, idxs: torch.Tensor,
                            vals: torch.Tensor) -> torch.Tensor:

        # Last-mean transform, i.e., calculate the mean over the left over probability after topk
        targets = torch.zeros(self.num_classes) + 1. / 527
        # print(f"{vals.shape=}, {self.num_classes=} {fill_val=} {vals.dtype=}")
        targets.scatter_(0, idxs, vals)
        return targets

    def _labeltransform_lastrand(self, idxs: torch.Tensor,
                                 vals: torch.Tensor) -> torch.Tensor:

        # Last-mean transform, i.e., calculate the mean over the left over probability after topk
        targets = torch.rand(self.num_classes) * vals.min()
        # print(f"{vals.shape=}, {self.num_classes=} {fill_val=} {vals.dtype=}")
        targets.scatter_(0, idxs, vals)
        return targets

    def _labeltransform_half(self, idxs: torch.Tensor,
                             vals: torch.Tensor) -> torch.Tensor:

        # Last-mean transform, i.e., calculate the mean over the left over probability after topk
        fill_val = (vals.min() / 2.0).item()
        # print(f"{vals.shape=}, {self.num_classes=} {fill_val=} {vals.dtype=}")
        targets = torch.full((self.num_classes, ),
                             dtype=vals.dtype,
                             fill_value=fill_val)
        targets.scatter_(0, idxs, vals)
        return targets

    def _labeltransform_zero(self, idxs: torch.Tensor,
                             vals: torch.Tensor) -> torch.Tensor:

        targets = torch.zeros((self.num_classes, ), dtype=vals.dtype)
        targets.scatter_(0, idxs, vals)
        return targets

    def _readdata(self,
                  hdf5path: str,
                  fname: str,
                  start: Optional[int] = None,
                  end: Optional[int] = None) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')

        data = self._datasetcache[hdf5path][f"{fname}"][start:end]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __readdf__(self):
        dfs = []
        epoch_value = self.epoch.value
        assert epoch_value is not None, "Need to first call .set_epoch(EPOCH)"
        #Glob the filepaths within the basedir, find .h5 files and corresponding .parquet
        hdf5paths, parquet_paths = [], []
        for f in self.basepath.glob(f'*ep{epoch_value}_*h5'):
            parquet_paths.append(str(f).replace('.h5', '.parquet'))
            hdf5paths.append(str(f))
        for i in range(len(parquet_paths)):
            hdf5path = hdf5paths[i]
            parquet_file = parquet_paths[i]
            df = pd.read_parquet(parquet_file, use_nullable_dtypes=True)
            df['start'] = df['start']
            df['end'] = df['end']
            df['hdf5path'] = hdf5path
            # Need the index column for loading data
            df = df.reset_index()
            dfs.append(df)
        self.logits_df = pd.concat(dfs, axis=0,
                                   keys=hdf5paths).reset_index(drop=True)
        self.hdfs = {hdf5path: File(hdf5path, 'r') for hdf5path in hdf5paths}

    def _set_epoch(self, epoch: int):
        # In cases that a user uses a higher epoch, we simply return to the beginning
        self.epoch.value = max(epoch % self.max_epochs, 1)  # Minimum is 1
        # Reset hdf5path files
        if self.hdfs is not None:
            for openfile in self.hdfs.values():
                openfile.close()
        self.logits_df = None
        self.hdfs = None
        self.__readdf__()

    def __getitem__(self, index):
        item = self.logits_df.iloc[index]
        index_of_value = item['index']
        hdf5path_logits = item['hdf5path']
        start = round(item['start'] * self.sample_rate)
        end = round(item['end'] * self.sample_rate)
        fname = item['filename']
        hdf5path_data = self.fname_to_hdf5[fname]
        data = self._readdata(hdf5path_data, fname, start, end)

        label_store = self.hdfs[hdf5path_logits]
        indexes = label_store['topk_indexes']
        values = label_store['topk_values']
        seeds = label_store['seeds']
        target_indx, target_val, spec_seed = map(
            torch.as_tensor,
            (indexes[index_of_value, :], values[index_of_value, :],
             seeds[index_of_value]))
        #Placeholder
        spec_aug_value = None
        spec_aug_min_value = None
        if self.wavtransforms is not None:
            with utils.FixSeedContext(seed=spec_seed):
                data = self.wavtransforms(data.view(1, 1, -1)).view(-1)
                # Pretty bad code here, but just ... no other way, we need to pass these variables to specaug ........
                spec_aug_value = torch.rand(1)
                spec_aug_min_value = torch.rand(1)
        target = self.target_transform(target_indx.long(), target_val.float())
        return data, target, spec_aug_value, spec_aug_min_value, fname

    def __len__(self):
        return len(self.logits_df)


class SequentialHDF5DataLoader(torch.utils.data.IterableDataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        wavtransforms: Callable,
        sample_rate: int = 16000,
        chunk_length: float = 10.0,
    ):
        super().__init__()
        self._datasetcache = {}
        self._dataframe = data_frame
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.chunk_length_in_samples = int(self.chunk_length * sample_rate)
        self.wavtransforms = wavtransforms

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][:]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __iter__(self):
        for index in range(len(self._dataframe)):
            fname, hdf5path = self._dataframe.iloc[index][[
                'filename', 'hdf5path'
            ]]
            data = self._readdata(hdf5path, fname)
            for i, chunk in enumerate(
                    data.split(self.chunk_length_in_samples, -1)):
                if chunk.shape[-1] < self.chunk_length_in_samples // 2:
                    break
                start_in_secs = i * self.chunk_length
                end_in_secs = min((i + 1) * self.chunk_length,
                                  data.shape[-1] / self.sample_rate)
                seed = np.random.randint(0, 1 << 31)
                spec_aug_value = None
                spec_aug_min_value = None
                with utils.FixSeedContext(seed=seed):
                    chunk = self.wavtransforms(chunk.view(1, 1, -1)).view(-1)
                    # WORST CODE HERE, BUT NO OTHER OPTION NOW, we need to pass these variables to specaug ........
                    spec_aug_value = torch.rand(1)
                    spec_aug_min_value = torch.rand(1)
                    # print(f"{fname=}, {spec_aug_min_value=}, {spec_aug_value=} {seed=}")
                yield chunk, start_in_secs, end_in_secs, fname, seed, spec_aug_value, spec_aug_min_value

    def __len__(self) -> int:
        return int(10. / self.chunk_length) * len(self._dataframe)


class HDF5DataLoader(torch.utils.data.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
    ):
        super().__init__()
        self._datasetcache = {}
        self._dataframe = data_frame

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][:]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        fname, hdf5path = self._dataframe.iloc[index][['filename', 'hdf5path']]
        data = self._readdata(hdf5path, fname)
        return data, fname

    def __len__(self):
        return len(self._dataframe)


class WeakHDF5Dataset(torch.utils.data.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        num_classes: int,
    ):
        super(WeakHDF5Dataset, self).__init__()
        self._dataframe = data_frame
        self._datasetcache = {}
        self._len = len(self._dataframe)
        self._num_classes = num_classes

    def __len__(self) -> int:
        return self._len

    def __del__(self):
        for k, cache in self._datasetcache.items():
            cache.close()

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][:]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return torch.as_tensor(data, dtype=torch.float32)

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, hdf5path = self._dataframe.iloc[index][[
            'filename', 'labels', 'hdf5path'
        ]]
        # Init with all-Zeros classes i.e., nothing present
        target = torch.zeros(self._num_classes, dtype=torch.float32).scatter_(
            0, torch.as_tensor(label_idxs), 1)
        data = self._readdata(hdf5path, fname)
        return data, target, fname


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = 0
    overall_end = len(dataset._dataframe)
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil(
            (overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    worker_start = overall_start + worker_id * per_worker
    worker_end = min(worker_start + per_worker, overall_end)
    dataset._dataframe = dataset._dataframe.iloc[worker_start:worker_end].copy(
    )


def pad(tensorlist: Sequence[torch.Tensor], padding_value: float = 0.):
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim, ) + trailing_dims + (num_raw_samples, )
    out_tensor = torch.full(out_dims,
                            fill_value=padding_value,
                            device=tensorlist[0].device,
                            dtype=torch.float32)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor


def sequential_pad(batches):
    datas, targets, *other = zip(*batches)
    return pad(datas), torch.stack(targets), *other


def sequential_pad_logits(batches):
    datas, *other = zip(*batches)
    return pad(datas), *other


if __name__ == "__main__":
    import pandas as pd
    for data in LogitsReader(pd.read_csv('data/balanced_train/labels/balanced.tsv',sep='\t'),'logits/ensemble5014/balanced/chunk_10/'):
        print(data)
