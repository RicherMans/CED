from typing import Dict, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import multiprocessing
from functools import partial
from h5py import File
from tqdm import tqdm
import soundfile as sf


def read_wav_soundfile(fname: str,
                       mono: bool = True,
                       dtype: str = 'int16') -> Tuple[np.ndarray, int]:
    y, sr = sf.read(fname, dtype=dtype)
    # Use one channel for mono storing, if input is stereo
    if y.ndim > 1 and mono:
        y = y.mean(1)
    return y.astype(dtype), sr


def proxy_read(params, mono: bool, dtype: str):
    filename, split = params
    wav, sr = read_wav_soundfile(filename, mono=mono, dtype=dtype)
    return split, filename, wav, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filelist', type=Path)
    parser.add_argument('outputdir', type=Path)
    parser.add_argument('-s', '--size_per_file', type=int, default=50000)
    parser.add_argument('-n', '--n_workers', type=int, default=4)
    parser.add_argument(
        '--duration',
        default=False,
        action='store_true',
        help="Append a column 'duration' for each respective sample")
    parser.add_argument('--hdf5_dirname', default='hdf5', type=Path)
    parser.add_argument('--index_dirname', default='labels', type=Path)
    parser.add_argument(
        '--filename_column',
        default='filename',
        type=str,
        help="The column name that identifies the files to extract")
    parser.add_argument(
        '--fullname',
        action='store_true',
        default=False,
        help=
        "Stores either fullname keys in hdf5 (entire path) or just each files individual name"
    )
    parser.add_argument(
        '--stereo',
        action='store_true',
        help="If or not to sum over multiple channels if input is multi-channel"
    )
    parser.add_argument('-d', '--delim', default='\t', type=str)
    parser.add_argument('--dtype', default='int16', type=str)
    parser.set_defaults(stereo=False)
    args = parser.parse_args()

    df: pd.DataFrame = pd.read_csv(args.input_filelist, sep=args.delim)
    assert f"{args.filename_column}" in df.columns, f"The column '{args.filename_column}' needs to be in the header"
    unique_fnames = df[args.filename_column].unique()
    idxs_per_sample = np.arange(len(unique_fnames)) // args.size_per_file
    hdf5_file_ids = np.unique(idxs_per_sample)

    hdf5_base_path = args.outputdir / args.hdf5_dirname
    labelidx_base_path = args.outputdir / args.index_dirname
    hdf5_base_path.mkdir(parents=True, exist_ok=True)
    labelidx_base_path.mkdir(parents=True, exist_ok=True)

    output_file_prefix = args.input_filelist.stem
    hdf5_files = [
        File(hdf5_base_path / f'{output_file_prefix}_{idx}.h5', 'w')
        for idx in hdf5_file_ids
    ]

    fname_to_hdf5path: Dict[str, str] = {}
    fname_to_duration: Dict[str, float] = {}

    with multiprocessing.Pool(processes=args.n_workers) as pool:
        for return_values in tqdm(pool.imap_unordered(
                partial(proxy_read, mono=not args.stereo, dtype=args.dtype),
                zip(unique_fnames, idxs_per_sample)),
                                  total=len(unique_fnames),
                                  desc='Dumping to HDF5',
                                  unit='file'):
            split_id, filename, wav, sr = return_values
            hdf5_file = hdf5_files[split_id]
            fname_to_hdf5path[filename] = hdf5_file.filename
            if args.duration:
                fname_to_duration[filename] = float(wav.shape[-1] / sr)
            if not args.fullname:
                filename = Path(filename).name
            hdf5_file[str(filename)] = wav

    [f.close() for f in hdf5_files]
    if args.duration:
        df['duration'] = df['filename'].map(fname_to_duration)
    df['hdf5path'] = df['filename'].map(fname_to_hdf5path)

    df.to_csv(labelidx_base_path / f"{output_file_prefix}.tsv",
              sep=args.delim,
              float_format='%.4f',
              index=False)

    print(f"Finished, final data can be found at {args.outputdir}")

if __name__ == "__main__":
    main()
