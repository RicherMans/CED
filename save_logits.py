from collections import defaultdict
import tempfile
from torch.cuda.amp import autocast
import os
from typing import List, Union
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from packaging import version
from h5py import File
from fire import Fire
import ignite.distributed as idist
import utils
import dataset
import models

SR = 16000
infer_mode = torch.inference_mode if version.parse(
    torch.__version__) > version.parse('1.8.2') else torch.no_grad

torch.backends.cudnn.deterministic = True


def create_model_proxy(model_name: Union[str, List], model_args):
    if isinstance(model_name, list):
        mdls = [
            getattr(models, mdl_name)(pretrained=True, **model_args) for mdl_name in model_name
        ]
        return models.EnsembleModel(mdls)
    else:
        return getattr(models, model_name)(pretrained=True, **model_args)


def forward_spec_amp(model, data):
    with autocast(enabled=True):
        return model.forward_spectrogram(data).to(torch.float32)


def forward_spec(model, data):
    return model.forward_spectrogram(data)


def _save_logits_parallel(rank, parquet_files, output_dir, wavtransforms,
                          spectransforms, config_parameters):
    model = create_model_proxy(config_parameters.model,
                               config_parameters.model_args)
    device = idist.device()
    logger.debug(f"Starting inference on {device=}")
    model = model.to(device).eval()

    if idist.get_world_size() > 1 and rank == 0:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

    dataloader = idist.auto_dataloader(
        dataset.SequentialHDF5DataLoader(
            pd.read_parquet(parquet_files[rank]),
            chunk_length=config_parameters.chunk_length,
            wavtransforms=wavtransforms,
        ),
        num_workers=config_parameters.num_workers,
        collate_fn=dataset.sequential_pad_logits,
        worker_init_fn=dataset.
        worker_init_fn,  # Required, otherwise workers have same data
        persistent_workers=
        True,  # super important otherwise worker_init_fn is called again, will take forever
        batch_size=config_parameters.batch_size)

    forward_function = forward_spec_amp if config_parameters.mode == 'amp' else forward_spec

    for epoch in tqdm(range(1, config_parameters.epochs + 1),
                      desc='Epoch',
                      disable=(rank != 0)):
        output_hdf5 = output_dir / f"logits_ep{epoch}_{rank}.h5"
        output_parquet = output_dir / f"logits_ep{epoch}_{rank}.parquet"
        indexed_data = defaultdict(list)
        c = 0
        with File(output_hdf5, 'w') as store:
            vals = store.create_dataset('topk_values',
                                        dtype='float16',
                                        maxshape=(None,
                                                  config_parameters.topk),
                                        shape=(config_parameters.batch_size,
                                               config_parameters.topk))
            idxs = store.create_dataset('topk_indexes',
                                        dtype='int16',
                                        maxshape=(None,
                                                  config_parameters.topk),
                                        shape=(config_parameters.batch_size,
                                               config_parameters.topk))
            seeds = store.create_dataset(
                'seeds',
                dtype='int32',
                maxshape=(None, ),
                shape=(config_parameters.batch_size, ))
            for data, starts, ends, fnames, seed, specaug_vals, specaug_minvals in tqdm(
                    dataloader, leave=False, desc='sample', disable=(rank
                                                                     != 0)):
                data = data.to(device)
                with torch.inference_mode():
                    data = model.front_end(data)
                    data = spectransforms(
                        data,
                        values=torch.tensor(specaug_vals),
                        min_values=torch.tensor(specaug_minvals))
                    y = forward_function(model, data).to('cpu')
                    # y = model.forward_spectrogram(data).to('cpu')
                    y_vals, y_idxs = y.topk(config_parameters.topk)
                    if c >= vals.shape[0]:
                        vals.resize(
                            (c + data.shape[0], config_parameters.topk))
                        idxs.resize(
                            (c + data.shape[0], config_parameters.topk))
                        seeds.resize((c + data.shape[0], ))
                    vals[c:c + y.shape[0], :] = y_vals.numpy()
                    idxs[c:c + y.shape[0], :] = y_idxs.numpy()
                    seeds[c:c + y.shape[0]] = seed

                    indexed_data['filename'].extend(fnames)
                    indexed_data['start'].extend(starts)
                    indexed_data['end'].extend(ends)
                    c += y.shape[0]
            # Reshape the data to not waste space
            vals.resize((c, config_parameters.topk))
            idxs.resize((c, config_parameters.topk))
            seeds.resize((c, ))
        ret_df = pd.DataFrame(indexed_data)
        ret_df['start'] = ret_df['start'].astype('float32')
        ret_df['end'] = ret_df['end'].astype('float32')
        ret_df.to_parquet(output_parquet, index=False, compression='brotli')


def save_parallel(config, **override_kwargs):
    # Best use with torchrun:
    # torchrun --nproc_per_node 8 save_logits.py logitconfig/balanced.yaml  --num_workers 8

    config_parameters = utils.parse_config_or_kwargs(config,
                                                     config_type=utils.Config,
                                                     **override_kwargs)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    model_name = config_parameters.model
    if isinstance(config_parameters.model, List):
        model_name = [
            mdl_name.replace('audiotransformer_', '')
            for mdl_name in config_parameters.model
        ]
        model_name = '_'.join(model_name)

    output_dir = Path(config_parameters.outputpath) / model_name / Path(
        config_parameters.train_data
    ).stem / f"chunk_{config_parameters.chunk_length:.0f}"
    output_dir.mkdir(exist_ok=True, parents=True)

    wavtransforms = utils.parse_wavtransforms(config_parameters.wavtransforms)
    spectransforms = utils.parse_spectransforms(
        config_parameters.spectransforms)

    torch.save(
        {
            'wavtransforms': wavtransforms,
            'spectransforms': spectransforms,
            'config': config_parameters
        }, output_dir / 'meta.pt')

    df = pd.read_csv(config_parameters.train_data,
                     converters={'filename': lambda x: Path(x).name},
                     sep='\t')
    #Split dataframe into parts
    dfs: List[pd.DataFrame] = np.array_split(df, world_size)

    tempfiles = [
        tempfile.NamedTemporaryFile(suffix='.parquet') for _ in range(len(dfs))
    ]
    # Dump splits for each worker process
    for i in range(len(dfs)):
        dfs[i].to_parquet(tempfiles[i].name, compression='brotli')

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        logger.info(f"Output can be found at {output_dir}")

    with idist.Parallel(backend='gloo') as parallel:
        parallel.run(_save_logits_parallel, [tmp.name for tmp in tempfiles],
                     output_dir, wavtransforms, spectransforms,
                     config_parameters)
    for tmp in tempfiles:
        tmp.close()


if __name__ == "__main__":
    Fire(save_parallel)
