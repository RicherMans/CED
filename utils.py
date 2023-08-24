from dataclasses import dataclass, field, asdict
from einops import rearrange
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from loguru import logger
from ignite.metrics import EpochMetric, RunningAverage
from sklearn.metrics import average_precision_score
import numpy as np
from packaging import version
import pandas as pd
import torch
import torch_audiomentations as wavtransforms
import yaml

import augment


@dataclass
class Config:
    train_data: Path
    model: str
    outputpath: Path = Path('logits/')
    model_args: Optional[Dict] = field(default_factory=dict)
    chunk_size: int = 50000 # For each dataframe to read at once
    chunk_length: float = 10.0 # For the crops
    num_workers:int = 8
    epochs: int = 40 # 120 for AS-20K, 10 for 2M
    batch_size: int = 128
    topk: int = 20

    mode: Optional[str] = None # Can also be amp
    # float_16: bool = True

    spectransforms: Union[List, Dict] = field(default_factory=dict)
    wavtransforms: Dict = field(default_factory=dict)

@dataclass
class TrainConfig:
    #Path stuff
    train_data: Path
    eval_data: Path
    logitspath: Path
    outputpath: Path = Path('experiments/')
    logfile: str = 'train.log'
    #Model stuff
    model: str = 'MobileNetV2'
    model_args: Optional[Dict[str,Any]] = field(default_factory=dict)
    pretrained: Optional[Path] = None

    #Dataloader Stuff
    mode: str = 'noamp' # Can also be amp for fp16
    num_workers: int = 4
    batch_size: int = 32
    eval_batch_size: int = batch_size
    num_classes: int = 527
    label_type: str = 'zero'
    max_aug_epochs: int = 0 # Default do infer from data
    # training stuff
    loss: str = 'BCELoss'
    loss_args: Dict = field(default_factory=dict)

    optimizer: str = 'Adam8bit'
    optimizer_args: Dict = field(default_factory=lambda: {'lr': 0.001})

    epoch_length: Optional[int] = None
    epochs: int = 120

    sampler: Optional[str] = None # Can be 'balanced'

    mixup: Optional[float] = None
    decay_frac: float = 0.1
    warmup_iters: Optional[int] = 5000
    warmup_epochs: Optional[int] = None
    use_scheduler: bool = True
    valid_every: int = 1
    early_stop: int = 10
    average: bool = True
    n_saved: int = 4


    disable_consistency: bool = False
    debug: bool = False # Enables using a PSL model during training

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Metrics:
    # mAP: Any = field(default_factory=lambda y_pred, y_tar: EpochMetric(
    # lambda y_pred, y_tar: np.nanmean(
    # average_precision_score(y_tar.to('cpu').numpy(),
    # y_pred.to('cpu').numpy(),
    # average=None)),
    # check_compute_fn=False))
    mAP: Callable[..., EpochMetric] = lambda :EpochMetric(
        lambda y_pred, y_tar: np.nanmean(
            average_precision_score(y_tar.to('cpu').numpy(),
                                    y_pred.to('cpu').numpy(),
                                    average=None)))

    def to_dict(self):
        return asdict(self)

    def get_metrics(self, metric_names: List[str]) -> Dict[str, EpochMetric]:
        own_dict = self.to_dict()
        return {met: own_dict[met]() for met in metric_names}


# AnyConfig = Callable[..., Union[Config, TrainConfig]]
AnyConfig = Union[Config, TrainConfig]

class KwargsSequential(torch.nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        for mod in self._modules.values():
            x = mod(x, *args, **kwargs)
        return x


def mixup(x: torch.Tensor, lamb: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """

    x1 = rearrange(x.flip(0), 'b ... -> ... b')
    x2 = rearrange(x.detach(), 'b ... -> ... b')
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')

def mixup_simple(x: torch.Tensor, lamb: torch.Tensor):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """

    x1 = rearrange(x.flip(0), 'b ... -> ... b')
    x2 = rearrange(x.detach(), 'b ... -> ... b')
    mixed = x1 * lamb + x2 * (1. - lamb)
    return rearrange(mixed, '... b -> b ...')


def mixup_criterion(x: torch.Tensor, y: torch.Tensor, lamb: torch.Tensor,
                    criterion: torch.nn.Module):
    """                                                                                     x: Tensor of shape ( batch_size, ... )         
    lamb: lambdas [0,1] of shape (batch_size)
    """
    mixed_loss = lamb * criterion(
        x, y.flip(0)).mean(-1) + (1. - lamb) * criterion(x, y).mean(-1)
    return mixed_loss.mean()


def parse_wavtransforms(transforms_dict: Dict):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans_name, v in transforms_dict.items():
        transforms.append(getattr(wavtransforms, trans_name)(**v))

    return torch.nn.Sequential(*transforms)


def parse_spectransforms(transforms: Union[List, Dict]):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    if isinstance(transforms, dict):
        return KwargsSequential(*[
            getattr(augment, trans_name)(**v)
            for trans_name, v in transforms.items()
        ])
    elif isinstance(transforms, list):
        return KwargsSequential(*[
            getattr(augment, trans_name)(**v)
            for item in transforms
            for trans_name, v in item.items()
        ])
    else:
        raise ValueError("Transform unknown")


def read_tsv_data(path: str, basename=True, nrows:Optional[int] = None) -> pd.DataFrame:
    if version.parse(pd.__version__) >= version.parse('2.rc1'):
        # Super fast, with pyarrow
        df: pd.DataFrame = pd.read_csv(path,
                                       sep='\t',
                                       nrows=nrows,
                                       engine='pyarrow')
        # engine='pyarrow',
        # dtype_backend='pyarrow')
    else:
        df: pd.DataFrame = pd.read_csv(path, sep='\s+', nrows=nrows)
    if 'labels' in df.columns:
        df['labels'] = df['labels'].astype(str)
        df['labels'] = df['labels'].str.split(';').apply(
            lambda x: np.array(x, dtype=int).tolist()).reset_index(drop=True)
    if 'prob' in df.columns:
        df['prob'] = df['prob'].str.split(';').apply(
            lambda x: np.array(x, dtype=float)).reset_index(drop=True)
    if 'idxs' in df.columns:
        df['idxs'] = df['idxs'].str.split(';').apply(
            lambda x: np.array(x, dtype=int).tolist()).reset_index(drop=True)

    if basename:
        df['filename'] = df['filename'].str.rsplit('/',n=1).str[-1]
    return df


def read_tsv_data_chunked(path: str,
                          chunk_length: int = 2,
                          chunk_hop: Optional[int] = None,
                          nrows: Optional[int] = None,
                          basename: bool = True):
    df = pd.read_csv(path, sep='\t', nrows=nrows).dropna(
    )  #drops some indices during evaluation which have no labels
    #Super slow otherwise
    if basename:  # Get basename instead of abspath
        df['filename'] = df['filename'].str.rsplit('/',n=1).str[-1]
    if 'label' in df.columns:
        df['labels'] = df['label']
        del df['label']
    if 'labels' in df.columns and not pd.api.types.is_numeric_dtype(
            df['labels']):
        df['labels'] = df['labels'].str.split(';').apply(
            lambda x: np.array(x, dtype=int).tolist())
    elif 'labels' in df.columns and pd.api.types.is_numeric_dtype(
            df['labels']):
        # Single labels, just transform to [LABEL] for mat for dataloader
        df['labels'] = df['labels'].apply(
            lambda x: np.array([x], dtype=int).tolist())
    if chunk_hop == None:
        chunk_hop = chunk_length

    df['from'] = df['duration'].apply(lambda x: np.arange(0, x, chunk_hop))
    df = df.explode('from')
    # Maximum between max duration and chunk lengths as duration
    df['to'] = np.minimum(df['from'] + chunk_length, df['duration'])
    # If there are any 0.0 duration elements, just drop em
    df = df.dropna()
    return df.reset_index(drop=True)  # In case index has been modified


def parse_config_or_kwargs(config_file, config_type: Type[AnyConfig],
                           **kwargs) -> AnyConfig:
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = config_type(**dict(yaml_config, **kwargs))
    # In case some arguments were not passed, replace with default ones
    return arguments


class DictWrapper(object):
    def __init__(self, adict):
        self.dict = adict

    def state_dict(self):
        return self.dict

    def load_state_dict(self, state):
        self.dict = state


def average_models(models: List[str]):
    model_res_state_dict = {}
    state_dict = {}
    has_new_structure = False
    for m in models:
        cur_state = torch.load(m, map_location='cpu')
        if 'model' in cur_state:
            has_new_structure = True
            model_params = cur_state.pop('model')
            # Append non "model" items, encoder, optimizer etc ...
            for k in cur_state:
                state_dict[k] = cur_state[k]
            # Accumulate statistics
            for k in model_params:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += model_params[k]
                else:
                    model_res_state_dict[k] = model_params[k]
        else:
            for k in cur_state:
                if k in model_res_state_dict:
                    model_res_state_dict[k] += cur_state[k]
                else:
                    model_res_state_dict[k] = cur_state[k]

    # Average
    for k in model_res_state_dict:
        # If there are any parameters
        if model_res_state_dict[k].ndim > 0:
            model_res_state_dict[k] /= float(len(models))
    if has_new_structure:
        state_dict['model'] = model_res_state_dict
    else:
        state_dict = model_res_state_dict
    return state_dict


def _overlap(start1, end1, start2, end2):
    """Does the range (start1, end1) overlap with (start2, end2)?"""
    return (start1 <= start2 <= end1 or start1 <= end2 <= end1
            or start2 <= start1 <= end2 or start2 <= end1 <= end2)


class FixSeedContext(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __exit__(self, *_):
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)


def load_pretrained(model: torch.nn.Module, trained_model: dict):
    if 'model' in trained_model:
        trained_model = trained_model['model']
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model[k].shape)
    }
    assert len(pretrained_dict) > 0, "Couldnt load pretrained model"
    # Found time positional embeddings ....
    if 'time_pos_embed' in trained_model.keys():
        pretrained_dict['time_pos_embed'] = trained_model['time_pos_embed']
        pretrained_dict['freq_pos_embed'] = trained_model['freq_pos_embed']
    if 'cache' in model.__class__.__name__.lower():
        logger.debug("Found a cached model, moving qkv values over.")
        for k, v in trained_model.items():
            if 'qkv' in k:
                pretrained_dict[k] = v

    logger.info(
        f"Loading {len(pretrained_dict)} Parameters for model {model.__class__.__name__}"
    )
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs="+")
    parser.add_argument('-o',
    '--output',
    required=True,
    help="Output model (pytorch)")
    args = parser.parse_args()
    mdls = average_models(args.models)
    torch.save(mdls, args.output)
