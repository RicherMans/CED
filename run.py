from typing import Callable, List, Optional, Tuple, Dict, Any, Union
from loguru import logger
from fire import Fire
import pandas as pd
import uuid
import numpy as np
from dataclasses import asdict

import dataset
import utils
import torch
import sys
import datetime
from pathlib import Path
import ignite
import models
from ignite.contrib.handlers import ProgressBar, create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import (Checkpoint, DiskSaver, global_step_from_engine,
                             EarlyStopping)

logger.configure(handlers=[{
    "sink": sys.stderr,
    "format": "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}",
    'level': 'DEBUG',
}])

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')


def transfer_to_device(batch, device=DEVICE):
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)

def _step_amp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Union[Callable, torch.nn.Module],
    spectransforms: Callable,
    config_parameters: utils.TrainConfig,
):

    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast
    scaler = GradScaler(enabled=True)
    logger.info("Using AMP")

    def _update(engine, batch):
        model.train()
        with torch.enable_grad():
            x, y, specaug_value, specaug_min_value, fnames = transfer_to_device(
                batch)
            if config_parameters.disable_consistency:
                specaug_value = torch.rand_like(torch.tensor(range(len(x))).float())
                specaug_min_value = torch.rand_like(torch.tensor(range(len(x))).float())
            mixup_lamb = None
            if specaug_value is not None:
                x = model.front_end(x)

                x = spectransforms(x,
                                   values=torch.tensor(specaug_value),
                                   min_values=torch.tensor(specaug_min_value))
                if config_parameters.mixup is not None and config_parameters.mixup > 0.0:
                    mixup_lamb = torch.tensor(np.random.beta(
                        config_parameters.mixup,
                        config_parameters.mixup,
                        size=len(x)),
                                            device=DEVICE,
                                            dtype=torch.float32)
                    x = utils.mixup(x, mixup_lamb)
                with autocast(enabled=True):
                    model_pred = model.forward_spectrogram(x)
                    if isinstance(model_pred, tuple):
                        model_pred = model_pred[0]
                    if mixup_lamb is not None:
                        loss = utils.mixup_criterion(model_pred,
                                                    y,
                                                    lamb=mixup_lamb,
                                                    criterion=criterion)
                    else:
                        loss = criterion(model_pred, y)
            else:
                with autocast(enabled=True):
                    if config_parameters.mixup is not None and config_parameters.mixup > 0.0:
                        mixup_lamb = torch.tensor(np.random.beta(
                            config_parameters.mixup,
                            config_parameters.mixup,
                            size=len(x)),
                                                device=DEVICE,
                                                dtype=torch.float32)
                        x = utils.mixup(x, mixup_lamb)
                    model_pred = model(x)
                    if isinstance(model_pred, tuple):
                        model_pred = model_pred[0]
                    if mixup_lamb is not None:
                        loss = utils.mixup_criterion(model_pred,
                                                    y,
                                                    lamb=mixup_lamb,
                                                    criterion=criterion)
                    else:
                        loss = criterion(model_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
            return {
                'total_loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }

    return _update

def _step_ampbf16(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Union[Callable, torch.nn.Module],
    spectransforms: Callable,
    config_parameters: utils.TrainConfig,
):

    from torch.cuda.amp import autocast
    logger.info("Using AMP BF16")

    def _update(engine, batch):
        model.train()
        with torch.enable_grad():
            x, y, specaug_value, specaug_min_value, fnames = transfer_to_device(
                batch)
            if config_parameters.disable_consistency:
                specaug_value = torch.rand_like(torch.tensor(specaug_value).float())
                specaug_min_value = torch.rand_like(torch.tensor(specaug_min_value).float())
            mixup_lamb = None
            if specaug_value is not None:
                x = model.front_end(x)
                x = spectransforms(x,
                                   values=torch.tensor(specaug_value),
                                   min_values=torch.tensor(specaug_min_value))
                if config_parameters.mixup is not None and config_parameters.mixup > 0.0:
                    mixup_lamb = torch.tensor(np.random.beta(
                        config_parameters.mixup,
                        config_parameters.mixup,
                        size=len(x)),
                                            device=DEVICE,
                                            dtype=torch.float32)
                    x = utils.mixup(x, mixup_lamb)
                with autocast(enabled=True, dtype=torch.bfloat16):
                    model_pred = model.forward_spectrogram(x)
                    if isinstance(model_pred, tuple):
                        model_pred = model_pred[0]
                    if mixup_lamb is not None:
                        loss = utils.mixup_criterion(model_pred,
                                                    y,
                                                    lamb=mixup_lamb,
                                                    criterion=criterion)
                    else:
                        loss = criterion(model_pred, y)
            else:
                with autocast(enabled=True, dtype=torch.bfloat16):
                    model_pred = model(x)
                    if isinstance(model_pred, tuple):
                        model_pred = model_pred[0]
                    loss = criterion(model_pred, y)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            return {
                'total_loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }

    return _update

def _step(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: Union[Callable, torch.nn.Module],
          spectransforms: Callable,
          config_parameters: utils.TrainConfig,
          ):

    def _update(engine, batch):
        model.train()
        with torch.enable_grad():
            optimizer.zero_grad(set_to_none=True)
            x, y, specaug_value, specaug_min_value, fnames = transfer_to_device(
                batch)
            if config_parameters.disable_consistency:
                specaug_value = torch.rand_like(torch.tensor(specaug_value).float())
                specaug_min_value = torch.rand_like(torch.tensor(specaug_min_value).float())
            mixup_lamb = None
            if specaug_value is not None:
                x = model.front_end(x)
                x = spectransforms(x,
                                   values=torch.tensor(specaug_value),
                                   min_values=torch.tensor(specaug_min_value))
                if config_parameters.mixup is not None and config_parameters.mixup > 0.0:
                    #Spectrogram level specaug
                    mixup_lamb = torch.tensor(np.random.beta(
                    config_parameters.mixup, config_parameters.mixup, size=len(x)),
                    device=DEVICE,
                    dtype=torch.float32)
                    x = utils.mixup(x, mixup_lamb)
                model_pred = model.forward_spectrogram(x)
            else:
                model_pred = model(x)
            if isinstance(model_pred, tuple):
                model_pred = model_pred[0]
            if mixup_lamb is not None:
                loss = utils.mixup_criterion(model_pred,
                                             y,
                                             lamb=mixup_lamb,
                                             criterion=criterion)
            else:
                loss = criterion(model_pred, y)
            loss.backward()
            optimizer.step()
            return {
                'total_loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
            }

    return _update


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Union[Callable, torch.nn.Module],
    spectransforms: Callable,
    config_parameters: utils.TrainConfig,
    mode: Optional[str] = None,
):

    if mode == 'amp':
        return _step_amp(model, optimizer, criterion, spectransforms, config_parameters)
    elif mode == 'amp_bf16':
        return _step_ampbf16(model, optimizer, criterion, spectransforms, config_parameters)
    return _step(model, optimizer, criterion,spectransforms, config_parameters)

def log_basic_info(params):
    config_parameters = params['params']
    logger.info(f"Running on device {DEVICE}")
    logger.info(f"Storing output in {params['outputdir']}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.current_device()}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
    for k, v in asdict(config_parameters).items():
        logger.info(f"{k} : {v}")


def create_engine(engine_function, evaluation_metrics: Optional[List[str]] = None):
    engine = Engine(engine_function)
    ProgressBar().attach(engine, output_transform=lambda x: x)

    if evaluation_metrics:
        eval_mets = utils.Metrics().get_metrics(evaluation_metrics)
        for name, metric in eval_mets.items():
            metric.attach(engine, name)
    return engine


class Runner(object):
    def __init__(self, seed: int = 42, nthreads: int = 1):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(nthreads)
        logger.info(f"Using seed {seed}")

    def __setup(self,
                config: Path,
                **override_kwargs) -> Dict[str, Any]:
        config_parameters = utils.parse_config_or_kwargs(
            config, config_type=utils.TrainConfig, **override_kwargs)
        outputdir = Path(config_parameters.outputpath) / Path(
            config).stem / f"{config_parameters.model}" / "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'),
                uuid.uuid1().hex)
        outputdir.mkdir(exist_ok=True, parents=True)
        log_fname = config_parameters.logfile
        output_log = outputdir / log_fname
        logger.add(
            output_log,
            enqueue=True,
            level='INFO',
            format=
            "[<red>{level}</red> <green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
        )
        return_params = {'outputdir': outputdir, 'params': config_parameters}
        log_basic_info(return_params)
        return return_params

    def train(self, config: Union[Path, str], **overwrite_kwargs):
        param_dict = self.__setup(Path(config), **overwrite_kwargs)
        config_parameters = param_dict['params']
        outputdir = param_dict['outputdir']
        meta_data = torch.load(Path(config_parameters.logitspath) / 'meta.pt')
        spectransforms = meta_data['spectransforms']
        wavtransforms = meta_data['wavtransforms']

        model = getattr(models, config_parameters.model)(
            pretrained=True, **config_parameters.model_args)

        logger.info(model)
        if config_parameters.pretrained is not None:
            utils.load_pretrained(model,
                                  trained_model=torch.load(
                                      config_parameters.pretrained,
                                      map_location='cpu'))

        model = model.to(DEVICE).train()
        if config_parameters.optimizer == 'Adam8bit':
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(
                model.parameters(),
                **config_parameters.optimizer_args)  # add bnb optimizer
        else:
            optimizer = getattr(torch.optim, config_parameters.optimizer)(
                model.parameters(), **config_parameters.optimizer_args)

        criterion = getattr(
            torch.nn, config_parameters.loss)(**config_parameters.loss_args)


        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, targets, *_ = transfer_to_device(batch)
                model_pred = model(data)
                if isinstance(model_pred, tuple):
                    model_pred = model_pred[0]
                return model_pred, targets

        def run_validation(engine, title=None):
            results = engine.state.metrics
            output_str_list = [
                f"{title:<10} Results - Epoch : {train_engine.state.epoch:<4}"
            ]
            for metric in results:
                if isinstance(results[metric], np.ndarray):
                    pass
                else:
                    output_str_list += [f"{metric} {results[metric]:<5.4f}"]
            output_str_list += [f"LR: {optimizer.param_groups[0]['lr']:.4e}"]
            logger.info(" ".join(output_str_list))

        train_engine = create_engine(
            create_supervised_trainer(model,
                                      optimizer,
                                      criterion,
                                      spectransforms,
                                      config_parameters,
                                      mode=config_parameters.mode))
        inference_engine = create_engine(
            _inference,
            evaluation_metrics=['mAP'])  # Common mAP between all datasets

        train_df = utils.read_tsv_data(config_parameters.train_data,
                                       basename=True)
        eval_df = utils.read_tsv_data(config_parameters.eval_data,
                                      basename=True)

        info_message = f"#Lengths: Train - {len(train_df)} Eval - {len(eval_df)}"
        logger.info(info_message)
        train_dataset = dataset.LogitsReader(
            train_df,
            logits_basepath=Path(config_parameters.logitspath),
            wavtransforms=wavtransforms
            if config_parameters.disable_consistency is False else None,
            label_type=config_parameters.label_type,
            max_epochs=config_parameters.max_aug_epochs,
            num_classes=config_parameters.num_classes)

        audioset_test_dataloader = torch.utils.data.DataLoader(
            dataset.WeakHDF5Dataset(eval_df,
                                    num_classes=config_parameters.num_classes),
            batch_size=config_parameters.eval_batch_size,
            num_workers=config_parameters.num_workers,
            shuffle=False,
            persistent_workers=True,
            collate_fn=dataset.sequential_pad,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config_parameters.batch_size,
            num_workers=config_parameters.num_workers,
            collate_fn=dataset.sequential_pad,
            shuffle=True)


        logger.info(f"Training Dataloader has size {len(train_dataloader)}")
        # Update Epoch for LogitsReader, Epoch starts at 1
        @train_engine.on(Events.EPOCH_STARTED)
        def restart_dataloader():
            train_engine.state.dataloader.dataset._set_epoch(train_engine.state.epoch)

        score_function = Checkpoint.get_default_score_fn(*['mAP', 1.0])
        checkpoint_saver = Checkpoint(
            {
                'model': model,
                'config': utils.DictWrapper(config_parameters.asdict()),
            },
            DiskSaver(outputdir),
            n_saved=config_parameters.n_saved,
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=score_function)
        decay_steps = config_parameters.epochs * len(
            train_dataloader
        ) if config_parameters.epoch_length == None else config_parameters.epochs * config_parameters.epoch_length
        if config_parameters.use_scheduler:
            scheduler = CosineAnnealingScheduler(
                optimizer, 'lr', optimizer.param_groups[0]['lr'],
                optimizer.param_groups[0]['lr'] * config_parameters.decay_frac,
                decay_steps)
            logger.info(f"Using scheduler {scheduler.__class__.__name__} with {decay_steps} steps.")

            warmup_iters_num = None
            if config_parameters.warmup_iters is not None:
                warmup_iters_num = config_parameters.warmup_iters
            elif config_parameters.warmup_epochs is not None:
                warmup_iters_num = config_parameters.warmup_epochs * len(
                    train_dataloader)
            if warmup_iters_num is not None:
                logger.info(
                    f"Warmup with {warmup_iters_num}, if you want to disable warmup pass warmup_iters = None"
                )
                scheduler = create_lr_scheduler_with_warmup(
                    scheduler,
                    warmup_start_value=0.0,
                    warmup_duration=warmup_iters_num)
            train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
        earlystop_handler = EarlyStopping(
            patience=config_parameters.early_stop,
            score_function=score_function,
            trainer=train_engine)
        # Stop on Wensheng no improvement
        inference_engine.add_event_handler(Events.COMPLETED,
                                                    earlystop_handler)

        inference_engine.add_event_handler(Events.COMPLETED,
                                                    checkpoint_saver)

        @train_engine.on(
            Events.EPOCH_COMPLETED(every=config_parameters.valid_every))
        def valid_eval(train_engine):
            with inference_engine.add_event_handler(
                    Events.COMPLETED, run_validation,
                    f"{config_parameters.eval_data}"):
                inference_engine.run(audioset_test_dataloader)

        train_engine.run(
            train_dataloader,
            max_epochs=config_parameters.epochs,
            epoch_length=config_parameters.epoch_length,
        )
        output_model = outputdir / checkpoint_saver.last_checkpoint
        if config_parameters.average:
            logger.info("Averaging best models ...")
            output_model = outputdir / 'averaged.pt'

            averaged_state_dict = utils.average_models(
                [outputdir / f.filename for f in checkpoint_saver._saved])
            torch.save(averaged_state_dict, output_model)

            model.load_state_dict(averaged_state_dict['model'], strict=True)
        else:
            logger.info(f"Loading best model {output_model}")
            model.load_state_dict(torch.load(output_model)['model'],
                                  strict=True)
        with inference_engine.add_event_handler(
                Events.COMPLETED, run_validation,
                f"{config_parameters.eval_data}"):
            inference_engine.run(audioset_test_dataloader)
        logger.info(f"Results can be found at {outputdir}")
        return output_model

    def _evaluate(
            self, inference_engine, audioset_test_dataloader):
        #Label Maps for audioset
        class_labels = 'data/class_labels_indices.csv'
        label_map_df = pd.read_csv(class_labels)
        label_map_df['display_name'] = label_map_df['display_name'].str.lower()
        label_maps = label_map_df.set_index(
            'index')['display_name'].to_dict()

        def log_metrics(engine, title, scale: float = 100):
            results = engine.state.metrics
            log = [f"{title:}"]
            for metric in results.keys():
                # Returned dict means that its for each class some result metric
                if isinstance(results[metric], np.ndarray):
                    if engine.label_maps is None:
                        engine.label_maps = {
                            idx: idx
                            for idx in range(len(results[metric]))
                        }
                    sorted_idxs = np.argsort(results[metric])[::-1]

                    for i, cl in enumerate(sorted_idxs):
                        log.append(
                            f"{metric} Class {engine.label_maps[cl]} : {results[metric][cl]*scale:<4.2f}"
                        )
                else:
                    log.append(f"{metric} : {results[metric]*scale:<4.2f}")
            logger.info("\n".join(log))

        inference_engine.label_maps = label_maps
        with inference_engine.add_event_handler(Events.COMPLETED, log_metrics,
                                                "Audioset Eval"):
            inference_engine.run(audioset_test_dataloader)

    def evaluate(self,
                 experiment_path: Union[str, Path],
                 test_data: str = 'data/eval_asedata.csv',
                 mode: Optional[str] = None, # can also be amp
                 ):
        from torch.cuda.amp import autocast
        experiment_path = Path(experiment_path)
        model_dump_path = None
        if experiment_path.is_file():
            # Is the file itself
            model_dump_path = experiment_path
            self.experiment_path = experiment_path.parent
        else:
            # Is a directory,need to find file
            model_dump_path = next(experiment_path.glob('*pt'))
            self.experiment_path = experiment_path
        model_dump = torch.load(model_dump_path, map_location='cpu')
        config_parameters = model_dump['config']
        if isinstance(config_parameters, dict):
            config_parameters = utils.TrainConfig(**config_parameters)
        num_classes = config_parameters.num_classes
        if 'pretrained' in config_parameters.model_args:
            config_parameters.model_args.pop(
                'pretrained')  # Dont need pretraining here

        model = getattr(
            models, config_parameters.model)(**config_parameters.model_args)
        model = model.to(DEVICE).eval()
        model.load_state_dict(model_dump['model'])

        def _inference(engine, batch):
            model.eval()
            with torch.no_grad():
                data, targets, lengths, *_ = transfer_to_device(batch)
                with autocast(enabled=mode == 'amp'):
                    clip_out = model(data)
                return clip_out, targets


        audioset_eval_df = utils.read_tsv_data(test_data, basename=True)
        dataloader = torch.utils.data.DataLoader(
            dataset.WeakHDF5Dataset(audioset_eval_df, num_classes=num_classes),
            batch_size=config_parameters.eval_batch_size,
            num_workers=4,
            shuffle=False,
            collate_fn=dataset.sequential_pad,
        )

        engine = create_engine(_inference,
                               evaluation_metrics=[
                                   'mAP'
                               ])
        logger.add(Path(self.experiment_path) /
                   f'evaluation_{Path(test_data).stem}.txt',
                   format='{message}',
                   level='INFO',
                   mode='w')
        self._evaluate(engine, dataloader)

if __name__ == '__main__':
    Fire(Runner)
