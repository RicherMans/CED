import sys
import torch
from typing import Callable, Dict, Optional


_model_functions: Dict[str, Callable] = {
}  # mapping of model names to entrypoint fns


def register_model(fn: Callable) -> Callable:
    mod = sys.modules[fn.__module__]
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_functions[model_name] = fn
    return fn


def build_mdl(model_fn,
              pretrained: bool = False,
              pretrained_url: Optional[str] = None,
              **model_kwargs
              ):
    mdl = model_fn(**model_kwargs)
    if pretrained and pretrained_url is not None:
        if 'http' in pretrained_url:
            dump = torch.hub.load_state_dict_from_url(pretrained_url,
                                                      map_location='cpu')
        else:
            dump = torch.load(pretrained_url, map_location='cpu')
        if 'model' in dump:
            dump = dump['model']
        # torch.
        mdl.load_state_dict(dump, strict=False)
    return mdl

def list_models():
    return _model_functions.keys()
