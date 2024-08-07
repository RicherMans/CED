import argparse
from typing import Dict

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

import models

DEVICE = torch.device('cpu')


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxExportModel(nn.Module):
    def __init__(self,model):
        super(OnnxExportModel, self).__init__()
        self.model = model

    def forward(self, mel):
        # mel = mel.permute(0,2,1)
        # If necessary, transpose can be performed here, switching the feature dimensions with the time dimensions.
        # If transposed, the following dynamic axes should also be interchanged.
        feat = self.model.forward_spectrogram(mel)
        return feat


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=
        f"Public Checkpoint [{','.join(models.list_models())}] or Experiment Path",
        nargs='?',
        choices=models.list_models(),
        default='ced_mini')

    parser.add_argument(
        '--max-frames',
        type=int,
        default=1012,
        help="Max number of frames the model can process."
        )

    args = parser.parse_args()


    model = getattr(models, args.model)(target_length=args.max_frames,
                                        pretrained=True)
    dummy_input = torch.ones(1, model.n_mels, args.max_frames)
    model = model.to(DEVICE).eval()

    model = OnnxExportModel(model)
    out = model(dummy_input)
    print(f"Model Output is {out.shape}")

    output_model = args.model + '.onnx'
    torch.onnx.export(model,
                      dummy_input,
                      output_model,
                      do_constant_folding=True,
                      verbose=False,
                      opset_version=12,
                      input_names=['feats'],
                      output_names=['prob'],
                      dynamic_axes={
                          'feats': {
                              0: 'batch_size',
                              2: 'time_dim'
                          },
                          'prob': {
                              0: 'batch_size'
                          }
                      })

    meta_data = {
        "model_type": "CED",
        "version": "1.0",
        "model_author": "RicherMans",
        "url": "https://github.com/RicherMans/CED",
    }
    add_meta_data(filename=output_model, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = args.model + ".int8.onnx"
    quantize_dynamic(
        model_input=output_model,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    print(f"Results is at {filename_int8}")
    # ced_mini onnx-39.1mb int8onnx-9.8mb


if __name__ == "__main__":
    main()
