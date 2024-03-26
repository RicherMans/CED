import torch
import argparse
import models
import torch.nn as nn
import onnx
from typing import Dict
from onnxruntime.quantization import QuantType, quantize_dynamic

DEVICE = torch.device('cpu')
SR = 16000

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

    args = parser.parse_args()


    model = getattr(models, args.model)(pretrained=True)
    model = model.to(DEVICE).eval()

    with torch.no_grad():
        class mymodel(nn.Module):
            def __init__(self):
                super(mymodel, self).__init__()
                self.model = model

            def forward(self, mel):
                # mel = mel.permute(0,2,1) #如果有需要，可以在这里进行一次转置，使特征维度和时间维度对调 如果对调，下面的动态轴也要对换
                feat = self.model.forward_spectrogram(mel)
                return feat

        model = mymodel()
        dummy_input = torch.ones(1, 64, 301)
        out = model(dummy_input)
        print(out.shape)


        output_model = args.model+'.onnx'
        torch.onnx.export(
            model, dummy_input,
            output_model,
            do_constant_folding=True,
            verbose=False,
            opset_version=12,
            input_names=['feats'],
            output_names=['prob'],
            dynamic_axes={
                'feats': {0: 'batch_size', 1: 'channel_dim', 2: 'feature_dim'}
            }
        )

        meta_data = {
            "model_type": "CED",
            "version": "1.0",
            "model_author": "RicherMans",
            "url": "https://github.com/RicherMans/CED",
        }
        add_meta_data(filename=output_model,meta_data=meta_data)

        print("Generate int8 quantization models")

        filename_int8 = args.model + ".int8.onnx"
        quantize_dynamic(
            model_input=output_model,
            model_output=filename_int8,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        #ced_mini onnx-39.1mb int8onnx-9.8mb

if __name__ == "__main__":
    main()
