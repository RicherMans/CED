import torch
import argparse
import models
import torch.nn as nn

DEVICE = torch.device('cpu')
SR = 16000


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

if __name__ == "__main__":
    main()
