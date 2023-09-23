import torch
import pandas as pd
import torchaudio
from pathlib import Path
import argparse
import models
from models.checkpoints import list_models

DEVICE = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_wav', type=Path, nargs="+")
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
        '-k',
        '--topk',
        type=int,
        help="Print top-k results",
        default=3,
    )
    parser.add_argument(
        '-c',
        '--chunk_length',
        type=float,
        help="Chunk Length for inference",
        default=10.0,
    )
    args = parser.parse_args()
    cl_lab_idxs_file = Path(
        __file__
    ).parent / 'datasets/audioset/data/metadata/class_labels_indices.csv'

    label_maps = pd.read_csv(
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        if not cl_lab_idxs_file.exists() else cl_lab_idxs_file).set_index(
            'index')['display_name'].to_dict()

    model = getattr(models, args.model)(pretrained=True)
    model = model.to(DEVICE).eval()

    with torch.no_grad():
        for wavpath in args.input_wav:
            wave, sr = torchaudio.load(wavpath)
            assert sr == SR, "Models are trained on 16khz, please sample your input to 16khz"
            with torch.no_grad():
                print(f"===== {str(wavpath):^20} =====")
                for chunk_idx, chunk in enumerate(
                        wave.split(int(args.chunk_length * sr), -1)):
                    output = model(chunk.to(DEVICE)).squeeze(0)
                    for k, (prob,
                            label) in enumerate(zip(*output.topk(args.topk))):
                        lab_idx = label.item()
                        label_name = label_maps[lab_idx]
                        print(
                            f"[{chunk_idx*args.chunk_length}s]-[{(chunk_idx+1)*args.chunk_length}s] Topk-{k+1} {label_name:<30} {prob:.4f}"
                        )
                    print()

if __name__ == "__main__":
    main()
