import argparse
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.transforms as aut


# Taken from torchaudio
def amp_to_db(x,
              top_db: Optional[float] = None,
              multiplier: float = 10.0,
              amin=1e-10):
    x_db = multiplier * np.log10(np.clip(x, a_min=amin, a_max=None))
    if top_db is not None:
        x_db = np.maximum(x_db, np.amax(x_db) - top_db)
    return x_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_wav')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=
        f"Path to exported onnx model",
        nargs='?',
        default='ced_mini.int8.onnx')
    parser.add_argument('--chunk',default=5, type=float)

    args = parser.parse_args()

    front_end = torch.nn.Sequential(
        aut.MelSpectrogram(f_min=0,
                           sample_rate=16000,
                           win_length=512,
                           center=False,
                           n_fft=512,
                           f_max=8000,
                           hop_length=160,
                           n_mels=64), aut.AmplitudeToDB(top_db=120))

    wav, sr = torchaudio.load(args.input_wav)
    #Stereo
    if wav.ndim == 2:
        wav = wav.mean(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(args.model,
                                providers=providers,
                                sess_options=sess_options)

    start = 0
    chunk_length = int(args.chunk * 16000)
    while True:
        wavt = wav[start:start + chunk_length]
        if wavt.shape[-1] == 0:
            break
        start += chunk_length

        mel = front_end(wavt)
        mel = mel.unsqueeze(0)
        y = sess.run(None, input_feed={'feats': mel.numpy()})
        y = y[0][0]

        argmax = np.argmax(y)
        print(f"{argmax=} {y[argmax]=}")

        sorted_indices = np.argsort(y)
        top3_indices = sorted_indices[-3:]
        top3_indices = top3_indices[::-1]
        print(f"Top-3: {top3_indices}")


if __name__ == "__main__":
    main()
