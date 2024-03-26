import torch
import torchaudio
import torchaudio.transforms as aut
import onnxruntime as ort
import numpy as np
import argparse
import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_wav')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        metavar=
        f"Public Checkpoint [{','.join(models.list_models())}] or Experiment Path",
        nargs='?',
        choices=models.list_models(),
        default='ced_mini.onnx')

    args = parser.parse_args()

    trans1 = aut.MelSpectrogram(f_min=0,
                                sample_rate=16000,
                                win_length=512,
                                center=True,
                                n_fft=512,
                                f_max=8000,
                                hop_length=160,
                                n_mels=64)
    trans2 = aut.AmplitudeToDB(top_db=120)

    wav, sr = torchaudio.load(args.input_wav)
    wav = wav[0]
    wavlen = wav.shape[0]
    padding_size = 48000 - (wavlen % 48000)
    wav = torch.nn.functional.pad(wav, (0, padding_size), 'constant', 0)


    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.model, providers=providers, sess_options=sess_options)

    wavlen = wav.shape[0]
    start = 0
    while start + 16000 * 3 <= wavlen: #3秒检测一次
        wavt = wav[start:start + 16000 * 3]
        start += 16000 * 3

        mel = trans1(wavt)
        mel = trans2(mel)
        mel = mel.unsqueeze(0)
        x = sess.run(None, input_feed={'feats': mel.numpy()})
        x = x[0][0]

        argmax = np.argmax(x)
        print(argmax, x[argmax])

        sorted_indices = np.argsort(x)
        top3_indices = sorted_indices[-3:]
        top3_indices = top3_indices[::-1]
        print(top3_indices)



if __name__ == "__main__":
    main()


