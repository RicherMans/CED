import argparse

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import torch
import torchaudio

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

    opts = knf.FbankOptions()
    opts.mel_opts.is_librosa= False
    opts.mel_opts.num_bins = 64
    opts.mel_opts.htk_mode = 0
    opts.mel_opts.low_freq = 0
    opts.mel_opts.norm = ""
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = 0
    opts.frame_opts.window_type = "hann"
    opts.frame_opts.preemph_coeff = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.frame_length_ms = 32
    opts.frame_opts.frame_shift_ms = 10
    print(opts)
    online_whisper_fbank = knf.OnlineFbank(opts)

    audio, sr = torchaudio.load(args.input_wav)
    online_whisper_fbank.accept_waveform(sampling_rate=16000, waveform=audio.numpy().flatten())
    online_whisper_fbank.input_finished()


    features = []
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)
    mel = features.unsqueeze(0).permute(0,2,1)

    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.model, providers=providers, sess_options=sess_options)

    mellen = mel.shape[2]
    start = 0
    while True:
        #10-second detection once
        melt = mel[:,:,start:start + 1000]
        if melt.shape[-1] == 0:
            break
        start += 1000

        x = sess.run(None, input_feed={'feats': melt.numpy()})
        x = x[0][0]

        argmax = np.argmax(x)
        print(argmax, x[argmax])

        sorted_indices = np.argsort(x)
        top3_indices = sorted_indices[-3:]
        top3_indices = top3_indices[::-1]
        print(top3_indices)


if __name__ == "__main__":
    main()


