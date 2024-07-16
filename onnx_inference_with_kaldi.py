import argparse

import kaldi_native_fbank as knf
import numpy as np
from onnx_inference_with_torchaudio import amp_to_db
import onnxruntime as ort
import torch
import torchaudio



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
    opts.use_log_fbank=False
    # print(opts)
    online_fbank = knf.OnlineFbank(opts)

    audio, sr = torchaudio.load(args.input_wav)
    online_fbank.accept_waveform(sampling_rate=16000, waveform=audio.numpy().flatten())
    online_fbank.input_finished()


    features = []
    for i in range(online_fbank.num_frames_ready):
        f = amp_to_db(online_fbank.get_frame(i))
        features.append(f)

    features = np.stack(features, axis=-1)[None,...]

    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.model, providers=providers, sess_options=sess_options)

    start = 0
    chunk_length = int(args.chunk * 100) # in number of frames, per frame 10ms
    while True:
        #10-second detection once
        melt = features[:,:,start:start + chunk_length]
        if melt.shape[-1] == 0:
            break
        start += chunk_length

        x = sess.run(None, input_feed={'feats': melt})
        x = x[0][0]

        argmax = np.argmax(x)
        print(argmax, x[argmax])

        sorted_indices = np.argsort(x)
        top3_indices = sorted_indices[-3:]
        top3_indices = top3_indices[::-1]
        print(top3_indices)


if __name__ == "__main__":
    main()


