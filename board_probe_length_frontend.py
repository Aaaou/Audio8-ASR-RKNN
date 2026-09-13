"""Diagnose fixed-length RKNN frontend/pre-fill agreement against exported FP32 tensors."""
import argparse, json
from pathlib import Path
import numpy as np
from rknnlite.api import RKNNLite

def rel_l2(a, b):
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64)) /
                 max(np.linalg.norm(b.astype(np.float64)), 1e-30))

def open_rknn(path):
    r = RKNNLite(verbose=False)
    assert r.load_rknn(str(path)) == 0 and r.init_runtime() == 0
    return r

p = argparse.ArgumentParser()
p.add_argument('--dir', type=Path, required=True)
p.add_argument('--prefill-rknn', type=Path, required=True)
p.add_argument('--adapter-rknn', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(); d = a.dir
enc = adp = pre = None
try:
    enc = open_rknn('/root/audio8-asr/encoder_f800_v2/audio_encoder_f800_fp16.rknn')
    adp = open_rknn(a.adapter_rknn)
    pre = open_rknn(a.prefill_rknn)
    feat = np.load(d/'input_features.npy').astype('float32')
    ref_embed = np.load(d/'input_embeddings.npy').astype('float32')
    pos = np.load(d/'audio_positions.npy').reshape(-1)
    audio = adp.inference(inputs=[enc.inference(inputs=[feat])[0]])[0]
    n = len(pos)
    assembled = ref_embed.copy(); assembled[0, pos, :] = audio[:n]
    actual = pre.inference(inputs=[assembled])[0]
    ideal = pre.inference(inputs=[ref_embed])[0]
    ref_logits = np.load(d/'last_logits.npy')
    result = {
        'audio_placeholders': n,
        'adapter_audio_rel_l2_vs_fp32_injected': rel_l2(audio[:n], ref_embed[0, pos, :]),
        'prefill_logits_rel_l2_rknn_ideal_vs_fp32': rel_l2(ideal, ref_logits),
        'prefill_logits_rel_l2_frontend_vs_rknn_ideal': rel_l2(actual, ideal),
        'argmax': {'fp32': int(ref_logits.argmax()), 'rknn_ideal_embedding': int(ideal.argmax()), 'rknn_frontend_embedding': int(actual.argmax())},
    }
    a.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
finally:
    for r in (enc, adp, pre):
        if r: r.release()
