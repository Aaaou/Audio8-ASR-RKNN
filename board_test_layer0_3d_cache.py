"""RK3576 acceptance probe for the rank-3 CPU-owned KV cache contract."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    return None


def l2(actual, reference):
    return float(np.linalg.norm(actual.astype(np.float64) - reference.astype(np.float64)) /
                 np.linalg.norm(reference.astype(np.float64)))


def summary(samples):
    a = np.asarray(samples, dtype=np.float64) * 1000
    return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95))}


ap = argparse.ArgumentParser()
ap.add_argument('--dir', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
d = args.dir
load = lambda name: np.load(d / (name + '.npy')).astype(np.float32)
hidden, cos, sin = load('hidden'), load('cos'), load('sin')
full_k, full_v = load('full_k'), load('full_v')
key_ref, value_ref, hidden_ref = load('key_delta_reference'), load('value_delta_reference'), load('hidden_reference')
before = rss_mib()
kv, block = RKNNLite(verbose=False), RKNNLite(verbose=False)
try:
    assert kv.load_rknn(str(d / 'kv_3d_fp16.rknn')) == 0
    assert block.load_rknn(str(d / 'block_3d_fp16.rknn')) == 0
    assert kv.init_runtime() == 0 and block.init_runtime() == 0
    after_init = rss_mib()
    key_delta, value_delta = kv.inference(inputs=[hidden, cos, sin])
    append_start = time.perf_counter()
    # The CPU-owned cache update: append only the just-produced delta.
    runtime_k = np.concatenate((full_k[:, :-1, :], key_delta), axis=1)
    runtime_v = np.concatenate((full_v[:, :-1, :], value_delta), axis=1)
    append_s = time.perf_counter() - append_start
    out = block.inference(inputs=[hidden, runtime_k, runtime_v, cos, sin])[0]
    kv_times, block_times, append_times = [], [], []
    for _ in range(30):
        t = time.perf_counter(); kd, vd = kv.inference(inputs=[hidden, cos, sin]); kv_times.append(time.perf_counter() - t)
        t = time.perf_counter(); rk = np.concatenate((full_k[:, :-1, :], kd), axis=1); rv = np.concatenate((full_v[:, :-1, :], vd), axis=1); append_times.append(time.perf_counter() - t)
        t = time.perf_counter(); out = block.inference(inputs=[hidden, rk, rv, cos, sin])[0]; block_times.append(time.perf_counter() - t)
    result = {
        'contract': {'cache_owner': 'CPU', 'cache_input_layout': '[heads, sequence, head_dim]', 'cache_shape': list(full_k.shape), 'cache_bytes_fp32_kv': int(full_k.nbytes + full_v.nbytes)},
        'accuracy': {'key_delta_normalized_l2': l2(key_delta, key_ref), 'value_delta_normalized_l2': l2(value_delta, value_ref), 'layer0_hidden_normalized_l2': l2(out, hidden_ref)},
        'latency_ms': {'kv_npu': summary(kv_times), 'cpu_append': summary(append_times), 'block_npu': summary(block_times), 'first_cpu_append_ms': append_s * 1000},
        'rss_mib': {'before_init': before, 'after_init': after_init, 'after_runs': rss_mib()},
    }
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
finally:
    kv.release(); block.release()
