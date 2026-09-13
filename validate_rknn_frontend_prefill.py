"""Strictly validate the frozen RKNN audio frontend through Qwen2 first logits.

No generation-state or KV-cache code is involved.  Each sample compares
CPU-FP32 original audio projection against the chained FP16 RKNN encoder and
adapter, then compares the downstream CPU Qwen2 prefill logits.  The fixed
RKNN Qwen2 prefill graph is also checked using the NPU-produced embeddings.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = "Please transcribe this audio."

def measure(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    d = a-r
    return {"max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
      "normalized_l2": float(np.linalg.norm(d)/np.linalg.norm(r)),
      "cosine": float(np.sum(a*r)/(np.linalg.norm(a)*np.linalg.norm(r))),
      "reference_argmax": int(r.argmax()), "actual_argmax": int(a.argmax()),
      "argmax_match": bool(r.argmax() == a.argmax())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True); p.add_argument("--encoder-rknn",type=Path,required=True)
    p.add_argument("--adapter-rknn",type=Path,required=True); p.add_argument("--qwen-prefill-rknn",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int,default=2)
    args=p.parse_args()
    from rknnlite.api import RKNNLite
    processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation="eager").eval()
    engines=[RKNNLite(verbose=False) for _ in range(3)]
    try:
      for engine,path in zip(engines,(args.encoder_rknn,args.adapter_rknn,args.qwen_prefill_rknn)):
        if engine.load_rknn(str(path)) != 0 or engine.init_runtime() != 0: raise RuntimeError(f"RKNN init failed: {path}")
      rows=[]
      for row in json.loads(args.manifest.read_text())[:args.limit]:
        conversation=[{"role":"user","content":[{"type":"audio","path":row["audio"]},{"type":"text","text":PROMPT}]}]
        batch=dict(processor.apply_chat_template(conversation,return_tensors="pt",sampling_rate=16000,audio_padding="max_length",add_generation_prompt=True,audio_max_length=128000,text_kwargs={"padding":"longest","truncation":True,"max_length":1000}))
        features=batch["input_features"][0]
        audio_positions=torch.nonzero(batch["input_ids"][0].eq(model.audio_token_id),as_tuple=False).flatten()
        with torch.inference_mode():
          cpu_embeddings=model._inject_audio_embeddings(batch["input_ids"],batch["input_features"],batch.get("feature_lens"))
          cpu_logits=model.language_model(inputs_embeds=cpu_embeddings,use_cache=False,return_dict=True).logits[:,-1,:]
          cpu_encoded=model.audio_encoder(features,feature_lens=batch.get("feature_lens")).last_hidden_state
        npu_encoded=np.asarray(engines[0].inference(inputs=[features.numpy().astype(np.float32)])[0],dtype=np.float32)
        npu_audio=np.asarray(engines[1].inference(inputs=[npu_encoded])[0],dtype=np.float32)
        npu_embeddings=model.get_input_embeddings()(batch["input_ids"]).clone()
        npu_embeddings[0,audio_positions,:]=torch.from_numpy(npu_audio)
        with torch.inference_mode():
          cpu_qwen_on_npu_frontend=model.language_model(inputs_embeds=npu_embeddings,use_cache=False,return_dict=True).logits[:,-1,:]
        npu_qwen_logits=np.asarray(engines[2].inference(inputs=[npu_embeddings.numpy().astype(np.float32)])[0],dtype=np.float32)
        rows.append({"id":row.get("id"),"prompt_tokens":int(batch["input_ids"].shape[1]),"audio_tokens":int(audio_positions.numel()),
          "encoder_npu_vs_cpu":measure(npu_encoded,cpu_encoded.detach().numpy()),
          "audio_embeddings_npu_vs_cpu":measure(npu_audio,cpu_embeddings[0,audio_positions,:].detach().numpy()),
          "qwen_cpu_logits_npu_frontend_vs_cpu_frontend":measure(cpu_qwen_on_npu_frontend.detach().numpy(),cpu_logits.detach().numpy()),
          "qwen_rknn_logits_vs_cpu_on_same_npu_frontend":measure(npu_qwen_logits,cpu_qwen_on_npu_frontend.detach().numpy())})
      payload={"scope":"Frozen FP16 RKNN encoder+adapter and fixed S=110 Qwen2 no-cache prefill; no KV-cache code executed","samples":rows}
      args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
    finally:
      for engine in engines: engine.release()
if __name__=="__main__": main()
