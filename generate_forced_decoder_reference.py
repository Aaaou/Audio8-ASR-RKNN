"""FP32 reference for cache-scheduler stress: greedy decode intentionally ignores EOS."""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PROMPT = 'Please transcribe this audio.'
p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--audio', type=Path, required=True)
p.add_argument('--steps', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(); torch.set_grad_enabled(False)
processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation='eager').eval()
conversation = [{'role':'user','content':[{'type':'audio','path':str(a.audio)},{'type':'text','text':PROMPT}]}]
batch = dict(processor.apply_chat_template(conversation, return_tensors='pt', sampling_rate=16000, audio_padding='max_length', add_generation_prompt=True, audio_max_length=128000, text_kwargs={'padding':'longest','truncation':True,'max_length':1000}))
with torch.inference_mode():
    embeds = model._inject_audio_embeddings(batch['input_ids'], batch['input_features'], batch.get('feature_lens'))
    out = model.language_model(inputs_embeds=embeds, use_cache=True, return_dict=True)
    cache = out.past_key_values; token = out.logits[:, -1:].argmax(-1); tokens = [int(token)]
    start = cache.layers[0].keys.shape[-2]
    for step in range(a.steps):
        position = torch.tensor([[start + step]])
        hidden = model.get_input_embeddings()(token)
        out = model.language_model(inputs_embeds=hidden, past_key_values=cache, use_cache=True, cache_position=torch.tensor([start + step]), position_ids=position, return_dict=True)
        cache = out.past_key_values; token = out.logits[:, -1:].argmax(-1); tokens.append(int(token))
a.output.write_text(json.dumps({'mode':'greedy decoder scheduler stress; EOS intentionally ignored','prefill_length':int(start),'tokens':tokens,'steps_after_first_token':a.steps,'eos_id':model.config.eos_token_id}, indent=2))
