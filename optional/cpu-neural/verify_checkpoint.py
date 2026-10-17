"""Verify serialization and encoder update, without rerunning the held-out test."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from predict_cpu import predict
root=Path(__file__).resolve().parent
base=load_file(root/'pretrained/model.safetensors'); tuned=load_file(root/'output/checkpoint/model.safetensors')
key='bert.encoder.layer.0.attention.self.query.weight'
assert key in base and key in tuned
change=float(torch.abs(base[key]-tuned[key]).max())
assert change>0
short=predict('This is a fictional software pipeline test.',root/'output/checkpoint')
again=predict('This is a fictional software pipeline test.',root/'output/checkpoint')
assert short==again, 'Reloaded checkpoint must produce identical probe predictions'
assert short['flag']=='INSUFFICIENT_EVIDENCE' and short['human_review_required']
assert abs(sum(short['research_probabilities'].values())-1)<1e-6
long=predict('word '*600,root/'output/checkpoint')
assert long['reason_code']=='OVERLENGTH_REQUIRES_REVIEWED_WINDOWS' and 'research_probabilities' not in long
assert predict('',root/'output/checkpoint')['reason_code']=='EMPTY_INPUT'
result={'checkpoint_reload':'passed','deterministic_predictions_across_two_reloads':'passed','pretrained_encoder_tensor':key,'encoder_max_absolute_weight_change':change,'operational_abstention':'passed','overlength_rejection':'passed','empty_input_rejection':'passed','held_out_test_rerun':False}
(root/'output/verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
