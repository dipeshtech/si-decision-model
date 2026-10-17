"""Synthetic-only full-encoder CPU experiment. No clinical or deployment validity."""
import argparse, hashlib, json, os, random, re, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, average_precision_score
from transformers import BertTokenizerFast, BertForSequenceClassification

ROOT = Path(__file__).resolve().parent

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def clean(text):
    # Synthetic bookkeeping has no semantic role; apply identically to every split.
    return re.sub(r'\s*Synthetic scenario \d+\.', '', text).strip()

def summary(rows, probs, labels):
    y = np.array([labels.index(r['label']) for r in rows]); pred = probs.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, labels=list(range(4)), average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(4))).tolist(),
            'classification_report': classification_report(y, pred, labels=list(range(4)), target_names=labels, output_dict=True, zero_division=0),
            'one_vs_rest_average_precision': {lab: float(average_precision_score(y==j, probs[:,j])) for j,lab in enumerate(labels)},
            'multiclass_brier_score': float(np.mean(np.sum((probs-np.eye(4)[y])**2, axis=1)))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=ROOT.parents[1]/'artifacts/synthetic-baseline-final'); p.add_argument('--output',type=Path,default=ROOT/'output'); args=p.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text()); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists(): raise RuntimeError('Completed run exists; use a new output directory. Do not retune on the test set.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    labels=cfg['labels']; names=['train','validation','calibration','test']; rows={s:[json.loads(l) for l in (args.data_dir/f'{s}.jsonl').read_text().splitlines()] for s in names}
    for s in names:
        assert set(r['label'] for r in rows[s])==set(labels)
        assert all(r['source_type']=='synthetic' for r in rows[s])
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for key in ['id','template_family','subject_id','conversation_id']:
                assert not ({r[key] for r in rows[a]} & {r[key] for r in rows[b]}), (a,b,key)
            assert not ({clean(r['text']) for r in rows[a]} & {clean(r['text']) for r in rows[b]}), (a,b,'text')
    manifest={s:{'count':len(rows[s]),'families':len(set(r['template_family'] for r in rows[s])), 'sha256':hashlib.sha256((args.data_dir/f'{s}.jsonl').read_bytes()).hexdigest()} for s in names}
    save(out/'data_manifest.json',manifest); save(out/'config.json',cfg)
    tokenizer=BertTokenizerFast(vocab_file=str(ROOT/'pretrained/vocab.txt'),do_lower_case=True)
    model=BertForSequenceClassification.from_pretrained(ROOT/'pretrained',local_files_only=True,use_safetensors=True,num_labels=4,id2label=dict(enumerate(labels)),label2id={l:i for i,l in enumerate(labels)})
    assert all(x.requires_grad for x in model.parameters()); assert next(model.parameters()).device.type=='cpu'
    lengths={s:[len(tokenizer.encode(clean(r['text']),add_special_tokens=True)) for r in rows[s]] for s in names}
    length_audit={s:{'max_tokens':max(lengths[s]),'over_limit':sum(n>cfg['max_length'] for n in lengths[s])} for s in names}
    save(out/'length_audit.json',length_audit)
    if any(v['over_limit'] for v in length_audit.values()): raise ValueError('Overlength synthetic records: window and relabel before training; silent truncation forbidden.')
    encoded={s:tokenizer([clean(r['text']) for r in rows[s]],padding=True,truncation=True,max_length=cfg['max_length'],return_tensors='pt') for s in names}
    targets={s:torch.tensor([labels.index(r['label']) for r in rows[s]]) for s in names}
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    def predict(s):
        model.eval(); values=[]
        with torch.inference_mode():
            for start in range(0,len(rows[s]),cfg['batch_size']):
                values.append(model(**{k:v[start:start+cfg['batch_size']] for k,v in encoded[s].items()}).logits.softmax(-1).numpy())
        return np.concatenate(values)
    started=time.perf_counter(); best=-1; history=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train(); order=torch.randperm(len(rows['train'])); loss_sum=0
        for ids in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True)
            result=model(**{k:v[ids] for k,v in encoded['train'].items()},labels=targets['train'][ids]); result.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); loss_sum+=result.loss.item()*len(ids)
        metrics=summary(rows['validation'],predict('validation'),labels)
        line={'epoch':epoch,'train_loss':loss_sum/len(order),'validation_macro_f1':metrics['macro_f1'],'validation_accuracy':metrics['accuracy'],'elapsed_seconds':time.perf_counter()-started}
        history.append(line); print(json.dumps(line),flush=True)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1']; best_epoch=epoch; model.save_pretrained(out/'checkpoint',safe_serialization=True); tokenizer.save_pretrained(out/'checkpoint')
        save(out/'history.json',history)
    train_seconds=time.perf_counter()-started
    model=BertForSequenceClassification.from_pretrained(out/'checkpoint',local_files_only=True,use_safetensors=True)
    # Thresholds selected on calibration only, never test. No posterior recalibration claimed.
    cp=predict('calibration'); cy=targets['calibration'].numpy(); thresholds={}
    for j,lab in enumerate(labels):
        candidates=[]
        for cutoff in np.unique(cp[:,j]):
            mask=(cp.argmax(1)==j)&(cp[:,j]>=cutoff)
            if mask.any() and float((cy[mask]==j).mean())>=cfg['minimum_calibration_precision']: candidates.append(float(cutoff))
        thresholds[lab]=min(candidates) if candidates else None
    save(out/'thresholds.json',thresholds)
    # Single final test evaluation after checkpoint/thresholds are fixed.
    test_started=time.perf_counter(); tp=predict('test'); test_seconds=time.perf_counter()-test_started
    test_metrics=summary(rows['test'],tp,labels); ty=targets['test'].numpy(); pred=tp.argmax(1)
    accepted=np.array([thresholds[labels[j]] is not None and float(p[j])>=thresholds[labels[j]] for p,j in zip(tp,pred)])
    test_metrics['experimental_threshold_coverage']=float(accepted.mean())
    test_metrics['experimental_threshold_accuracy_on_accepted']=float((pred[accepted]==ty[accepted]).mean()) if accepted.any() else None
    result={'warning':'SYNTHETIC ONLY. Unapproved custom taxonomy. Not clinically validated. All operational outputs require human review and abstain pending separate approval.', 'base_model':cfg['model_id'],'base_revision':cfg['model_revision'],'base_license':cfg['license'],'training_mode':cfg['training_mode'],'device':'cpu','parameter_count':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,'best_validation_macro_f1':best,'training_and_validation_seconds':train_seconds,'test_inference_seconds':test_seconds,'test_examples':len(rows['test']),'test':test_metrics,'thresholds':thresholds,'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__}}
    save(out/'results.json',result); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
