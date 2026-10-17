"""Local research inference. Text is never logged; operational flag always abstains."""
import argparse, json, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from train_cpu import clean

def predict(text, checkpoint, max_length=128):
    if not isinstance(text,str) or not text.strip():
        return {'flag':'INSUFFICIENT_EVIDENCE','human_review_required':True,'reason_code':'EMPTY_INPUT'}
    tokenizer=BertTokenizerFast.from_pretrained(checkpoint,local_files_only=True)
    tokens=tokenizer(clean(text),return_tensors='pt',truncation=False)
    if tokens['input_ids'].shape[1]>max_length:
        return {'flag':'INSUFFICIENT_EVIDENCE','human_review_required':True,'reason_code':'OVERLENGTH_REQUIRES_REVIEWED_WINDOWS','input_tokens':tokens['input_ids'].shape[1],'max_input_tokens':max_length}
    model=BertForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,use_safetensors=True).eval()
    torch.set_num_threads(4)
    with torch.inference_mode(): probabilities=model(**tokens).logits.softmax(-1)[0].tolist()
    return {'flag':'INSUFFICIENT_EVIDENCE','human_review_required':True,'reason_code':'SYNTHETIC_ONLY_UNAPPROVED_CODEBOOK','research_candidate_label':model.config.id2label[max(range(4),key=probabilities.__getitem__)],'research_probabilities':{model.config.id2label[i]:p for i,p in enumerate(probabilities)}}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--input-json',required=True,type=Path); p.add_argument('--checkpoint',type=Path,default=Path(__file__).parent/'output/checkpoint'); args=p.parse_args()
    record=json.loads(args.input_json.read_text()); print(json.dumps(predict(record['text'],args.checkpoint)))
