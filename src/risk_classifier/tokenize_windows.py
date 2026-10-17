"""Automatic local fast-tokenizer preprocessing without downloads or remote code."""
import argparse
import hashlib
import json
from pathlib import Path
from .data import assert_disjoint, write_jsonl
from .windows import window_records


def prepare(splits, tokenizer, tokenizer_revision, max_input_tokens, overlap, reserved_tokens):
    if not getattr(tokenizer,'is_fast',False):
        raise ValueError('A local fast tokenizer with offset mappings is required')
    limit=getattr(tokenizer,'model_max_length',None)
    if not isinstance(limit,int) or not 0 < limit < 1_000_000:
        raise ValueError('Tokenizer has no verified finite model limit; provide local model config')
    if max_input_tokens > limit:
        raise ValueError('Requested window exceeds the tokenizer model limit')
    if reserved_tokens < tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError('Reserve must include at least tokenizer special-token overhead')
    assert_disjoint(splits)
    prepared=[]
    for split, rows in splits.items():
        for row in rows:
            encoded=tokenizer(row['text'],add_special_tokens=False,truncation=False,return_offsets_mapping=True)
            offsets=[list(pair) for pair in encoded['offset_mapping'] if pair[1] > pair[0]]
            current=dict(row,split=split,token_offsets=offsets,tokenizer_revision=tokenizer_revision)
            prepared.append(current)
    result=window_records(prepared,max_input_tokens,overlap,reserved_tokens)
    # Re-tokenization can change tokens at a cut. Verify the actual window too.
    for window in result:
        actual=tokenizer(window['text'],add_special_tokens=False,truncation=False)['input_ids']
        if len(actual)+reserved_tokens > max_input_tokens:
            raise ValueError('Re-tokenized window exceeds budget; revise boundary settings')
        window['verified_content_tokens']=len(actual)
    return result


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--splits',required=True,help='Directory containing train/validation/calibration/test.jsonl')
    p.add_argument('--tokenizer',required=True,help='Explicit local tokenizer directory; no downloads')
    p.add_argument('--tokenizer-revision',required=True,help='Verified source revision for this local tokenizer')
    p.add_argument('--output',required=True)
    p.add_argument('--max-input-tokens',type=int,required=True)
    p.add_argument('--overlap',type=int,default=32)
    p.add_argument('--reserved-tokens',type=int,default=32)
    args=p.parse_args(argv)
    from transformers import AutoTokenizer
    local=Path(args.tokenizer).resolve(strict=True)
    if not local.is_dir():p.error('Tokenizer must be a local directory')
    tokenizer=AutoTokenizer.from_pretrained(str(local),local_files_only=True,trust_remote_code=False,use_fast=True)
    config_file=local/'config.json'
    if config_file.is_file():
        model_config=json.loads(config_file.read_text())
        positions=model_config.get('max_position_embeddings')
        if isinstance(positions,int) and 0 < positions < 1_000_000:
            tokenizer.model_max_length=min(tokenizer.model_max_length,positions)
    splits={name:[json.loads(line) for line in (Path(args.splits)/f'{name}.jsonl').read_text().splitlines() if line.strip()] for name in ['train','validation','calibration','test']}
    windows=prepare(splits,tokenizer,args.tokenizer_revision,args.max_input_tokens,args.overlap,args.reserved_tokens)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):p.error('Output directory must be new or empty')
    for name in splits:
        write_jsonl(out/f'{name}.jsonl',[r for r in windows if r['split']==name])
    fingerprints={file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in local.iterdir() if file.is_file() and (file.name.startswith('tokenizer') or file.name in ('vocab.txt','special_tokens_map.json'))}
    (out/'manifest.json').write_text(json.dumps({'tokenizer_revision':args.tokenizer_revision,'local_tokenizer_file_sha256':fingerprints,'windows':len(windows),'labels':'unknown','raw_splits_unchanged':True},indent=2))
    print(json.dumps({'windows':len(windows),'labels':'unknown','output':str(out)}))

if __name__ == '__main__':main()
