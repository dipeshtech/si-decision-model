"""Budgeted windows AFTER group splitting; no inherited transcript supervision.

Requires externally produced real-tokenizer character offsets. Does not guess a
model's token budget from words or download a tokenizer. Output labels are unknown.
"""
import argparse
import json
from pathlib import Path
from .data import assert_disjoint, write_jsonl

SPLITS={'train','validation','calibration','test'}


def window_records(records, max_input_tokens=2048, overlap=256, reserved_tokens=256):
    if any(type(x) is not int for x in (max_input_tokens,overlap,reserved_tokens)):
        raise ValueError('Token settings must be integers')
    capacity=max_input_tokens-reserved_tokens
    if max_input_tokens <= 0 or reserved_tokens < 0 or capacity <= 0 or not 0 <= overlap < capacity:
        raise ValueError('Require positive content capacity and overlap below that capacity')
    splits={name:[] for name in SPLITS}
    for row in records:
        if row.get('split') not in SPLITS:
            raise ValueError('Split original grouped records before creating windows')
        splits[row['split']].append(row)
    assert_disjoint(splits)
    output=[]
    for row in records:
        text=row['text']
        offsets=row.get('token_offsets')
        if not isinstance(offsets,list) or not offsets or not row.get('tokenizer_revision'):
            raise ValueError('Actual tokenizer offsets and pinned tokenizer_revision required')
        previous=0
        for pair in offsets:
            if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair):
                raise ValueError('Offsets must be integer [start,end] pairs')
            start,end=pair
            if not previous <= start < end <= len(text) or text[previous:start].strip():
                raise ValueError('Invalid offsets or uncovered non-whitespace text')
            previous=end
        if text[previous:].strip(): raise ValueError('Offsets silently truncate input text')
        stride=capacity-overlap
        for index, begin in enumerate(range(0,len(offsets),stride)):
            end=min(begin+capacity,len(offsets))
            char_start,char_end=offsets[begin][0],offsets[end-1][1]
            record={k:row[k] for k in ('subject_id','conversation_id','template_family','duplicate_cluster','split','source_type','tokenizer_revision') if k in row}
            record.update(id=f"{row['id']}::window-{index}",original_id=row['id'],
                          text=text[char_start:char_end],label='unknown',target_label='unknown',
                          label_source='unannotated_window',review_needed=True,
                          transcript_label_provenance=row.get('label','unknown'),
                          token_start=begin,token_end=end,char_start=char_start,char_end=char_end,
                          content_tokens=end-begin,reserved_tokens=reserved_tokens,
                          max_input_tokens=max_input_tokens,overlap_tokens=overlap,
                          boundary_policy='Token windows may cut turns; original role markers remain in text. Review context before labeling.')
            output.append(record)
            if end == len(offsets): break
    return output


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-input-tokens',type=int,default=2048)
    p.add_argument('--overlap',type=int,default=256)
    p.add_argument('--reserved-tokens',type=int,default=256)
    args=p.parse_args(argv)
    rows=[json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    windows=window_records(rows,args.max_input_tokens,args.overlap,args.reserved_tokens)
    if Path(args.output).exists(): p.error('Output exists; choose a new path')
    write_jsonl(args.output,windows)
    print(json.dumps({'windows':len(windows),'labels':'unknown','training_ready':False}))

if __name__ == '__main__': main()
