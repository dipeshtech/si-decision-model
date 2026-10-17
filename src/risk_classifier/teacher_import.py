"""Normalize teacher annotations without treating teacher labels as human gold."""
import argparse
import hashlib
import json
from pathlib import Path
from .data import taxonomy, validate, write_jsonl
from .evaluation import probabilities


def normalize(rows, spec, codebook_hash, codebook_version):
    result = []
    for row in rows:
        # Field aliases are explicit. Missing privacy-preserving groups fail.
        required = ('example_id','participant_group','conversation_group','text','teacher_label')
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError('Teacher rows require example_id, participant_group, conversation_group, text and teacher_label')
        if row['teacher_label'] not in spec['labels'] + ['unknown']:
            raise ValueError('Teacher label outside taxonomy')
        human = row.get('human_label')
        if human is not None and human not in spec['labels'] + ['unknown']:
            raise ValueError('Human label outside taxonomy')
        target = human if human is not None else row['teacher_label']
        normalized = {'id':row['example_id'], 'subject_id':row['participant_group'],
                      'conversation_id':row['conversation_group'], 'text':row['text'],
                      'label':target, 'target_label':target, 'teacher_label':row['teacher_label'],
                      'human_label':human, 'label_source':'human' if human is not None else 'teacher',
                      'source_type':row.get('source_type','real'),
                      'codebook_version':codebook_version,'codebook_sha256':codebook_hash,
                      'review_needed':target == 'unknown' or (human is not None and human != row['teacher_label']),
                      'evidence':row.get('evidence',{})}
        for key in ('template_family','duplicate_cluster','teacher_model','teacher_prompt_version','timestamp','context_cutoff'):
            if key in row: normalized[key] = row[key]
        if 'teacher_probabilities' in row:
            p = probabilities(row['teacher_probabilities'],len(spec['labels']))
            if abs(p.sum()-1) > 1e-6 or not row.get('probability_provenance'):
                raise ValueError('Soft targets require normalized probabilities and provenance')
            normalized['teacher_probabilities'] = p.tolist()
            normalized['probability_provenance'] = row['probability_provenance']
        if normalized['source_type'] == 'synthetic' and not normalized.get('template_family'):
            raise ValueError('Synthetic inputs require template_family')
        result.append(normalized)
    return validate(result,spec)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--taxonomy',required=True)
    parser.add_argument('--codebook',required=True)
    parser.add_argument('--codebook-version',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    spec=taxonomy(args.taxonomy)
    content=Path(args.codebook).read_bytes()
    digest=hashlib.sha256(content).hexdigest()
    rows=[json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    out=Path(args.output)
    if out.exists(): parser.error('Output exists; choose a new path')
    normalized=normalize(rows,spec,digest,args.codebook_version)
    write_jsonl(out,normalized)
    manifest={'format':'teacher-import-v1','records':len(normalized),'codebook_version':args.codebook_version,
              'codebook_sha256':digest,'label_order':spec['labels'],
              'teacher_labels_are_human_gold':False,
              'codebook_storage':'Keep one governed copy outside Git; referenced by version/hash per row.'}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'records':len(normalized),'output':str(out),'teacher_labels_are_human_gold':False}))

if __name__ == '__main__': main()
