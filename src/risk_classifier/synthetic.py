"""Non-operational synthetic workflow fixtures, NEVER a clinical training dataset."""
import hashlib
import json
from pathlib import Path
from .data import write_jsonl

LABELS = ['NO_RISK', 'SI3', 'SI4', 'SI5']
CONTEXTS = ['At a check-in', 'During a support conversation', 'In a private reflection', 'After a difficult week', 'In response to a current-feelings question']


def generate(output, taxonomy_path):
    spec_bytes = Path(taxonomy_path).read_bytes()
    codebook_hash = hashlib.sha256(spec_bytes).hexdigest()
    rows = []
    patterns = json.loads(Path(__file__).with_name('synthetic_patterns.json').read_text())
    for number in range(800):
        label = LABELS[number % len(LABELS)]
        family = (number // len(LABELS)) % len(patterns[label])
        variant = number // (len(LABELS) * len(patterns[label]))
        text = f"{CONTEXTS[variant % len(CONTEXTS)]}: {patterns[label][family]} Synthetic scenario {number:04}."
        rows.append({'id': f'demo-{number:04}', 'subject_id': f'synthetic-person-{number:04}',
                     'conversation_id': f'synthetic-conversation-{number:04}',
                     'template_family': f'family-{label}-{family:02}', 'text': text, 'label': label,
                     'teacher_label': None, 'human_label': None, 'target_label': label,
                     'label_source': 'synthetic_rule', 'source_type': 'synthetic',
                     'codebook_version': 'custom-si-draft-v0', 'codebook_sha256': codebook_hash,
                     'evidence': ({'plan':'false','means':'false','timing':'unknown'} if label == 'SI3'
                                  else {'plan':'true','means':'true','timing':'true'} if label == 'SI5'
                                  else {'plan':'unknown','means':'unknown','timing':'unknown'}),
                     'review_needed': True})
    write_jsonl(output, rows)
    return rows
