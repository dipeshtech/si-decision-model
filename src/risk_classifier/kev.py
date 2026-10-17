"""Source-verified Kev interchange; no weights, installs, downloads or training here."""
from pathlib import Path

KEV_COMMIT = '1f04cfa28e6a708485b28a9585010b9e436b776e'
BASE = 'Qwen/Qwen3.8-27B'
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def require_approved(spec):
    if spec.get('status') != 'approved' or spec.get('unresolved'):
        raise ValueError('Approve and resolve the annotation codebook before export or training')
    if set(spec.get('definitions', {})) != set(spec['labels']):
        raise ValueError('An approved definition is required for every label')
    if spec['mode'] != 'multilabel' and spec.get('no_risk_label') not in spec['labels']:
        raise ValueError('Exclusive/ordinal taxonomy needs an explicitly approved no-qualifying-content class')


def export_records(records, spec):
    require_approved(spec)
    exported, skipped = [], 0
    for row in records:
        if spec['mode'] == 'multilabel':
            questions = {}
            for label in spec['labels']:
                state = row['labels'][label]
                if state != 'unknown':
                    questions[label] = {'type': 'noul', 'instructions': spec['definitions'][label], 'label': state == 'true'}
            if not questions:
                skipped += 1
                continue
        else:
            if row['label'] == 'unknown':
                skipped += 1
                continue
            # choice preserves SI semantic names; ordinal mode does NOT imply
            # that clinical categories are equally spaced numerical outcomes.
            questions = {'severity': {'type': 'choice', 'instructions': spec['question'],
                                     'criteria': spec['definitions'], 'label': row['label']}}
        exported.append({'state': row['text'], 'questions': questions,
                         '_meta': {'id': row['id'], 'group_id': row['subject_id']}})
    if not exported:
        raise ValueError('No known training annotations remain')
    return exported, skipped


def train_plan(data_path, spec, options=None):
    """Produce an argv template for raw-Qwen LoRA, never execute it.

    The input path is canonical project data. Export the train split first; the
    template deliberately names the resulting Kev file, not the canonical data.
    """
    require_approved(spec)
    path = Path(data_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Dataset path must be a file')
    settings = {'epochs':1,'lr':2e-5,'head_lr':1e-4,'lora':16,'head_dim':256,'batch':1,'accum':8,
                'device':'cuda','dtype':'bf16','weights_dtype':'bf16','checkpointing':1,'max_state':2048,'seed':0}
    if options is not None:
        supported = set(settings) | {'base','base_revision','initialization'}
        if set(options) - supported:
            raise ValueError('Unsupported training configuration key')
        if options.get('base', BASE) != BASE or options.get('base_revision',BASE_REVISION) != BASE_REVISION:
            raise ValueError('This verified adapter only supports the pinned Qwen3.8-27B base')
        if options.get('initialization','raw_base_lora_new_head') != 'raw_base_lora_new_head':
            raise ValueError('Full-weight checkpoint initialization needs a separately verified configuration')
        settings.update({k:v for k,v in options.items() if k in settings})
    for key in ('epochs','lora','head_dim','batch','accum'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError('Positive integer training settings required')
    for key in ('lr','head_lr'):
        if not isinstance(settings[key], (int,float)) or not 0 < settings[key] < 1:
            raise ValueError('Learning rates must be in (0,1)')
    if type(settings['max_state']) is not int or not 384 <= settings['max_state'] <= 65536:
        raise ValueError('max_state must be an integer between 384 and 65536')
    if settings['device'] != 'cuda' or settings['dtype'] != 'bf16' or settings['weights_dtype'] != 'bf16':
        raise ValueError('Only reviewed CUDA bf16 plan is currently supported')
    if settings['checkpointing'] not in (0,1) or type(settings['seed']) is not int:
        raise ValueError('Invalid checkpointing or seed')
    argv = ['python','-m','kev.train','--data','artifacts/kev-train.jsonl','--base',BASE,
            '--base_revision',BASE_REVISION,'--out','checkpoints/si-experiment']
    for key,value in settings.items():
        argv.extend(['--'+key,str(value)])
    return {
        'executed': False,
        'initialization': 'raw Qwen base with a new pointer head; NOT released Kev-27B warm start',
        'upstream_commit': KEV_COMMIT,
        'argv_template': argv,
        'prerequisites': [
            'Pinned upstream checkout and reviewed dependencies; compatible CUDA GPU capacity.',
            'Export only the train split; use separate validation data for hyperparameter selection.',
            'Check tokenizer truncation and loss of relevant evidence; 2048 is an example, not validated.',
            '27B GPU fit has not been tested. No GPU resources or paid jobs are provisioned.',
            'Opt in by manually running the reviewed upstream command in the intended environment.',
            'Do not use full-weight released Kev-27B with ordinary LoRA init_from.',
        ],
        'full_weight_alternative': 'Released Kev-27B requires matching --full_ft 1 initialization, pinned checkpoint revision, matching head/config, and reviewed multi-GPU capacity. No unpinned warm-start command is generated.',
    }
