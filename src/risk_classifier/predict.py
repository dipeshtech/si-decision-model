"""Local flags-only baseline inference; draft models abstain by default."""
import argparse
import json
from pathlib import Path
from .baseline import predict
from .evaluation import flag
from .kev import require_approved


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--thresholds',required=True)
    p.add_argument('--input',required=True,help='JSONL with id and text; keep private')
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    model=json.loads(Path(args.model).read_text())
    thresholds=json.loads(Path(args.thresholds).read_text())
    rows=[json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    try:
        require_approved(model['taxonomy'])
        approved=not model.get('synthetic_only',True)
    except (ValueError,KeyError):
        approved=False
    predictions=predict(model,rows)
    outputs=[{'id':r['id'],**flag(r['probabilities'],model['labels'],thresholds,approved)} for r in predictions]
    out=Path(args.output)
    if out.exists(): p.error('Output exists; choose a new path')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(row)+'\n' for row in outputs))
    print(json.dumps({'records':len(outputs),'output':str(out),'synthetic_or_unapproved_abstains':not approved}))

if __name__ == '__main__': main()
