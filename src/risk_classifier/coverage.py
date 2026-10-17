"""Audit declared codebook coverage; counts never establish semantic correctness."""
import argparse
import json
from pathlib import Path

KINDS = {'positive','negative','boundary','context'}


def report(manifest, rows):
    rules = {r['rule_id']: r for r in manifest['rules']}
    if len(rules) != len(manifest['rules']):
        raise ValueError('Duplicate rule IDs')
    if any(not isinstance(row.get('template_family'),str) or not row['template_family'] for row in rows):
        raise ValueError('Every coverage row requires a template family')
    train_families = {r['template_family'] for r in rows if r.get('split') == 'train'}
    challenge_families = {r['template_family'] for r in rows if r.get('split') == 'challenge'}
    unknown = sorted({x for row in rows for x in row.get('rule_ids',[]) if x not in rules})
    findings = {}
    for rule_id in rules:
        matching = [r for r in rows if rule_id in r.get('rule_ids',[])]
        train = {r['template_family'] for r in matching if r.get('split') == 'train'}
        challenge = {r['template_family'] for r in matching if r.get('split') == 'challenge'}
        counts = {kind:sum(r.get('case_type') == kind for r in matching) for kind in sorted(KINDS)}
        gaps = [f'fewer_than_two_{k}' for k,v in counts.items() if v < 2]
        if not challenge: gaps.append('missing_challenge_family')
        if train & challenge: gaps.append('challenge_family_overlaps_train')
        reviewed = sum(r.get('reviewed') is True for r in matching)
        if reviewed == 0: gaps.append('no_reviewed_examples')
        if any(r.get('unresolved',True) for r in matching): gaps.append('unresolved_examples')
        if not any(len(r.get('rule_ids',[])) > 1 for r in matching): gaps.append('missing_rule_interaction')
        findings[rule_id] = {'example_count':len(matching),'case_counts':counts,
                             'reviewed_count':reviewed,'unresolved_count':sum(r.get('unresolved',True) is True for r in matching),
                             'challenge_family_count':len(challenge),'gaps':gaps}
    return {'scope':manifest.get('scope','unspecified'),'semantic_coverage_proven':False,
            'unknown_rule_ids':unknown,'rules':findings,'global_family_overlap':sorted(train_families & challenge_families),
            'warning':'Counts audit metadata only. Sample rules cannot establish coverage of an unavailable full codebook.'}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--examples',required=True)
    args=p.parse_args(argv)
    manifest=json.loads(Path(args.manifest).read_text())
    rows=[json.loads(line) for line in Path(args.examples).read_text().splitlines() if line.strip()]
    result=report(manifest,rows)
    print(json.dumps(result,indent=2))
    return result

if __name__ == '__main__': main()
