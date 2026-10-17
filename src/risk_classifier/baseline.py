"""Genuinely trained inexpensive CPU baseline; JSON parameters, no pickle loading."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import taxonomy, load_records, split_records, write_jsonl
from .synthetic import generate
from .evaluation import select_thresholds, evaluate


def fit(rows, labels, options=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    settings={'max_features':20000,'min_df':1,'ngram_max':2,'regularization_c':1.0,'max_iter':500,'seed':42}
    if options:
        if set(options)-set(settings): raise ValueError('Unknown baseline option')
        settings.update(options)
    for key in ('max_features','min_df','ngram_max','max_iter'):
        if type(settings[key]) is not int or settings[key] < 1: raise ValueError('Positive integer baseline options required')
    if not isinstance(settings['regularization_c'],(int,float)) or not np.isfinite(settings['regularization_c']) or settings['regularization_c'] <= 0:
        raise ValueError('regularization_c must be positive and finite')
    if type(settings['seed']) is not int: raise ValueError('Seed must be an integer')
    known = [r for r in rows if r['label'] != 'unknown']
    if set(r['label'] for r in known) != set(labels):
        raise ValueError('Training split must contain all configured classes')
    vectorizer = TfidfVectorizer(ngram_range=(1,settings["ngram_max"]), min_df=settings["min_df"], sublinear_tf=True, max_features=settings["max_features"])
    features = vectorizer.fit_transform([r['text'] for r in known])
    model = LogisticRegression(max_iter=settings['max_iter'], C=settings['regularization_c'], class_weight='balanced', random_state=settings['seed'])
    model.fit(features, [r['label'] for r in known])
    return {'format':'tfidf-logistic-json-v1', 'labels':labels, 'classes':model.classes_.tolist(),
            'vocabulary':{k:int(v) for k,v in vectorizer.vocabulary_.items()},
            'idf':vectorizer.idf_.tolist(), 'coef':model.coef_.tolist(),
            'intercept':model.intercept_.tolist(),
            'training_records':len(known), 'synthetic_only':all(r.get('source_type') == 'synthetic' for r in known),
            'training_parameters':settings, 'feature_parameters':{'ngram_range':[1,settings['ngram_max']],'sublinear_tf':True,'max_features':settings['max_features'],'min_df':settings['min_df']},
            'warning':'Synthetic demonstration only. Not Qwen, Kev, a clinical classifier or a safety assessment.'}


def predict(model, rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if 'taxonomy' in model:
        digest = __import__('hashlib').sha256(json.dumps(model['taxonomy'],sort_keys=True).encode()).hexdigest()
        if digest != model.get('taxonomy_sha256') or model['taxonomy']['labels'] != model['labels']:
            raise ValueError('Model taxonomy integrity mismatch')
    if not np.isfinite(np.asarray(model['coef'],dtype=float)).all() or not np.isfinite(np.asarray(model['intercept'],dtype=float)).all():
        raise ValueError('Nonfinite model parameters')
    if model.get('format') != 'tfidf-logistic-json-v1':
        raise ValueError('Unsupported model format')
    features=model.get('feature_parameters',{})
    if set(features) != {'ngram_range','sublinear_tf','max_features','min_df'} or features['sublinear_tf'] is not True:
        raise ValueError('Unexpected feature parameters')
    ngrams=features['ngram_range']
    if not isinstance(ngrams,list) or len(ngrams)!=2 or ngrams[0]!=1 or type(ngrams[1]) is not int or ngrams[1]<1:
        raise ValueError('Invalid ngram range')
    vectorizer = TfidfVectorizer(ngram_range=tuple(ngrams), sublinear_tf=True, vocabulary=model['vocabulary'])
    vectorizer.idf_ = np.asarray(model['idf'])
    x = vectorizer.transform([r['text'] for r in rows])
    logits = x @ np.asarray(model['coef']).T + np.asarray(model['intercept'])
    if logits.shape[1] != len(model['classes']):
        raise ValueError('Unsupported binary model or malformed coefficients')
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    order = [model['classes'].index(label) for label in model['labels']]
    return [{'id':row['id'], 'probabilities':p[order].tolist()} for row,p in zip(rows,probs)]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train an explicit synthetic-only CPU demonstration')
    parser.add_argument('--taxonomy', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--settings',help='JSON baseline hyperparameters; unknown keys fail')
    args = parser.parse_args(argv)
    options=json.loads(Path(args.settings).read_text()) if args.settings else {}
    spec = taxonomy(args.taxonomy)
    if spec['labels'] != ['NO_RISK','SI3','SI4','SI5']:
        parser.error('Synthetic demo requires the four named labels in fixed order')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error('Refusing to overwrite nonempty output directory')
    out.mkdir(parents=True, exist_ok=True)
    rows = generate(out/'synthetic.jsonl', args.taxonomy)
    splits = split_records(rows, seed=options.get('seed',42), stratify=True)
    for name, records in splits.items():
        write_jsonl(out/f'{name}.jsonl', records)
    model = fit(splits['train'], spec['labels'],options)
    model['taxonomy'] = spec
    model['taxonomy_sha256'] = __import__('hashlib').sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    (out/'baseline.json').write_text(json.dumps(model))
    # Test safe serialization round-trip before publishing artifact.
    restored = json.loads((out/'baseline.json').read_text())
    calibration_predictions = predict(restored, splits['calibration'])
    thresholds = select_thresholds(splits['calibration'], calibration_predictions, spec['labels'])
    predictions = predict(restored, splits['test'])
    write_jsonl(out/'test-predictions.jsonl', predictions)
    metrics = evaluate(splits['test'], predictions, spec['labels'], thresholds, approved=True)
    metrics['warning'] = 'SYNTHETIC PLUMBING ONLY: underlying statement families are disjoint, but only 64 illustrative statement templates exist; limited context coverage and synthetic wording make these metrics non-generalizable. No clinical validity.'
    (out/'synthetic-evaluation.json').write_text(json.dumps(metrics,indent=2))
    (out/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    print(json.dumps({'trained':True,'model':'CPU TF-IDF/logistic regression','records':len(rows),
                      'split_sizes':{k:len(v) for k,v in splits.items()},'artifact':str(out/'baseline.json'),
                      'synthetic_only':True,'qwen_trained':False},indent=2))

if __name__ == '__main__':
    main()
