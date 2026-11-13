"""Published multiclass paths and metrics; labels and training stay separate."""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
OLD_CODE = ROOT.parent
sys.path.insert(0, str(OLD_CODE))
DATASETS = {
    'banking77': {'results': OLD_CODE / 'results/imbalance_all', 'classes': 77},
    'clinc150': {'results': OLD_CODE / 'clinc150/results', 'classes': 150},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item()) + '\n')
    temporary.replace(path)


def preserve(capture=False):
    """Validate the two frozen label caches supplied with this publication."""
    manifest = json.loads((ROOT / 'labels/manifest.json').read_text())
    for dataset in DATASETS:
        assert sha(ROOT / 'labels' / f'{dataset}.npz') == manifest[dataset]['labels_sha256']
    return len(DATASETS)


def class_metrics(tp, predicted, support):
    import numpy as np
    tp, predicted, support = [np.asarray(x, dtype=float) for x in (tp, predicted, support)]
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, predicted + support, out=np.zeros_like(tp), where=predicted + support > 0)
    return precision, recall, f1


def scalar_metrics(tp, predicted, support):
    import numpy as np
    tp, predicted, support = [np.asarray(x, dtype=float) for x in (tp, predicted, support)]
    p, r, f = class_metrics(tp, predicted, support)
    weights = support / support.sum()
    accuracy = float(tp.sum() / support.sum())
    return {'accuracy': accuracy, 'micro_precision': accuracy, 'micro_recall': accuracy, 'micro_f1': accuracy,
            'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
            'weighted_precision': float(weights @ p), 'weighted_recall': float(weights @ r), 'weighted_f1': float(weights @ f)}


def from_predictions(truth, prediction, k):
    import numpy as np
    support = np.bincount(truth, minlength=k)
    predicted = np.bincount(prediction, minlength=k)
    tp = np.bincount(truth[truth == prediction], minlength=k)
    return scalar_metrics(tp, predicted, support), tp, predicted, support
