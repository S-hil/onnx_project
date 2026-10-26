"""Reproducible synthetic experiment. Not surgical data or clinical validation."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['clear', 'dark', 'low_detail']


def metrics(image):
    g = image.astype(np.float64)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return [float(g.mean()), float(lap.var())]


def scene(seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:180, :320]
    image = np.full((180, 320), rng.uniform(60, 185))
    for _ in range(8):
        image += rng.uniform(2, 22) * np.sin(x*rng.uniform(.02, .65) + y*rng.uniform(.02, .65) + rng.uniform(0, 6))
    for _ in range(12):
        xx, yy = rng.integers(0, 280), rng.integers(0, 145)
        image[yy:yy+int(rng.integers(8, 35)), xx:xx+int(rng.integers(8, 40))] += rng.uniform(-65, 65)
    image = np.clip(image + rng.normal(0, rng.uniform(.1, 2), image.shape), 0, 255)
    return image, rng


def variants(seed):
    image, rng = scene(seed)
    fy = np.fft.fftfreq(180)[:, None]
    fx = np.fft.fftfreq(320)[None, :]
    sigma = rng.uniform(1.5, 5)
    blur = np.fft.ifft2(np.fft.fft2(image) * np.exp(-2*np.pi**2*sigma**2*(fx*fx+fy*fy))).real
    dark = image * rng.uniform(.12, .48)
    return [np.rint(np.clip(a, 0, 255)).astype(np.uint8) for a in (image, dark, blur)]


def dataset():
    # Split scene IDs first; all related variants remain in the same partition.
    ids = np.random.default_rng(20261026).permutation(180)
    split = {int(s): ('train' if i < 108 else 'validation' if i < 144 else 'test') for i, s in enumerate(ids)}
    records = []
    for seed in range(180):
        for label, image in enumerate(variants(seed)):
            b, d = metrics(image)
            records.append(dict(scene_id=seed, split=split[seed], label=label, brightness=b, detail=d))
    return records


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def evaluate(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per_class = []
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum()-tp); fn = int(cm[i].sum()-tp)
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        per_class.append(dict(label=name, precision=precision, recall=recall, f1=f1, support=int(cm[i].sum())))
    return dict(accuracy=float(np.mean(y == pred)), macro_f1=float(np.mean([r['f1'] for r in per_class])), confusion_matrix=cm.tolist(), per_class=per_class)


def export_model(mean, std, weights, bias, path):
    arrays = dict(offset=np.ones(2, np.float32), mean=mean.astype(np.float32), std=std.astype(np.float32), weights=weights.astype(np.float32), bias=bias.astype(np.float32))
    nodes = [helper.make_node('Add', ['metrics', 'offset'], ['positive']), helper.make_node('Log', ['positive'], ['logged']), helper.make_node('Sub', ['logged', 'mean'], ['centered']), helper.make_node('Div', ['centered', 'std'], ['features']), helper.make_node('MatMul', ['features', 'weights'], ['linear']), helper.make_node('Add', ['linear', 'bias'], ['logits']), helper.make_node('Softmax', ['logits'], ['probabilities'], axis=1)]
    graph = helper.make_graph(nodes, 'synthetic_visibility_softmax_v1', [helper.make_tensor_value_info('metrics', TensorProto.FLOAT, ['N', 2])], [helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, ['N', 3])], [numpy_helper.from_array(a, name) for name, a in arrays.items()])
    model = helper.make_model(graph, producer_name='camera-quality-monitor', opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    model.doc_string = 'Learned softmax regression on synthetic grayscale scene features. Class order: clear,dark,low_detail. No surgical validation.'
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = dataset()
    parts = {}
    for name in ('train', 'validation', 'test'):
        subset = [r for r in records if r['split'] == name]
        parts[name] = (np.array([[r['brightness'], r['detail']] for r in subset]), np.array([r['label'] for r in subset]))
    train, y = parts['train']
    mean = np.log1p(train).mean(axis=0)
    std = np.log1p(train).std(axis=0)
    X = (np.log1p(train)-mean)/std
    Xv = (np.log1p(parts['validation'][0])-mean)/std
    one_hot = np.eye(3)[y]
    trials = []
    best = None
    # Hyperparameter selection uses validation only. Fixed epoch budget, deterministic zero initialization.
    for l2 in (0., .01, .1):
        W = np.zeros((2, 3)); b = np.zeros(3)
        history = []
        for epoch in range(2000):
            prob = softmax(X@W+b)
            err = (prob-one_hot)/len(y)
            W -= .08*(X.T@err+l2*W)
            b -= .08*err.sum(axis=0)
            if epoch % 200 == 0 or epoch == 1999:
                history.append(dict(epoch=epoch+1, cross_entropy=float(-np.log(prob[np.arange(len(y)), y]+1e-12).mean())))
        val = evaluate(parts['validation'][1], softmax(Xv@W+b).argmax(axis=1))
        trials.append(dict(l2=l2, validation_macro_f1=val['macro_f1']))
        if best is None or val['macro_f1'] > best[0]:
            best = (val['macro_f1'], l2, W.copy(), b.copy(), history)
    _, chosen, W, b, history = best
    model_path = out_dir/'visibility-trained.onnx'
    export_model(mean, std, W, b, model_path)
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    features, truth = parts['test']
    probs = session.run(None, {'metrics': features.astype(np.float32)})[0]
    reference = softmax(((np.log1p(features)-mean)/std)@W+b)
    parity = float(np.max(np.abs(probs-reference)))
    assert parity < 1e-5
    predicted = probs.argmax(axis=1)
    # Fixed existing thresholds, not optimized using this experiment's test partition.
    baseline = np.where(features[:, 0] < 45, 1, np.where(features[:, 1] < 60, 2, 0))
    # Fairer comparator: select rule thresholds using validation only.
    val_features, val_truth = parts['validation']
    selected_rule = None
    for tb in range(15, 101, 5):
        for td in (5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300):
            val_pred = np.where(val_features[:, 0] < tb, 1, np.where(val_features[:, 1] < td, 2, 0))
            f1 = evaluate(val_truth, val_pred)['macro_f1']
            if selected_rule is None or f1 > selected_rule[0]:
                selected_rule = (f1, tb, td)
    val_f1, tuned_tb, tuned_td = selected_rule
    tuned_pred = np.where(features[:, 0] < tuned_tb, 1, np.where(features[:, 1] < tuned_td, 2, 0))
    single = features[:1].astype(np.float32)
    for _ in range(20): session.run(None, {'metrics': single})
    durations=[]
    for _ in range(200):
        start=time.perf_counter(); session.run(None, {'metrics': single}); durations.append((time.perf_counter()-start)*1000)
    errors=[]
    test_records=[r for r in records if r['split']=='test']
    for r, prediction, base, p in zip(test_records, predicted, baseline, probs):
        if prediction != r['label'] or base != r['label']:
            errors.append(dict(scene_id=r['scene_id'], truth=CLASSES[r['label']], learned=CLASSES[int(prediction)], baseline=CLASSES[int(base)], brightness=r['brightness'], detail=r['detail'], probabilities=p.tolist()))
    digest=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    report=dict(dataset='synthetic-procedural-grayscale-v1', warning='Synthetic held-out scenes only. Not surgical footage, not webcam validation, and not a clinical performance estimate.', seed=20261026, classes=CLASSES, samples={k:len(v[1]) for k,v in parts.items()}, scene_groups=dict(train=108, validation=36, test=36), split_unit='base scene seed; all transformations share the same split', features=['mean weighted RGB proxy', 'Laplacian variance'], preprocessing='log1p of both metrics, standardized with training-set mean/std only; embedded in ONNX', training=dict(algorithm='multiclass softmax regression', epochs=2000, learning_rate=.08, selected_l2=chosen, trials=trials, history=history), baseline_thresholds=dict(brightness=45, detail=60, policy='dark priority, else low_detail, else clear'), learned=evaluate(truth,predicted), baseline=evaluate(truth,baseline), parity_max_abs_error=parity, timing=dict(provider='Python ONNX Runtime CPU', platform=platform.platform(), sample_batch_size=1, warmup_runs=20, measured_runs=200, inference_p50_ms=float(np.percentile(durations,50)), inference_p95_ms=float(np.percentile(durations,95)), excludes='image generation, preprocessing outside ONNX, browser WASM, acquisition and rendering'), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), dataset_manifest_sha256=digest, errors=errors)
    report['tuned_baseline'] = evaluate(truth, tuned_pred)
    report['tuned_baseline_selection'] = dict(brightness=tuned_tb, detail=tuned_td, validation_macro_f1=val_f1, candidates=18*11, policy='dark priority; validation selection only')
    (out_dir/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    (out_dir/'dataset-manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    (out_dir/'learned-reference.json').write_text(json.dumps(dict(mean=mean.tolist(), std=std.tolist(), weights=W.tolist(), bias=b.tolist(), probes=features[:8].tolist(), expected=probs[:8].tolist()),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('samples','learned','baseline','parity_max_abs_error')},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out', type=Path, default=ROOT/'dist/reports');args=parser.parse_args();run(args.out)
