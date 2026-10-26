"""Checks actual learned model artifacts, split integrity, and metric calculations."""
import importlib.util
import json
import unittest
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training', ROOT/'scripts/train_visibility.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((ROOT/'dist/reports/dataset-manifest.json').read_text())
        cls.ref = json.loads((ROOT/'dist/reports/learned-reference.json').read_text())
        cls.report = json.loads((ROOT/'dist/reports/evaluation.json').read_text())

    def test_related_scenes_never_cross_splits(self):
        seen = {}
        for r in self.records:
            seen.setdefault(r['scene_id'], set()).add(r['split'])
        self.assertEqual(len(seen), 180)
        self.assertTrue(all(len(s)==1 for s in seen.values()))
        self.assertEqual(len(self.records), 540)
        self.assertEqual({k:sum(r['split']==k for r in self.records) for k in ('train','validation','test')}, self.report['samples'])

    def test_scaling_is_fitted_only_on_training_data(self):
        x = np.array([[r['brightness'],r['detail']] for r in self.records if r['split']=='train'])
        np.testing.assert_allclose(np.log1p(x).mean(axis=0), self.ref['mean'],atol=1e-10)
        np.testing.assert_allclose(np.log1p(x).std(axis=0), self.ref['std'],atol=1e-10)

    def test_actual_onnx_probabilities_match_reference_and_report(self):
        session=ort.InferenceSession(str(ROOT/'dist/reports/visibility-trained.onnx'), providers=['CPUExecutionProvider'])
        records=[r for r in self.records if r['split']=='test']
        x=np.array([[r['brightness'],r['detail']] for r in records],dtype=np.float32)
        actual=session.run(None,{'metrics':x})[0]
        z=(np.log1p(x.astype(np.float64))-self.ref['mean'])/self.ref['std']
        expected=training.softmax(z@np.array(self.ref['weights'])+self.ref['bias'])
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(actual.sum(axis=1),1,atol=1e-6)
        result=training.evaluate(np.array([r['label'] for r in records]),actual.argmax(axis=1))
        self.assertEqual(result['confusion_matrix'],self.report['learned']['confusion_matrix'])
        self.assertAlmostEqual(result['macro_f1'],self.report['learned']['macro_f1'])

    def test_metrics_count_false_positives_and_false_negatives(self):
        result=training.evaluate(np.array([0,1,2]),np.array([0,0,2]))
        self.assertEqual(result['confusion_matrix'],[[1,0,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(result['accuracy'],2/3)
        self.assertEqual(result['per_class'][1]['recall'],0)

    def test_scene_generation_is_reproducible(self):
        a=training.variants(9); b=training.variants(9)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        self.assertLess(training.metrics(a[1])[0],training.metrics(a[0])[0])
        self.assertLess(training.metrics(a[2])[1],training.metrics(a[0])[1])

if __name__=='__main__': unittest.main()
