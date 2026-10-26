from pathlib import Path
import numpy as np
import onnxruntime as ort
root=Path(__file__).resolve().parents[1]
session=ort.InferenceSession(str(root/'dist/models/visibility-baseline.onnx'),providers=['CPUExecutionProvider'])
rng=np.random.default_rng(7)
metrics=rng.uniform(0,300,(1000,2)).astype(np.float32)
metrics[:4]=[[0,0],[45,60],[90,30],[22,150]]
for thresholds in ([45,60],[5,5],[150,300]):
    t=np.array(thresholds,dtype=np.float32)
    actual=session.run(None,{'metrics':metrics,'thresholds':t})[0]
    expected=100*np.clip(np.min(metrics/t,axis=1),0,1)
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-6)
print('ONNX parity passed: 3,000 comparisons including threshold boundaries.')
