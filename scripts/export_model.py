"""Export an interpretable visibility scoring graph; no learned parameters."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

root = Path(__file__).resolve().parents[1]
nodes = [
    helper.make_node('Div', ['metrics', 'thresholds'], ['ratios']),
    helper.make_node('ReduceMin', ['ratios'], ['weakest'], axes=[1], keepdims=0),
    helper.make_node('Clip', ['weakest', 'zero', 'one'], ['normalized']),
    helper.make_node('Mul', ['normalized', 'hundred'], ['scores']),
]
graph = helper.make_graph(nodes, 'visibility_baseline_v1', [
    helper.make_tensor_value_info('metrics', TensorProto.FLOAT, ['N', 2]),
    helper.make_tensor_value_info('thresholds', TensorProto.FLOAT, [2]),
], [helper.make_tensor_value_info('scores', TensorProto.FLOAT, ['N'])], [
    helper.make_tensor('zero', TensorProto.FLOAT, [], [0]),
    helper.make_tensor('one', TensorProto.FLOAT, [], [1]),
    helper.make_tensor('hundred', TensorProto.FLOAT, [], [100]),
])
model = helper.make_model(graph, producer_name='clarity-portfolio',
    opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
model.doc_string = ('Deterministic visibility baseline: 100 * clip(min(B/Tb,D/Td),0,1). '
                    'Not trained; not clinically validated. Inputs must be finite, nonnegative; thresholds positive.')
onnx.checker.check_model(model)
out = root / 'dist/models/visibility-baseline.onnx'
onnx.save(model, out)
print(f'Validated and exported {out.name} ({out.stat().st_size} bytes)')
