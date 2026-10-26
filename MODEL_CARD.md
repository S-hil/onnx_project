# Synthetic visibility softmax model

## Intended use

Portfolio demonstration of supervised training, source-group splitting, ONNX export, parity validation, browser inference, and transparent evaluation. Not a clinical tool. Not trained on surgical footage. No company affiliation is implied.

## Model

Multiclass logistic / softmax regression with 2 inputs and 3 outputs; six learned weights and three learned biases. Two training-set feature means and standard deviations are embedded with `log1p` preprocessing. Classes: clear, dark, low_detail. ONNX opset 13, float32, dynamic batch. All training is implemented in NumPy and exported using ONNX helpers; there is no pretrained backbone.

## Data and labels

540 generated 320×180 grayscale images from 180 independent procedural scene seeds. Each base scene has an undegraded, darkened, and blurred variant. Labels follow generation operations and are imperfect proxies for visibility. Seed-group split: 108 train / 36 validation / 36 test scenes. Dataset generation needs no downloaded images or private patient information.

## Evaluation

See `dist/reports/evaluation.json` for measured metrics, errors, model SHA-256, manifest hash, training configuration, and timing. Held-out synthetic accuracy and macro F1 are both about 98.1%. The fixed, untuned baseline has 88.9% accuracy and 88.6% macro F1. This advantage does not establish superiority over a tuned baseline or generalization beyond the generator. Reported CPU timings are environment-specific.

## Deployment

The original ONNX heuristic and learned model both run locally in ONNX Runtime Web / WASM. Threshold controls adjust the heuristic only. Learned softmax outputs are uncalibrated. No rejection threshold or out-of-distribution detector exists. Model predictions in the overlay are separate from rule-based warning colors.

## Limits

Synthetic-to-real domain shift, small held-out scene count, grayscale-only training, two-feature information bottleneck, single degradation per image, no temporal or clinical evaluation. Avoid interpreting clear predictions as evidence of a safe or unobstructed surgical view. Refit and evaluate on properly licensed, labeled target-domain data before any performance claims about that domain.

Validation-tuned rule comparison: thresholds 65 / 45 were selected from 198 candidates using validation macro F1 only. Their held-out synthetic accuracy is 89.8% and macro F1 is 89.9%, compared with 98.1% for the learned classifier. This remains a small synthetic experiment, not real-world superiority evidence.
