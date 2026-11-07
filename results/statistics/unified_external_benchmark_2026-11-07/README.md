# Unified six-model computation and CPU timing

Measured on 2026-11-07. These observations replace the mixed-device efficiency columns in manuscript Table 5(E), without changing classification results or overwriting the earlier raw records.

All models use FP32, batch one, CPU i5-14600KF, four intra-op threads and one inter-op thread. Each of five independent processes randomizes model order reproducibly and performs 50 warm-ups followed by 200 timed forwards per model. Mean latency and its reported sample standard deviation are calculated from the five process means. The 6,000 individual observations remain in raw_rep0.csv through raw_rep4.csv; pooled iteration SDs and percentiles are stored separately and are not substituted for the five-mean SD.

GFLOPs consistently count convolution, linear and matrix-multiplication operators at two scalar floating-point operations per MAC, including the declared in-model input adapters. Counted operator totals and model/checkpoint hashes are retained in details_rep0.json through details_rep4.json. All six counts were checked against the independently retained per-operator records. HSMF-Net is 0.635870016 GFLOPs under this common rule; 0.3294 is not used in the unified external comparison.

protocol.json records the fixed shapes, execution mode, model order, checkpoint rule and timing boundary. This is a forward-only benchmark on prepared tensors; it does not include acquisition, file decoding or external preprocessing. The benchmark does not train, evaluate test labels or select parameters based on classification scores.

summary.csv contains computational and timing summaries; comparison_table.csv combines these measurements with the unchanged classification summary used in the manuscript. verification.json records the arithmetic and document-consistency checks.
