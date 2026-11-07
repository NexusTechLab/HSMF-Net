# Baseline specification

Local verification date: 5 November 2026. These records document the archived S1–S8 configurations and the available implementation; they do not constitute historical end-to-end reproduction or new training results.

## S16.1 Evidence and verification scope

Forty copied resolved configurations were compared with each original run's config_snapshot.yaml: all agree after YAML parsing. Within each model, the five seeds have identical configuration fields except seed and experiment-name labels (output location is excluded from this comparison). All 40 epoch histories contain 100 rows. The eight current model constructions reproduce the archived registered parameter totals; current state-dictionary names/shapes match each model's seed-42 checkpoint. Eight CPU synthetic-input forward passes return one four-class logit vector. No research images/cubes were inferred, no pretrained weights downloaded, no training launched, and no new latency or operation-count measurements made.

Sources: all40_config_fields.csv, config_comparison.json, architecture_audit.json, model_protocol_matrix.csv, S2_bottlenecks.csv and S2/S4/S5/S6_layer_shapes.csv. The original configurations are retained in configs/archived/. Parameter/schema agreement does not authenticate historical runtime inputs.

## S16.2 S2 architecture and matching rule

S2 is an internally implemented pure-3D HSI baseline. Here, “mirror” refers specifically to reuse of the eleven-block MobileNetV3-Small inverted-residual configuration table from the RGB backbone: channel widths, expansion widths, kernel choices, SE placement and block order are retained, while convolutions, depthwise convolutions, normalization, SE and global pooling are replaced by their 3D counterparts. It does not denote a parameter-matched mirror of the proposed hybrid HSI encoder. The input is $N\times1\times50\times24\times24$. A bias-free $3\times3\times3$ stem maps 1 to 16 channels with stride $(1,2,2)$ and padding $(1,1,1)$, followed by BatchNorm3d and Hardswish. Spectral depth remains 50 throughout the eleven bottlenecks; stride-two blocks downsample only the spatial axes. A final $1\times1\times1$ projection maps 96 to 576 channels, followed by 3D global average pooling and a $576\to1024\to4$ classifier with Hardswish and Dropout(0.35). S3 instead uses the hybrid encoder and a $576\to192\to4$ head. The complete S2 block table, kernel/stride/padding values, SE, residual rules and layer-level records are provided in Supplementary Baseline Specification S16. S2 and S3 therefore compare complete HSI configurations with different heads, initialization requests and parameter budgets.

| Block | Cin / expanded / Cout | DW kernel | DW stride (D,H,W) | Padding | SE | Activation | Residual | Output (N,C,D,H,W) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 / 16 / 16 | [3, 3, 3] | [1, 2, 2] | [1, 1, 1] | True | ReLU | False | [1, 16, 50, 6, 6] |
| 2 | 16 / 72 / 24 | [3, 3, 3] | [1, 2, 2] | [1, 1, 1] | False | ReLU | False | [1, 24, 50, 3, 3] |
| 3 | 24 / 88 / 24 | [3, 3, 3] | [1, 1, 1] | [1, 1, 1] | False | ReLU | True | [1, 24, 50, 3, 3] |
| 4 | 24 / 96 / 40 | [5, 5, 5] | [1, 2, 2] | [2, 2, 2] | True | Hardswish | False | [1, 40, 50, 2, 2] |
| 5 | 40 / 240 / 40 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | True | [1, 40, 50, 2, 2] |
| 6 | 40 / 240 / 40 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | True | [1, 40, 50, 2, 2] |
| 7 | 40 / 120 / 48 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | False | [1, 48, 50, 2, 2] |
| 8 | 48 / 144 / 48 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | True | [1, 48, 50, 2, 2] |
| 9 | 48 / 288 / 96 | [5, 5, 5] | [1, 2, 2] | [2, 2, 2] | True | Hardswish | False | [1, 96, 50, 1, 1] |
| 10 | 96 / 576 / 96 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | True | [1, 96, 50, 1, 1] |
| 11 | 96 / 576 / 96 | [5, 5, 5] | [1, 1, 1] | [2, 2, 2] | True | Hardswish | True | [1, 96, 50, 1, 1] |


All main-path convolutions are bias-free and followed by BatchNorm3d (epsilon 0.001, momentum 0.01). Each block optionally expands channels with a 1×1×1 convolution, applies a grouped depthwise cubic convolution with groups equal to expanded width, optionally applies SE3D, and projects with a 1×1×1 convolution without a following activation. Expansion and depthwise activations are listed above. Expansion is omitted when its width equals the input width. Pointwise convolutions use unit stride and zero padding. SE3D globally pools D/H/W, squeezes to make_divisible(expanded/4,8), applies a biased 1×1×1 convolution, ReLU, another biased 1×1×1 convolution and Hardsigmoid. Residual addition is used only for unit stride and equal input/output widths. The final 96→576 pointwise block uses BatchNorm3d and Hardswish. Spatial sizes progress 24→12 (stem)→6→3→2→1; the spectral axis remains 50 until AdaptiveAvgPool3d(1). The stem is a dense convolution; it is not called a depthwise convolution.

The matching rule is architectural reuse of the RGB MobileNetV3-Small table, not matching S3's spectral compression, spatial downsampling, classifier, parameters or operations. S2 has 1,753,124 parameters versus S3's 924,434; their classifier widths are 1024 and 192, respectively, and their pretraining requests differ.

## S16.3 Fusion baselines

For S4–S6, the final RGB and hybrid-HSI feature maps are independently globally averaged to 576-dimensional vectors and projected by biased Linear(576,256) layers followed by Hardswish. Denote these projected vectors by $u_r,u_h\in\mathbb{R}^{256}$. S4 concatenates them, $z_4=[u_r;u_h]\in\mathbb{R}^{512}$. S5 uses $g=\sigma(W_g[u_r;u_h]+b_g)\in(0,1)^{256}$ and $z_5=g\odot u_r+(1-g)\odot u_h$. Its gate is a single biased Linear(512,256) followed by sigmoid, with no hidden layer or gate dropout. S4 and S5 use classifier widths $512\to256\to4$ and $256\to256\to4$, respectively; both heads use Hardswish and Dropout(0.35). Thus, their comparison includes classifier-capacity differences.

S6 is termed modality-token self-attention fusion (Token-Attn). Its historical configuration key is cross_attention, but its operation is joint self-attention over two modality tokens, rather than separate RGB-query/HSI-key-value cross-attention. It forms $T=[u_r;u_h]+E\in\mathbb{R}^{2\times256}$ with a learnable modality embedding $E$. The same LayerNorm-transformed sequence supplies the query, key and value inputs to multi-head attention; separate learned projections produce the actual Q, K and V tensors. Four heads have 64 features each, and softmax acts over the two key tokens, giving a $2\times2$ attention matrix per head. Attention-weight and attention-output dropout are 0.1. Pre-normalized residual attention and a $256\to512\to256$ Hardswish feed-forward network with dropout 0.35 use separate learned tanh residual scales initialized to zero. A $512\to256\to2$ Hardswish–sigmoid gate produces one scalar per modality to combine the output-normalized updated token with its original projected vector. The two resulting vectors are concatenated and classified by $512\to256\to4$. Supplementary Baseline Specification S16 gives the full equations and initialization. This baseline performs neither pixel-wise nor wavelength-axis attention.

For S6, let X=stack(u_r,u_h), T=X+E and L=LN_attn(T). For each head h, learned biased linear maps form Q_h=L W_Q,h+b_Q,h, K_h=L W_K,h+b_K,h and V_h=L W_V,h+b_V,h. The attention rule is:

$$
A_h=\operatorname{softmax}_{\mathrm{key\ token}}(Q_hK_h^\top/\sqrt{64}),\qquad
M=\operatorname{Concat}_h(\operatorname{Drop}_{0.1}(A_h)V_h)W_O+b_O.
$$

Thus, the inputs to Q/K/V are the same normalized sequence, while projected Q/K/V need not be equal. Batch/head/token attention weights have shape N×4×2×2. Separate residual scales a and b implement:

$$
U=T+\tanh(a)\operatorname{Drop}_{0.1}(M),\qquad
V=U+\tanh(b)\operatorname{FFN}(\operatorname{LN}_{\mathrm{ffn}}(U)).
$$

The FFN is Linear(256,512)→Hardswish→Dropout(0.35)→Linear(512,256)→Dropout(0.35). E, a and b initialize to zero. A biased Linear(512,256)→Hardswish→Linear(256,2)→sigmoid gate consumes the concatenated updated tokens, producing g_r and g_h. Its last linear weight initializes to zero and bias to −2. For i in {r,h}, s_i=g_i LN_out(V_i)+(1−g_i)u_i; z_6=[s_r;s_h]. Each g_i is one scalar broadcast over the modality's 256 channels, unlike S5's 256-channel gate. The S6 classifier is Linear(512,256)→Hardswish→Dropout(0.35)→Linear(256,4).

The code class CrossModalAttention and configuration value cross_attention are retained as historical identifiers; the scientific description is modality-token self-attention. Existing figure legends using Cross-Attn refer to this S6 operator, not conventional directional spatial cross-attention. No figure is redrawn for this clarification.

## S16.4 Parameter accounting

| Model | Registered parameters | Parameters on invoked path | Retained uncalled heads |
| --- | --- | --- | --- |
| S1 | 1,521,956 | 1,521,956 | 0 |
| S2 | 1,753,124 | 1,753,124 | 0 |
| S3 | 924,434 | 924,434 | 0 |
| S4 | 2,874,170 | 2,167,666 | 706,504 |
| S5 | 2,939,962 | 2,233,458 | 706,504 |
| S6 | 3,534,142 | 2,827,638 | 706,504 |
| S7 | 2,659,402 | 1,952,898 | 706,504 |
| S8 | 2,659,402 | 1,952,898 | 706,504 |


Invoked-path counts come from synthetic evaluation with direct-parameter ownership tracked at called modules; MultiheadAttention.out_proj is counted because its parameters are used functionally even though its Linear.forward hook is bypassed. Uncalled parameters are the retained RGB and HSI branch classifiers, not removed weights or a new compact model. They remain in the checkpoint and registered totals, so registered parameter counts and computational work describe different budgets. Values do not measure gradients, data-dependent importance or hardware performance. No reported parameter totals are replaced by invoked-path counts.

## S16.5 Recorded training configurations

We compared all 40 resolved configurations with the corresponding original-run config_snapshot.yaml files; every pair agrees after YAML parsing. Each model has the same non-seed settings across its five repeats, and all 40 epoch histories contain 100 epochs. The shared recorded settings are batch size 16, split seed 42 with validation/test ratios 0.15/0.15, AdamW (learning rate $5\times10^{-4}$, weight decay $8\times10^{-4}$), ten warm-up epochs, cosine scheduling (configured $T_{\max}=100$, minimum learning rate $5\times10^{-6}$), weighted smoothed focal loss (exponent 1.5, smoothing 0.05), Mixup ($\alpha=0.1$, probability 0.5) and EMA (decay 0.999, tau 2000). The class-weight exponent is one with a 1.4 ZM379 multiplier. Shared configuration and split rules do not authenticate the historical sample identities or actual pretrained-weight loading. The checkpoint rule and EMA/evaluation-stream discrepancy are specified above.

| Model | Bands | RGB / HSI size | Dropout | AMP | Gradient clip | Weighted sampler | Pretraining request |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | NA | 224 / — | 0.2 | True | 1.0 | ZM379 ×1.6; replacement | Yes; loading unverified |
| S2 | 50 | — / 24 | 0.35 | False | 0.5 | No | No |
| S3 | 50 | — / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |
| S4 | 50 | 224 / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |
| S5 | 50 | 224 / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |
| S6 | 50 | 224 / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |
| S7 | 50 | 224 / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |
| S8 | 24 | 224 / 24 | 0.35 | False | 0.5 | No | Yes; loading unverified |


The models follow a common training and tuning framework, with the model-specific settings disclosed in Table 4(C), rather than universally identical recipes. S1 uses horizontal flipping, a padded random crop, RandAugment(2,9), random erasing 0.25 and replacement-based inverse-frequency sampling with a 1.6 ZM379 multiplier. HSI-only S2–S3 use independent spatial flips and the recorded spectral perturbations. Fusion S4–S8 use joint horizontal/vertical flips (0.5 each), RGB RandAugment(2,9) and erasing 0.15, plus HSI rolling/noise/scaling/band dropout; the separate HSI flip probability is zero. Image inputs use ImageNet normalization; HSI uses the corresponding per-band normalization cache. The top-level RGB augmentation fields in HSI-only YAML files are not evidence that these operations act on HSI. Pretraining is disabled for S2 and requested for the other models; historical loaded layers and weights remain unverified. The authors confirm a common tuning principle, but no historical search log was recovered to establish the configuration-selection criterion or exclude test-set-informed choices. We therefore do not describe these runs as identically tuned, capacity-matched tests.

All models use training seeds 42–46. Shared EMA settings also include evaluation start epoch 15, minimum 1,500 updates, evaluation after three post-warm-up epochs and switch-drop threshold 0.01. All use early-stopping patience 18 with minimum epoch 100, making early stopping unable to shorten the recorded 100-epoch runs. S1's collapse monitor starts at epoch 1 and can abort; S2–S8 start monitoring at epoch 10 without abort-on-collapse. All observed runs nevertheless reach epoch 100. S1 uses four workers and prefetch factor four; the other models use two workers and prefetch factor two. All record pin_memory and persistent_workers as true, AMP is requested only for S1, and deterministic training is false throughout. Device-dependent AMP activation is not inferred solely from its configuration flag.

HSI perturbations common where applicable: spatial roll up to one pixel, spectral noise SD 0.003, reflectance scale 0.97–1.03, band-drop probability 0.2 and maximum fraction 0.08. S2/S3 use horizontal and vertical HSI flips with probability 0.5. Fusion loaders apply the flips jointly and disable the separate HSI flip. Both modality tensors in paired Mixup share the mixing coefficient and permutation; single-modality runs mix their own input only. RGB tensors are normalized with ImageNet mean/SD; HSI tensors use the cached per-band foreground statistics. All HSI models request SPA with edge exclusions 10/15 and preprocessed mean spectra; S2–S7 share the archived 50-band cache/statistics paths, S8 requests the archived 24-band counterparts. ROI cropping is disabled where configured. Historical input inventory, image acquisition identity, normalization-cache provenance and pretrained loading remain bounded by the existing audits.

Checkpoint selection uses strict increase in the unweighted mean of validation minibatch accuracies; exact ties retain the earlier epoch. The separate checkpoint audit verifies the first logged maximum and EMA selection stream for all 40 runs. The supplied historical evaluation script loads the main model entry. Which stream generated the archived predictions is unresolved. A common selection rule is therefore not proof of an identical historical test-inference stream.

## S16.6 Classification performance and efficiency

| Model | Bands | Accuracy (%) | Macro-F1 (%) | Params (M) | Recorded G operations | Recorded CPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | NA | 81.16 ± 1.09 | 81.09 ± 1.07 | 1.521956 | 0.0615 | 11.64 ± 0.86 |
| S2 | 50 | 86.14 ± 4.65 | 86.03 ± 4.84 | 1.753124 | 0.0790 | 17.93 ± 1.19 |
| S3 | 50 | 93.30 ± 0.94 | 93.32 ± 0.95 | 0.924434 | 0.4780 | 23.91 ± 2.45 |
| S4 | 50 | 93.23 ± 1.45 | 93.23 ± 1.46 | 2.874170 | 0.5392 | 49.98 ± 14.50 |
| S5 | 50 | 92.77 ± 1.41 | 92.73 ± 1.43 | 2.939962 | 0.5392 | 47.58 ± 15.58 |
| S6 | 50 | 93.72 ± 1.42 | 93.69 ± 1.46 | 3.534142 | 0.5404 | 47.88 ± 14.67 |
| S7 | 50 | 94.11 ± 1.15 | 94.11 ± 1.18 | 2.659402 | 0.5466 | 48.51 ± 3.68 |
| S8 | 24 | 95.58 ± 1.70 | 95.63 ± 1.67 | 2.659402 | 0.3294 | 22.73 ± 0.86 |


Accuracy and Macro-F1 are recalculated means ± sample SD across seeds 42–46, with 570 archived test predictions per run, sourced from results/model_summary.csv. CPU entries are historical within-benchmark timing means ± SD from raw_model_complexity.csv, not across-training-seed SDs. G operations preserve the archived THOP-derived values; scalar FLOP versus MAC convention and historical tool/operator coverage remain unverified. The original five-warm-up/thirty-timed-pass benchmark does not establish stable percentile timing or full preprocessing/embedded throughput. These values are not newly measured here. Table 4(B)'s paired intervals and Holm-adjusted tests govern the statistical limits; none of the seven adjusted comparisons establishes pairwise accuracy superiority at 0.05.

S4–S7 have the same recorded SPA-50 input budget and common nominal fusion training recipe but different heads, modulation paths and parameter budgets. S8 uses SPA-24. Comparing S8 with S4/S5/S6 does not isolate the fusion operator. The separate same-band fusion-control suite is recorded under results/statistics/matched24/. Historical file/physical-sample identities are not authenticated by equal split seeds.

## Author confirmation and unresolved evidence

The author confirmed in this chat that “uniform” means a common training/tuning principle with archived model differences disclosed. No YAML field has been overwritten to make it appear equal. That confirmation is not an explicit record of the search objective or a declaration that test results never informed configuration choices; no such historical search log was recovered. Successful historical pretrained loading, exact data/input identities and prediction-weight stream also remain unresolved. These issues limit a fully validated end-to-end fairness claim. This revision documents actual available evidence instead of reporting them as completed checks.

Figure 3 and Figure 4 image files remain unchanged, as requested by the author. Word files are not synchronized by this Markdown revision.


Revision verification: all checks in revision_validation.json pass, including preservation of all numbered display equations, the 15 historical model-result rows, image references, original Figure 3, and blue response excerpts matching the manuscript. Existing images are reused without modification.
