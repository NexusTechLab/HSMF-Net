# Architecture specification

Audit date: 5 November 2026. The frozen source snapshot and archived S7/S8 configurations were instantiated on CPU with synthetic batch-one inputs. No pretrained weights were downloaded, no original-image predictions were generated, and no training or classification experiment was performed. This describes the inspected snapshot, not a certification of all historical runtime conditions.

## Artifacts and audit environment

{'python': '3.12.11', 'torch': '2.8.0+cu129', 'torchvision': '0.23.0+cu129', 'thop': '2.1.6'}

S8_model_architecture.txt contains the full module tree; S7/S8_layer_summary.csv provide layer settings, parameter counts and tensor shapes; S8_bottlenecks.csv lists the compact block specification. The original settings remain in configs/archived/. architecture_audit.json and checkpoint_structure_checks.json record the inspected source/configuration and key/shape checks.

## RGB branch

The snapshot defines its own MobileNetV3-Small implementation; torchvision is the requested ImageNet weight source. The RGB stem is bias-free Conv2d(3,16,3,stride=2,padding=1), BN(eps=0.001,momentum=0.01), Hardswish. Input channels remain three. Intermediate output is after features[6], i.e. features[:7], at (N,40,14,14). The remaining features[7:] end at features[12], a 1×1 96→576 convolution/BN/Hardswish, yielding (N,576,7,7).

Bottlenecks apply optional 1×1 expansion, grouped depthwise convolution, optional SE, and bias-free 1×1 projection plus BN without a projection activation. RGB SE uses global average pooling, pointwise reduction/expansion, ReLU and Hardsigmoid; reduced width follows the local make_divisible(expanded_channels//4,8) rule. Residual addition requires stride one and equal input/output widths.

### RGB inverted residuals

| Index | Cin | Expanded | Ratio | DW kernel | DW stride | SE | Activation | Residual | Cout | Output (NCHW) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 16 | 1 | 3 | 2 | True | ReLU | False | 16 | [1, 16, 56, 56] |
| 2 | 16 | 72 | 4.5 | 3 | 2 | False | ReLU | False | 24 | [1, 24, 28, 28] |
| 3 | 24 | 88 | 3.66667 | 3 | 1 | False | ReLU | True | 24 | [1, 24, 28, 28] |
| 4 | 24 | 96 | 4 | 5 | 2 | True | Hardswish | False | 40 | [1, 40, 14, 14] |
| 5 | 40 | 240 | 6 | 5 | 1 | True | Hardswish | True | 40 | [1, 40, 14, 14] |
| 6 | 40 | 240 | 6 | 5 | 1 | True | Hardswish | True | 40 | [1, 40, 14, 14] |
| 7 | 40 | 120 | 3 | 5 | 1 | True | Hardswish | False | 48 | [1, 48, 14, 14] |
| 8 | 48 | 144 | 3 | 5 | 1 | True | Hardswish | True | 48 | [1, 48, 14, 14] |
| 9 | 48 | 288 | 6 | 5 | 2 | True | Hardswish | False | 96 | [1, 96, 7, 7] |
| 10 | 96 | 576 | 6 | 5 | 1 | True | Hardswish | True | 96 | [1, 96, 7, 7] |
| 11 | 96 | 576 | 6 | 5 | 1 | True | Hardswish | True | 96 | [1, 96, 7, 7] |

### HSI 2D bottlenecks

| Index | Cin | Expanded | Ratio | DW kernel | DW stride | SE | Activation | Residual | Cout | Output (NCHW) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 256 | 1024 | 4 | 3 | 1 | False | ReLU | False | 24 | [1, 24, 12, 12] |
| 1 | 24 | 72 | 3 | 3 | 1 | False | ReLU | True | 24 | [1, 24, 12, 12] |
| 2 | 24 | 72 | 3 | 5 | 1 | True | Hardswish | False | 40 | [1, 40, 12, 12] |
| 3 | 40 | 120 | 3 | 5 | 1 | True | Hardswish | True | 40 | [1, 40, 12, 12] |
| 4 | 40 | 120 | 3 | 5 | 1 | True | Hardswish | False | 48 | [1, 48, 12, 12] |
| 5 | 48 | 288 | 6 | 5 | 2 | True | Hardswish | False | 96 | [1, 96, 6, 6] |
| 6 | 96 | 576 | 6 | 5 | 1 | True | Hardswish | True | 96 | [1, 96, 6, 6] |

## HSI operations and tensor layout

The HSI stem has two bias-free Conv3d layers, 1→16 and 16→32, each kernel 3×3×3, unit stride, padding 1×1×1, BatchNorm3d and ReLU. In tiny-spatial mode, only the second stage is followed by MaxPool3d(kernel=stride=(1,2,2)). Spectral depthwise convolution is 32→32, kernel (3,1,1), padding (1,0,0), stride one, groups 32, bias false, BN and Hardswish.

Adaptive average and maximum pooling are functional calls to output (N,32,8,12,12); their arithmetic mean is returned with view(N,32*8,12,12). There is no permute or explicit contiguous call in this pooling/view path. Merged channel q=8c+k, with channel index c outside spectral-bin index k. For S7 the incoming depth is 50 rather than 24, but the eight-bin output and 256-channel 2D input are the same.

The HSI bottleneck input/output/expansion specification is tabulated above. The local width rule is make_divisible(Cin*expand,8). Each expanded or depthwise convolution is bias-free and followed by BN and the tabulated activation; the final projection is 1×1 bias-free convolution followed by BN. HSI SE uses reduced width max(1,int(expanded_channels*0.25)), ReLU and Hardsigmoid. The first two planned 2D spatial downsamplings are suppressed by tiny-spatial mode. features2d[3] is the (N,40,12,12) intermediate output; features2d[7] is a 1×1 96→576 Conv/BN/Hardswish producing (N,576,6,6). HSI and fusion BatchNorm defaults are eps=1e-5, momentum=0.1.

## Fusion, propagation and classifier

Mid projections are 40→40; late projections are 576→96, each bias-free 1×1 Conv/BN/Hardswish. RGB adaptive average pooling changes 14×14→12×12 and 7×7→6×6. The HSI mid return projection is 40→40 Conv/BN/Hardswish. Each directional generator is biased 1×1 Conv C→C, Hardswish, Dropout2d(0.35), biased 1×1 Conv C→C. Its final convolution starts at zero. Scaling is residual x+gamma*x, without beta, sigmoid bounding or extra scaling. Generators operate on the partner aligned feature map.

Only the modulated HSI intermediate representation continues through the deeper HSI encoder. The RGB late representation continues from the original RGB intermediate backbone tensor. The classifier concatenates pooled late RGB, late HSI, mid RGB and mid HSI (96+96+40+40=272), then applies Linear(272,192), Hardswish, Dropout(0.35), Linear(192,4). The model returns four logits; no softmax is in the counted forward pass. Both original branch classification heads remain registered but are bypassed.

## Parameters: registration versus executed path

| Model | Registered | Declared trainable | Frozen parameters | Executed path | Bypassed heads |
| --- | --- | --- | --- | --- | --- |
| S8 | 2659402 | 2659402 | 0 | 1952898 | 706504 |
| S7 | 2659402 | 2659402 | 0 | 1952898 | 706504 |

Registered parameter counts exclude BN running buffers. The 706,504 bypassed parameters comprise 594,948 in the RGB branch classifier and 111,556 in the HSI branch classifier. They do not affect the fusion forward output. The manuscript 2.659 M value is the registered total, not 1.952898 M in the executed path. This distinction is disclosed without silently replacing the historical count.

## Weight initialization, freezing and checkpoint compatibility

The archived recipe requests torchvision mobilenet_v3_small(weights="DEFAULT"). In this audit environment DEFAULT resolves to MobileNet_V3_Small_Weights.IMAGENET1K_V1. The snapshot loader accepts matching key names and tensor shapes and can return without loading. An offline exercise used randomly initialized torchvision tensors only to test compatibility; it is not evidence that ImageNet weights were loaded in the original experiments. Historical successful loading and the originally resolved weight version remain unverified.

All 2,659,402 instantiated parameters have requires_grad=True. The snapshot trainer gives model.parameters() to the optimizer and has no RGB-backbone freeze/unfreeze schedule. The nontrainable EMA copy is distinct from freezing the trainable backbone. Checkpoint keys/shapes were checked for S7 and S8 seed 42 in both main and EMA streams, with exact agreement. No top-level pretraining provenance is present in those checkpoints. Shape agreement does not establish the original input identities, loaded initialization values or historical prediction stream. The earlier 40-checkpoint metadata audit remains separate.

## Operation counting and coverage

| Model | Input shapes | Conv/Linear-only GMACs | THOP count / 1e9 | Rounded THOP count |
| --- | --- | --- | --- | --- |
| S8 | [[1, 3, 224, 224], [1, 1, 24, 24, 24]] | 0.317935008 | 0.329383552 | 0.3294 |
| S7 | [[1, 3, 224, 224], [1, 1, 50, 24, 24]] | 0.531792288 | 0.546595456 | 0.5466 |

The independent Conv/Linear MAC tally uses output elements × (input channels/groups) × kernel volume for convolutions and output elements × input features for Linear. Bias additions and all other operators are excluded. One multiplication-plus-accumulation is one MAC in this tally; these values are not complete-network FLOPs.

The two-model values are also supplied in complexity_report.csv. The inspected THOP 2.1.6 code uses MAC-style counts for Conv2d/Conv3d (including grouped/depthwise kernels) and Linear, and also adds normalization and supported module-pooling operation estimates. It is consequently neither the Conv/Linear-only tally nor a complete scalar-FLOP tally. No factor-of-two conversion was applied. Hardswish, Hardsigmoid and AdaptiveSpectralPool have no registered leaf counting rule in this run; functional spectral pooling, its arithmetic combination, residual additions and elementwise modulation/SE multiplication are not comprehensively counted. Comparisons, tensor rearrangements and memory traffic are not represented as a complete hardware cost.

The new raw THOP totals round to the recorded 0.3294 and 0.5466, establishing numerical reproducibility under the stated current audit, not recovery of the original profiling environment. Historical GFLOPs labels remain provisional descriptions of this profiler count; they must not be interpreted as complete scalar FLOPs or simply renamed complete GMACs. A complete convention and all-model coverage remain required for final global relabeling. Profiler source hashes and runtime versions are in architecture_audit.json.
