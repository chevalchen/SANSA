1. Module Insertion Point: Mask Decoder Output

Optimal Location: Placed after the Mask Decoder (e.g., within models/sansa/sansa.py).



Reasoning against Memory Attention: The Memory Attention stage processes uncompressed, high-dimensional dense spatial features. Introducing attention entropy calculations or extra prediction networks here would drastically increase VRAM consumption and computational overhead.



Retaining Token-Level Efficiency: UncertainSAM’s core strength is its negligible computational cost. Extracting and concatenating only the 256-D Mask Token and 256-D IoU Token at the Mask Decoder's output ensures near-zero impact on inference latency.



2. Module Composition & Domain Adaptation

Architecture: Strictly follow UncertainSAM's lightweight design by deploying a 3-layer Multi-Layer Perceptron (MLP) with a hidden dimension of 512 and a 1-D Sigmoid output, taking the concatenated 512-D token vector as input.



Domain Logic Shift:



Interactive Segmentation (SAM): Uncertainty arises primarily from "prompt ambiguity" (e.g., ambiguous points or boxes).



Few-Shot Segmentation (SANSA): Uncertainty arises primarily from "failed feature alignment between the Support and Query sets."



Target Definition: The physical objective of this MLP is to predict the Expected IoU of the Query mask generated under the guidance of the current Support image. A lower predicted Expected IoU indicates higher alignment uncertainty (i.e., the Support image provides poor reference features or causes semantic contamination).



3. Modification Scheme & Training Pipeline (Minimal Compute Path)

To preserve the proven feature space of SANSA's AdaptFormer while minimizing computational demands, a Post-hoc Training strategy is adopted:



Phase 1: Data Preparation (No External Data Required)

Perform forward inference on the existing FSS training dataset (Base Classes, e.g., PASCAL-5i or COCO-20i) using pre-trained SANSA weights. For each sample, record the Mask Token, IoU Token, and the actual Target IoU (the IoU between the predicted mask and the Ground Truth).



Phase 2: Train the UQ-MLP

Freeze the entire SAM2 backbone and SANSA AdaptFormer. Train the 3-layer MLP using Mean Squared Error (MSE Loss) to fit the collected Target IoU values. Convergence requires only a few minutes.



Phase 3: Inference Strategy Modification (Multi-Shot Optimization)

For K-shot scenarios, replace SANSA's default average-pooling feature fusion strategy:



Generate independent 1-shot predictions for each of the K Support images against the Query.



Feed the resulting K token pairs into the trained UQ-MLP to obtain K "confidence scores" (Expected IoUs).



Dynamic Weighting: Apply a Softmax function to these K confidence scores to compute fusion weights. Use these weights to dynamically aggregate the multi-shot predicted masks (or Value projections in the Memory Bank), automatically suppressing low-quality Support samples that introduce high uncertainty.



4. Code Implementation Approach

Data Interception: Intercept the mask_tokens and iou_predictions variables at the final stage of SANSA's forward pass (typically after the decoder call in engine.py or inference_fss.py).



Minimal Intrusion: Implement the module by adding a standalone, lightweight uq_mlp.py script in the models/sansa/ directory. This requires no structural changes to the original SAM2 or SANSA computational graphs, ensuring a purely incremental update.