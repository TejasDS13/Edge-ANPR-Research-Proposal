<div style="page-break-after: always; break-after: page; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding-top: 140px; padding-bottom: 80px;">

<h1 style="font-size: 2.3em; line-height: 1.35; margin-bottom: 25px; font-weight: 800;">
Edge-ANPR: A Modular Algorithmic Architecture and Projected Performance Analysis for Unconstrained Traffic Environments
</h1>

<hr style="width: 70%; margin: 20px auto; border: 0.5px solid #bbb;">

<p style="font-size: 1.15em; font-weight: 500; margin-top: 15px; margin-bottom: 8px;">
Technical Research Paper &amp; System Architecture Specification
</p>

<p style="font-size: 0.95em; color: #bbb; margin-top: 5px;">
<strong>Target Domain:</strong> Intelligent Transportation Systems (ITS), Edge Deep Learning, Computer Vision<br>
<strong>Date:</strong> September 2026
</p>

</div>

## About the Author

**Tejas Handa** is an undergraduate researcher and 3rd-year **Bachelor of Technology (B.Tech)** candidate specializing in **Data Science** in the Department of Computer Science & Engineering at Guru Tegh Bahadur 4th Centenary Engineering College. 

Serving as a **Computer Vision Research Intern**, his applied research centers on deep learning model compression, lightweight neural representations, and latency-deterministic computer vision pipelines for embedded edge platforms. His technical focus addresses the challenges of real-time Intelligent Transportation Systems (ITS)—specifically optimizing deep sequence models, anchor-free object detection backbones, and spatial transformation networks under constrained thermal and memory envelopes (e.g., NVIDIA Jetson Orin Nano).

* **Academic Domain:** Data Science & Machine Learning Systems
* **Research Focus:** Edge AI, Model Optimization (TensorRT / INT8 Quantization), Real-Time Sequence Transcription
* **Professional Profiles:** [LinkedIn](http://linkedin.com/in/tejas-handa-95143432a)| [GitHub](https://github.com/TejasDS13)
* **Project Repository:** https://github.com/TejasDS13/Edge-ANPR-Research-Proposal)

---

<div align="center">

### Abstract
</div>

<div style="font-size: 1.08em; line-height: 1.65; text-align: justify; padding: 0 10px;">

Automatic Number Plate Recognition (ANPR) plays a vital role as a visual component in modern Intelligent Transportation Systems (ITS), in the processing of tolls automatically and in the provision of dynamic traffic routing. Yet, when computer vision systems are deployed on low-power edge accelerators within traffic environments that are unconstrained and not standardized, serious operational difficulties arise: there is arbitrary typography, two-tier dual-line stacked arrangements, extreme perspective skew, and strict limitations regarding thermal computing power. Traditional monolithic architectures often experience a series of failure modes throughout the stages of localization, geometric transformation, and character segmentation.

The paper introduces **Edge-ANPR**, a modular algorithmic framework that has been developed for use in real-time edge execution even when the operational conditions are not constrained. The recognition process is divided into four specific stages: (1) an anchor-free single-stage detector which assesses architectural changes among the YOLO variants (namely YOLOv8, YOLOv10, YOLO11, and forthcoming next-generation paradigms such as YOLO26) in order to achieve latency-deterministic plate extraction, (2) a differentiable Thin Plate Spline Spatial Transformer Network (TPS-STN) for performing uncalibrated non-rigid perspective rectification, (3) a segmentation-free CRNN-BiLSTM-CTC transcription engine, and (4) a deterministic CMVR syntax post-processor.

Rather than reporting unverified local physical runs, this study formalizes an additive latency model and establishes a projected performance envelope synthesized from published edge benchmarks. Under stated INT8/FP16 quantization assumptions on an NVIDIA Jetson Orin Nano, the modular pipeline is projected to sustain up to ~88 FPS at ~11.4 ms total frame latency, offering an academically rigorous blueprint for edge-tier traffic surveillance.

</div>

<div style="page-break-after: always; break-after: page;"></div>

## Interactive Table of Contents

- [About the Author](#about-the-author)
  - [Abstract](#abstract)
- [Interactive Table of Contents](#interactive-table-of-contents)
- [1. Introduction \& Research Motivation](#1-introduction--research-motivation)
  - [1.1 Context and Internship Objectives](#11-context-and-internship-objectives)
  - [1.2 Background and Operational Motivation](#12-background-and-operational-motivation)
  - [1.3 The Indian Operational Landscape: Non-Standardized Heterogeneity](#13-the-indian-operational-landscape-non-standardized-heterogeneity)
  - [1.4 Mathematical Formulation of the Pipeline](#14-mathematical-formulation-of-the-pipeline)
  - [1.5 Primary Research Contributions](#15-primary-research-contributions)
- [2. Comprehensive Architectural Evolution of YOLO in Edge ANPR](#2-comprehensive-architectural-evolution-of-yolo-in-edge-anpr)
  - [2.1 The Localization Bottleneck on Embedded Hardware](#21-the-localization-bottleneck-on-embedded-hardware)
  - [2.2 Comparative Dissection: YOLOv8 vs. YOLOv10 vs. YOLO11 vs. Next-Gen (YOLO26)](#22-comparative-dissection-yolov8-vs-yolov10-vs-yolo11-vs-next-gen-yolo26)
  - [2.3 Architectural \& System Specification Comparison Matrix](#23-architectural--system-specification-comparison-matrix)
- [3. A review of the literature and the algorithmic foundations](#3-a-review-of-the-literature-and-the-algorithmic-foundations)
  - [3.1 Baselines for the localisation and detection of licence plates](#31-baselines-for-the-localisation-and-detection-of-licence-plates)
  - [3.2 The correction of spatial transformation and distortion](#32-the-correction-of-spatial-transformation-and-distortion)
  - [3.3 Optical Character Recognition (OCR): Segmentation vs. Sequence Modeling](#33-optical-character-recognition-ocr-segmentation-vs-sequence-modeling)
  - [Architectural Trade-off and Engine Selection for Edge ANPR](#architectural-trade-off-and-engine-selection-for-edge-anpr)
  - [3.4 Indian standards for traffic regulation (CMVR)](#34-indian-standards-for-traffic-regulation-cmvr)
- [4. Proposed Methodology \& Pipeline Architecture](#4-proposed-methodology--pipeline-architecture)
  - [4.1 Stage 1: Anchor-Free Localization \& Layout Disambiguation](#41-stage-1-anchor-free-localization--layout-disambiguation)
  - [4.2 Stage 2: Parametric Thin Plate Spline Spatial Transformer Network (TPS-STN)](#42-stage-2-parametric-thin-plate-spline-spatial-transformer-network-tps-stn)
  - [4.3 Stage 3: Sequence-to-Sequence Transcription via CRNN-CTC](#43-stage-3-sequence-to-sequence-transcription-via-crnn-ctc)
    - [4.3.1 Architectural Decomposition](#431-architectural-decomposition)
    - [4.3.2 End-to-End Alignment-Free Optimization (CTC Formulation)](#432-end-to-end-alignment-free-optimization-ctc-formulation)
    - [4.3.3 Custom OCR Training Protocol \& Domain Adaptation](#433-custom-ocr-training-protocol--domain-adaptation)
  - [4.4 Stage 4: Indian CMVR Syntax Validation Engine](#44-stage-4-indian-cmvr-syntax-validation-engine)
- [5. Projected Performance \& Latency Analysis](#5-projected-performance--latency-analysis)
  - [5.1 Latency Decomposition Formulation](#51-latency-decomposition-formulation)
  - [5.2 Plate Localization: Architecture Profiles vs. Projected Backbone](#52-plate-localization-architecture-profiles-vs-projected-backbone)
  - [5.3 Sequence Transcription Fidelity under Perspective Skew](#53-sequence-transcription-fidelity-under-perspective-skew)
  - [5.4 Edge Compute \& Throughput Budgeting](#54-edge-compute--throughput-budgeting)
- [6. Deployment Challenges \& Architectural Limitations](#6-deployment-challenges--architectural-limitations)
  - [6.1 Quantization Drift and Hybrid Precision Budgeting](#61-quantization-drift-and-hybrid-precision-budgeting)
  - [6.2 Explicit Architectural Limitations](#62-explicit-architectural-limitations)
- [7. Conclusion \& Empirical Validation Roadmap](#7-conclusion--empirical-validation-roadmap)
  - [7.1 Conclusion](#71-conclusion)
  - [7.2 Empirical Validation Roadmap (Future Work)](#72-empirical-validation-roadmap-future-work)
- [References](#references)

<div style="page-break-after: always; break-after: page;"></div>

## 1. Introduction & Research Motivation

### 1.1 Context and Internship Objectives
This research study was initiated and developed as part of an applied **Computer Vision & Embedded Edge Intelligence Research Internship**. Modern industrial surveillance and Intelligent Transportation Systems (ITS) are rapidly moving away from centralized, high-bandwidth cloud computing toward localized roadside edge computing units. Transmitting uncompressed, high-definition multi-lane traffic streams across cellular backhauls incurs prohibitive bandwidth costs, variable latency spikes, and severe privacy risks.

The primary objective of this internship project is to bridge the design gap between heavy, high-accuracy desktop vision architectures and ultra-low-power, resource-constrained edge accelerators (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi 4B). Specifically, this paper establishes a theoretically sound and computationally efficient architectural blueprint—**Edge-ANPR**—that optimizes the Pareto frontier between high inference throughput (targeting $>60$ FPS real-time guarantees) and character recognition fidelity under non-cooperative, unconstrained environmental degradation.

### 1.2 Background and Operational Motivation
Intelligent Transportation Systems (ITS) have transitioned from static sensor-driven checkpoints into autonomous, vision-based surveillance infrastructures. Automatic Number Plate Recognition (ANPR) functions as a foundational visual layer enabling automated fee processing, dynamic congestion routing, and urban surveillance. While traditional computer vision relied on hand-engineered edge filters and morphological kernels, deep learning models now dominate plate localization and sequence transcription. Despite these advancements, maintaining real-time throughput on low-power edge accelerators under severe perspective and environmental distortions remains a complex engineering challenge.

### 1.3 The Indian Operational Landscape: Non-Standardized Heterogeneity
Deploying standard commercial ANPR architectures within the Indian traffic ecosystem exposes unique, domain-specific edge failure modes rarely encountered in standardized test corpora:
* **Structural Layout Disparity:** Two-wheelers and specialized freight carriers deploy square-profile, split-row stacked alphanumeric plates, whereas passenger cars feature horizontal single-line plates.
* **Typography and Font Arbitrariness:** While High Security Registration Plate (HSRP) regulations mandate standardized typography, millions of vehicles feature non-standard aspect ratios, decorative regional scripts, and localized decals.
* **Mounting Artifacts and Physical Occlusion:** Retention screws, structural bolts, and rust points regularly bisect alphanumeric glyphs, inducing catastrophic character fragmentation in conventional contour-splitting OCR engines.
* **Semantic Plate Classes:** Plates vary chromatically across functional domains: Private (White), Commercial (Yellow), Electric (Green), and Rental (Black), requiring chromatic invariance during deep feature extraction.

### 1.4 Mathematical Formulation of the Pipeline
The standard multi-stage ANPR pipeline is formalized as a composite sequence of parameterized spatial and linguistic mappings over an input surveillance video frame $I \in \mathbb{R}^{H \times W \times C}$:

$$\mathcal{P}_{\text{plate}} = \mathcal{F}_{\text{det}}(I; \Theta_{\text{det}})$$

$$\mathcal{P}_{\text{rect}} = \mathcal{T}_{\text{stn}}(\mathcal{P}_{\text{plate}}; \Theta_{\text{stn}})$$

$$\hat{\mathcal{S}} = \mathcal{R}_{\text{seq}}(\mathcal{P}_{\text{rect}}; \Theta_{\text{seq}})$$

Where:
* $\mathcal{F}_{\text{det}}$ denotes the plate detection sub-network parameterized by weights $\Theta_{\text{det}}$, extracting the bounding plate region of interest $\mathcal{P}_{\text{plate}} \subset I$.
* $\mathcal{T}_{\text{stn}}$ represents the spatial transformation mapping parameterized by $\Theta_{\text{stn}}$, yielding a canonical horizontal plate tensor $\mathcal{P}_{\text{rect}}$.
* $\mathcal{R}_{\text{seq}}$ is the sequence transcription network parameterized by $\Theta_{\text{seq}}$, decoding $\mathcal{P}_{\text{rect}}$ directly into an alphanumeric character sequence $\hat{\mathcal{S}} = \{s_1, s_2, \dots, s_L\}$ where $s_i \in \Sigma$ and $\Sigma$ is the statutory alphanumeric character vocabulary.

### 1.5 Primary Research Contributions
To evaluate trade-offs between precision, inference latency, and hardware constraints, this paper formulates the Edge-ANPR architectural framework. The key contributions of this study are:
1. **Applied Edge Architecture Blueprint:** System design born from an applied industrial internship context, systematically addressing edge compute constraints, memory bandwidth limits, and thermal throttling envelopes on roadside micro-nodes.
2. **Comprehensive YOLO Evolution Analysis:** An architectural deconstruction of modern real-time detectors (YOLOv8, YOLOv10, YOLO11, and emerging next-gen paradigms), evaluating how anchor-free heads and NMS elimination impact edge inference jitter.
3. **Differentiable Geometric Rectification Formulation:** Mathematical integration of a Thin Plate Spline (TPS) Spatial Transformer Network operating directly on localized feature tensors to eliminate brittle contour homography.
4. **Segmentation-Free Sequence Transcription:** Design of a continuous CRNN-BiLSTM-CTC transcription pipeline that avoids fragile character-level boundary segmentation under tight kerning and hardware fastener occlusions.
5. **Analytical Latency Decomposition:** Formulation of an explicit additive latency model ($T_{\text{total}} = T_{\text{det}} + T_{\text{stn}} + T_{\text{crnn}} + T_{\text{post}}$) alongside a projected compute envelope under INT8/FP16 quantization.

---

## 2. Comprehensive Architectural Evolution of YOLO in Edge ANPR

### 2.1 The Localization Bottleneck on Embedded Hardware
To localise a licence plate on embedded micro-nodes it is necessary to balance the precision of the spatial bounding box against the available compute resources. Traditional anchor-based systems cause a great deal of memory traffic since they generate thousands of dense candidate anchors on multi-scale feature maps. On edge devices memory bandwidth is usually the main bottleneck rather than the raw arithmetic FLOP capacity.

### 2.2 Comparative Dissection: YOLOv8 vs. YOLOv10 vs. YOLO11 vs. Next-Gen (YOLO26)
* **YOLOv8 (Anchor-Free Decoupled Baseline):** It separates the classification and regression branches and makes use of a Task-Aligned Assigner (TAL). Although it attains high spatial accuracy, it still depends on Non-Maximum Suppression (NMS) as part of the post-processing procedure. In the case of bumper-to-bumper traffic where the bounding boxes are heavily clustered, NMS causes varying and non-deterministic latency jitter.
* **YOLOv10 (NMS-Free Dual Assignment):** During training it uses dual label assignments (carrying out one-to-many assignments for optimization purposes and one-to-one assignments for inference). As a result, the entire Non-Maximum Suppression post-processing step is removed and inference on edge platforms becomes a deterministic single-pass tensor computation.
* YOLO11 (Refined Feature Flow): by incorporating C3k2 modules and using optimized SPPF structures, it achieves better feature representation for distant, low-resolution plates at the same time as reducing the total computational parameter overhead.
* **Next-Gen Paradigm (YOLO26 / Vision-Transformer Hybrids):** This denotes the next generation of paradigms which combine lightweight convolutional stages with coordinate self-attention heads and make use of hardware-aware INT8 neural architectural search (NAS). It gets rid of heuristic anchor priors altogether, enabling global context disambiguation in the case of bumper decoys.

<div style="page-break-before: always; break-before: page;"></div>

### 2.3 Architectural & System Specification Comparison Matrix

The operational feasibility of edge ANPR is governed not merely by raw accuracy metrics, but by the memory bandwidth and deterministic execution limits of roadside accelerators. Table 1 details the architectural trade-offs across YOLO generations under constrained edge compute profiles.

| Model Architecture | Parameters (M) & FLOPs (G) | Post-Processing Paradigm | Edge GPU Inference (Jetson Orin Nano, FP16) | CPU Host Inference (Quad-Core, INT8 / ONNX) | VRAM / RAM Footprint | Hardware Viability & Bottleneck Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8-Nano** | ~3.2 M / 8.7 G | Heuristic NMS | ~13.5 ms (~74 FPS) | ~42.0 ms (~24 FPS) | ~1.1 GB VRAM / ~450 MB RAM | High accuracy, but heuristic NMS induces dynamic latency jitter under bumper-to-bumper occlusion. |
| **YOLOv10-Nano** | ~2.7 M / 6.7 G | **Dual-Assignment (NMS-Free)** | **~8.2 ms (~122 FPS)** | **~26.5 ms (~38 FPS)** | **~0.8 GB VRAM / ~320 MB RAM** | **Optimal Edge Backbone:** Deterministic single-pass inference without post-processing latency spikes. |
| **YOLO11-Nano** | ~2.6 M / 6.5 G | Heuristic NMS | ~11.0 ms (~91 FPS) | ~34.0 ms (~29 FPS) | ~0.9 GB VRAM / ~360 MB RAM | Superior feature extraction on distant plates via C3k2 blocks, but still bounded by NMS overhead. |
| **Next-Gen Hybrid (YOLO26 Concept)** | ~2.2 M / 5.4 G | Direct Coordinate Regression | Projected ~6.8 ms (~147 FPS) | Projected ~21.0 ms (~48 FPS) | Projected ~0.7 GB VRAM / ~280 MB RAM | Transformer-CNN hybrid with INT8 hardware-aware compilation; theoretical limit for sub-10W nodes. |

*Table 1: Comparative architectural and edge hardware specification matrix for license plate localization backbones.*

---

## 3. A review of the literature and the algorithmic foundations

Modern ANPR systems have developed from simple morphological edge extractors into deep, integrated sequence-to-sequence networks.

### 3.1 Baselines for the localisation and detection of licence plates
The early methods placed great emphasis on vertical edge density and the use of morphological gradient transformations [1], but they deteriorated when exposed to complex background clutter, illumination gradients, and bumper grilles.

The transition to deep convolutional detectors bifurcated localization strategies:
* **Two-Stage Detectors (for example, Faster R-CNN):** Use a Region Proposal Network (RPN) together with RoI-pooling. Although they offer high precision, the multi-stage process results in a latency of more than 80 ms per frame, which hinders their use in real-time applications on edge devices [2].
* **Single-Stage Detectors (YOLO Series):** The approach of these detectors is to carry out bounding-box regression directly for the purpose of localization. Laroca et al. showed that single-stage models are able to achieve robust localization on unconstrained vehicle plates [3]. The more recent NMS-free models, such as YOLOv10, remove the bottlenecks associated with the Non-Maximum Suppression post-processing step, as a result of which the edge execution latency is reduced to below 10 ms [4].

### 3.2 The correction of spatial transformation and distortion
By fitting surveillance cameras onto gantries, acute pan and tilt angles (greater than 35°) are introduced. If an unrectified and skewed bounding crop is fed directly into an OCR engine, this results in a high rate of character errors.

The early methods used for remediation were based on the classical Hough transform or corner-point homography, and these became ineffective when the edges of the plate were dented or only partly obscured. In order to overcome this problem, Jaderberg et al. proposed the Spatial Transformer Network (STN) [5]. When using a Thin Plate Spline (TPS) interpolation grid, the network is able to learn non-rigid spatial coordinate transformations entirely by end-to-end training without the need for manually defined geometric corner labels.

### 3.3 Optical Character Recognition (OCR): Segmentation vs. Sequence Modeling
Character transcription methods fall into two primary categories:
* **Character Segmentation Pipelines: **Classical OCR approaches segmentation through connected component analysis or projection profiling prior to classification of individual glyphs. In unconstrained scenarios, paint peeling, mud spots, and retaining screws cause fusion of neighboring characters, resulting in cascade failures for subsequent classifiers.
* **Segmentation-Free Sequence Modeling:** The work of Shi et al. introduced the Convolutional Recurrent Neural Network (CRNN) that integrates feature extraction using a CNN, sequence modeling using Bidirectional LSTMs, and CTC loss [6]. The CTC decoding process enables direct transcription from feature slices without any segmentation of character boundaries.

### Architectural Trade-off and Engine Selection for Edge ANPR
Selecting an OCR paradigm for embedded edge deployment requires balancing perspective tolerance against strict computational budgets. Table 2 justifies the selection of a lightweight CRNN-CTC architecture over heavier scene-text alternatives.

| OCR Engine / Paradigm | Algorithmic Formulation | Skew & Occlusion Resilience | Edge Latency (Jetson Orin Nano, FP16) | Architectural Suitability for Edge-ANPR |
| :--- | :--- | :--- | :--- | :--- |
| **Tesseract v5 (LSTM)** | Binarized contour segmentation + 1D LSTM | Poor (Severe glyph fragmentation on bolts/screws) | ~24.5 ms (~40 FPS) |  Incompatible: High failure rate on tight Indian typography. |
| **EasyOCR (CRAFT + ResNet)** | Two-stage: CRAFT detector + attention recognizer | Moderate (Sensitive to acute spatial shear) | ~38.0 ms (~26 FPS) |  Incompatible: Redundant text-box detection adds latency overhead. |
| **TrOCR / Vision-Transformers** | Patch-based encoder + autoregressive decoder | Very High (Self-attention resolves noise) | ~145.0 ms (<7 FPS) |  Incompatible: Autoregressive token decoding exceeds edge thermal budgets. |
| **CRNN-CTC (Lightweight Backbone)** | Feature column slicing + BiLSTM + CTC decoding | **High (when paired with preceding TPS-STN)** | **~1.8 ms - 3.5 ms (>250 FPS)** |  Selected: Optimal Pareto frontier of low latency and segmentation immunity. |

*Table 2: Comparative evaluation of OCR paradigms under edge compute constraints and unconstrained environmental distortion.*

### 3.4 Indian standards for traffic regulation (CMVR)

The alphanumeric structure prescribed for vehicle registrations in India is defined by the statutory guidelines issued by the Ministry of Road Transport and Highways (MoRTH) in the Central Motor Vehicles Rules (CMVR) [7]. By applying this syntactic structure in the post-processing stage, deterministic error correction can be achieved with regard to visual ambiguities of the characters.

---

## 4. Proposed Methodology & Pipeline Architecture

The Edge-ANPR pipeline decouples license plate recognition into four distinct operational stages:

![Edge-ANPR Pipeline](figures/pipeline_architecture.png)

*Figure 1: High-level architectural pipeline of the proposed modular ANPR system, detailing the decoupling of localization, differentiable geometric rectification, sequence transcription, and deterministic syntax validation.*

### 4.1 Stage 1: Anchor-Free Localization & Layout Disambiguation
The localization stage employs an anchor-free single-stage detector (e.g., YOLOv10-Nano). The loss function jointly optimizes bounding-box coordinate regression and structural layout classification:

$$\mathcal{L}_{\text{det}} = \lambda_{\text{box}}\mathcal{L}_{\text{CIoU}}(B, \hat{B}) + \lambda_{\text{cls}}\mathcal{L}_{\text{BCE}}(c, \hat{c})$$

Where $B=[x_c, y_c, w, h]^T$ defines the predicted bounding coordinates, and $c \in \{0, 1\}$ classifies the plate geometry directly as Single-Line (passenger vehicles) or Dual-Line (motorcycles and commercial carriers).

### 4.2 Stage 2: Parametric Thin Plate Spline Spatial Transformer Network (TPS-STN)
Overhead surveillance cameras produce acute perspective distortions that compress character aspect ratios. Rather than relying on classical corner homography, rectification is formulated through a differentiable Thin Plate Spline (TPS) Spatial Transformer Network [5].

Let the localized plate crop be $I_p \in \mathbb{R}^{H_p \times W_p \times 3}$. A convolutional localization sub-network predicts coordinates for $K$ fiducial control points on the distorted crop, denoted as $C = [c_1, c_2, \dots, c_K] \in \mathbb{R}^{2 \times K}$. These control points map to a canonical regular target grid $C' = [c'_1, c'_2, \dots, c'_K] \in \mathbb{R}^{2 \times K}$.

The TPS transformation calculates coordinate displacements via radial basis kernel functions $\phi(r) = r^2 \ln(r)$:

$$\begin{bmatrix} x_i^s \\ y_i^s \end{bmatrix} = \begin{bmatrix} d_0 & d_x & d_y \\ c_0 & c_x & c_y \end{bmatrix} \begin{bmatrix} 1 \\ x_i^t \\ y_i^t \end{bmatrix} + \sum_{k=1}^K \begin{bmatrix} w_{k,x} \\ w_{k,y} \end{bmatrix} \phi(\vert{}\vert{}c'_k - (x_i^t, y_i^t)^T\vert{}\vert{})$$

Where:
* $(x_i^t, y_i^t)$ represents regular normalized coordinates on the target canvas $\mathcal{P}_{\text{rect}}$.
* $(x_i^s, y_i^s)$ denotes the corresponding warped sampling source coordinates on the input crop $I_p$.
* Matrix coefficients and kernel weights $w_k$ are determined by solving the linear boundary system.

To ensure differentiability for end-to-end training, target grid pixel values $V_i$ are sampled via bilinear interpolation:

$$V_i = \sum_{n=1}^{H_p} \sum_{m=1}^{W_p} I_p(n, m) \max(0, 1 - \vert{}x_i^s - m\vert{}) \max(0, 1 - \vert{}y_i^s - n\vert{})$$

### 4.3 Stage 3: Sequence-to-Sequence Transcription via CRNN-CTC
To prevent character boundary segmentation failures induced by physical fasteners, paint degradation, and variable kerning, the rectified plate tensor $\mathcal{P}_{\text{rect}} \in \mathbb{R}^{32 \times 128 \times 3}$ is transcribed through a segmentation-free CRNN-CTC sequence modeling architecture [6].

#### 4.3.1 Architectural Decomposition
1. **Feature Sequence Extraction (CNN Backbone):** A lightweight convolutional feature extractor (MobileNetV3 or ResNet-18 backbone) downsamples the spatial dimension along the vertical axis while retaining horizontal temporal resolution [6]. The output tensor maps to a feature vector sequence $x = (x_1, x_2, \dots, x_T)$, where each slice $x_t \in \mathbb{R}^D$ denotes the receptive field of a vertical column across the license plate [6].
2. **Recurrent Context Modeling (Deep BiLSTM):** A 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network with hidden state dimension $H = 256$ reads the sequence in both forward and backward temporal directions [6]. This allows bidirectional context disambiguation (e.g., leveraging linguistic syntax that an Indian plate prefix begins with dual alphabetic state tokens) [6, 7].
3. **Linear Softmax Projection:** Each hidden state $h_t$ is projected into a normalized probability distribution $y^t$ across the character alphabet vocabulary $\Sigma' = \Sigma \cup \{\epsilon\}$, where $\Sigma = \{0\text{--}9, \text{A--Z}\}$ and $\epsilon$ represents the Connectionist Temporal Classification (CTC) blank token [6].

$$\mathcal{P}(\pi | x) = \prod_{t=1}^T y_{\pi_t}^t$$

Where $\pi = (\pi_1, \pi_2, \dots, \pi_T)$ represents a valid frame-level sequence path. The transcription string $l$ is decoded via standard collapse mapping $\mathcal{B}$, which merges consecutive identical characters and subsequently strips null blank tokens ($\epsilon$) [6]:

$$\mathcal{B}(\text{D - D - } \epsilon \text{ - D - L - L}) \rightarrow \text{"DDL"}$$

#### 4.3.2 End-to-End Alignment-Free Optimization (CTC Formulation)
Unlike conventional bounding-box OCR, the network does not require localized character bounding annotations [6]. Training is driven directly by pairing the unsegmented plate crop with its complete ground-truth alphanumeric string label $l$ [6]. The network minimizes the negative log-likelihood across all valid collapse paths $\mathcal{B}^{-1}(l)$ [6]:

$$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(l)} \mathcal{P}(\pi | x)$$

Gradients propagate through both the recurrent BiLSTM layers and convolutional feature extraction heads simultaneously via the forward-backward dynamic programming algorithm [6].

#### 4.3.3 Custom OCR Training Protocol & Domain Adaptation
To transition the CRNN-CTC engine from an uninitialized network to a production-grade OCR, a two-stage training regimen is formulated:

1. **Synthetic Data Bootstrapping (Phase I):**
   * *Engine Formulation:* Using synthetic text rendering pipelines, $1.5 \times 10^6$ synthetic license plate crops are generated across Indian High Security Registration Plate (HSRP) mandatory typefaces (FE-Schrift, DIN 1451) and regional stylistic variations [7].
   * *Domain Perturbations:* Synthetic artifacts are randomly applied, including additive Gaussian noise, projective affine shearing ($\pm 20^\circ$), synthetic retention bolt occlusion masks, non-uniform specular glare, and chromatic domain shifts across white, yellow, green, and black backgrounds [7].
   * *Optimization:* The network is trained with an AdamW optimizer (initial learning rate $\eta = 10^{-3}$, weight decay $10^{-4}$, batch size 256) across 40 epochs with cosine annealing schedule.

2. **In-Domain Transfer Learning & Hard-Negative Mining (Phase II):**
   * Early CNN layers are frozen to retain low-level edge features, while the BiLSTM context layers and linear projection heads are fine-tuned using a curated dataset of $10,000$ annotated real-world operational plates [6].
   * Focus is placed on minimizing character-level confusion matrices between visually degenerate pairs ('0' vs. 'O', '8' vs. 'B', '1' vs. 'I') under a reduced learning rate ($\eta = 10^{-4}$) prior to INT8/FP16 quantization compilation [6, 7].

### 4.4 Stage 4: Indian CMVR Syntax Validation Engine
Post-processing enforces statutory Central Motor Vehicles Rules (CMVR) syntax through a deterministic validation grammar [7]:

$$\underbrace{[A-Z]^2}_{\text{State Code}} \quad \underbrace{[0-9]^{1,2}}_{\text{RTO District}} \quad \underbrace{[A-Z]^{0,3}}_{\text{Series Code}} \quad \underbrace{[0-9]^4}_{\text{Unique Registration}}$$

Positional token indices deterministically resolve common optical character confusions ('0' vs. 'O', '8' vs. 'B') based on positional syntax requirements.

---

## 5. Projected Performance & Latency Analysis

> **Methodological Disclaimer:** The evaluations presented in this section represent analytical performance projections synthesized from published literature benchmarks under explicit hardware and algorithmic assumptions. No localized physical empirical experiments were conducted for this proposal; all values serve as an architectural performance envelope.

### 5.1 Latency Decomposition Formulation
To eliminate ambiguity across operational frame rates, end-to-end processing latency ($T_{\text{total}}$) per input frame is formally defined as the cumulative sum of modular execution stages:

$$T_{\text{total}} = T_{\text{det}} + T_{\text{stn}} + T_{\text{crnn}} + T_{\text{post}}$$

Where:
* $T_{\text{det}}$: Execution latency of the anchor-free single-stage detector.
* $T_{\text{stn}}$: Differentiable TPS coordinate transformation and bilinear grid sampling latency.
* $T_{\text{crnn}}$: Recurrent sequence transcription latency across time steps.
* $T_{\text{post}}$: Deterministic regex parsing and Levenshtein token correction execution time.

Pipeline throughput for a non-batched, single-camera edge stream is bounded by:

$$\text{Throughput (FPS)} = \frac{1000}{T_{\text{total}} \text{ (in ms)}}$$

### 5.2 Plate Localization: Architecture Profiles vs. Projected Backbone
The table below contrasts established detector architectures against the projected operational envelope of the proposed anchor-free model on embedded edge hardware. Latencies represent synthesized FP16/INT8 deployment profiles on Jetson/edge accelerators:

| Architecture Paradigm | Architecture Formulation | Platform Profile / Target Precision | Reported / Projected mAP@0.5 (%) | Latency ($T_{\text{det}}$) | Parameter Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Faster R-CNN (ResNet-50)** | Ren et al. [2] | Two-Stage Baseline / FP32 | ~97.4% (ANPR Baseline Profile) | ~82.4 ms | 41.8 M |
| **SSD-MobileNetV2** | Liu et al. [8] | Jetson TX2 Benchmark / FP16 | ~88.2% (Mobile Baseline Profile) | ~14.1 ms | 4.3 M |
| **YOLOv8-Small** | Jocher et al. [9] | Jetson Orin Nano / FP16 | ~96.8% (Reported Edge Profile) | ~16.5 ms | 11.2 M |
| **Proposed Backbone (YOLOv10-N / RT-DETR)** | Projected (This Work) | Jetson Orin Nano / INT8 | **96.5% - 97.2%** *(Target Range)* | **~8.2 ms** *(Analytical Estimate)* | **~2.3 M** |

![Inference Latency vs. Precision](figures/detection_latency_tradeoff.png)

*Figure 2: Latency vs. Precision trade-off. While two-stage detectors provide high precision, their >80 ms latency is prohibitive for edge streaming. The projected anchor-free backbone targets sub-10 ms execution via NMS elimination.*

### 5.3 Sequence Transcription Fidelity under Perspective Skew
Standard OCR engines degrade significantly when acute camera placement introduces spatial tilt exceeding $35^\circ$. Rather than attributing specific CER percentages directly to foundational OCR architecture papers, the metrics below reflect an analytical evaluation envelope comparing unrectified baseline paradigms against the hypothesized resilience of our TPS-rectified pipeline:

| OCR Pipeline Architecture | Algorithmic Paradigm | Baseline Architecture Reference | Estimated Skewed CER ($>35^\circ$) | Estimated Dual-Line SRR (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Tesseract v5 (LSTM)** | Segmented Contours | Smith [10] (Engine Architecture) | ~40% - 45% (Unrectified Profile) | ~30% - 35% |
| **EasyOCR (CRAFT + ResNet)** | Word Bounding-Box | Baek et al. [11] / JaidedAI | ~18% - 20% (Unrectified Profile) | ~65% - 70% |
| **CRNN Baseline (No STN)** | Sequence-to-Sequence | Shi et al. [6] (Baseline Architecture) | ~14% - 16% (Unrectified Profile) | ~70% - 75% |
| **Proposed Pipeline (TPS + CRNN + CMVR)** | Unified Modular Sequence | *Projected Hypothesis (This Work)* | **2.0% - 3.5%** *(Hypothesized Range)* | **92.0% - 95.0%** *(Targeted)* |

*Note: Baseline CER ranges reflect synthesized behavioral degradation reported in unconstrained scene-text literature under acute perspective skew (>30°) without geometric rectification, rather than direct benchmark outputs of foundational architecture papers.*

![Character Error Rate under Skew](figures/cer_skew_distribution.png)

*Figure 3: Character Error Rate comparison under acute perspective skew (>35°). Rectification via TPS warping coupled with CMVR token correction is hypothesized to reduce CER significantly compared to unrectified baselines.*

### 5.4 Edge Compute & Throughput Budgeting
Based on the latency decomposition formula ($T_{\text{det}} \approx 8.2$ ms, $T_{\text{stn}} \approx 1.2$ ms, $T_{\text{crnn}} \approx 1.8$ ms, $T_{\text{post}} \approx 0.2$ ms $\rightarrow T_{\text{total}} \approx 11.4$ ms), the projected deployment characteristics across hardware tiers are outlined below:

| Target Platform Tier | Compute Envelope | Operating Precision | Projected $T_{\text{total}}$ | Projected Single-Stream Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Intel Core i7 (Baseline)** | 65W TDP | FP32 | ~18.2 ms | ~55 FPS |
| **NVIDIA Jetson Nano (4GB)** | 10W Power Mode | FP16 | ~28.5 ms | ~35 FPS |
| **NVIDIA Jetson Orin Nano** | 15W Power Mode | INT8 (TensorRT) | **~11.4 ms** | **~88 FPS** |
| **Raspberry Pi 4B (ARM-A72)** | 5W Budget | INT8 (ONNX-CPU) | ~84.6 ms | ~11 FPS |

![Edge Hardware Efficiency](figures/hardware_efficiency.png)

*Figure 4: Edge compute efficiency profile comparing power envelope (Watts) against projected pipeline inference throughput (FPS) across embedded accelerators under INT8/FP16 execution.*

---

## 6. Deployment Challenges & Architectural Limitations

Moving from computer vision models to real-world edge hardware located on roadside infrastructure brings unique engineering and architectural constraints into focus.

### 6.1 Quantization Drift and Hybrid Precision Budgeting

Edge accelerators like the NVIDIA Jetson Orin Nano deliver their performance at INT8 tensor precision.. Applying uniform post-training quantization (PTQ) across all parts of a modular sequence network can cause accuracy problems:

* **Convolutional Encoders:** The spatial feature representations used in single-stage detectors and early CNN layers handle INT8 scaling. These components remain stable when compressed.

* **Recurrent Sequence Layers:** The BiLSTM recurrent gates and the unnormalized logits feeding the Connectionist Temporal Classification (CTC) layer are sensitive to truncation. This makes them prone to confusion between looking characters, such as '0' and 'O' or '8' and 'B'.

* **Mitigation Strategy:** A hybrid approach is used. The localization sub-network ($T_{\text{det}}$) and the rectification sub-network ($T_{\text{stn}}$) run under INT8 TensorRT engines for speed. Meanwhile the CRNN transcription head ($T_{\text{crnn}}$) keeps FP16 precision to preserve accuracy.

### 6.2 Explicit Architectural Limitations

In line with academic standards we acknowledge the key limitations of this work:

1. **No In-Situ Empirical Validation:** The reported performance metrics, latency estimates ($T_{\text{total}} \approx 11.4$ ms) and character error rate ranges are based on literature benchmarks and hardware compute limits. They have not been verified through field deployment.

2. **Dual-Line Layout Aspect Ratio Gap:** While Thin Plate Spline (TPS) transformation helps correct distortions, standard square-aspect dual-line motorcycle plates pose challenges. They. Require splitting text lines during processing or using dual-pass sequence heads to avoid vertical compression when normalizing horizontally.

3. **Dynamic Environmental Extremes:** Conditions like headlights at night causing photon saturation or heavy mud and physical damage to license plates go beyond what standard homography and syntax rules can fix. These cases need sensor-level solutions, like -exposure HDR fusion to recover usable images.

---

## 7. Conclusion & Empirical Validation Roadmap

### 7.1 Conclusion
This paper formulated Edge-ANPR, a modular theoretical research architecture designed for low-latency Automatic Number Plate Recognition in unconstrained traffic ecosystems. By decoupling the pipeline into an anchor-free single-stage detector, a differentiable Thin Plate Spline Spatial Transformer Network (TPS-STN), a segmentation-free CRNN-CTC sequence decoder, and a deterministic CMVR syntax post-processor, the framework addresses the structural complexities of non-standard fonts and perspective distortions. Grounded in peer-reviewed literature benchmarks and an explicit latency decomposition model, the proposed pipeline is projected to sustain up to ~88 FPS at ~11.4 ms single-stream latency on edge-tier accelerators while substantially mitigating perspective Character Error Rates.

### 7.2 Empirical Validation Roadmap (Future Work)
To transition this research proposal into an empirically validated production system, subsequent work will follow a three-phase experimental protocol:
1. **Corpus Fine-Tuning:** Fine-tune the anchor-free detector and CRNN recognition heads on open-access unconstrained traffic benchmarks augmented with synthetic projective homography.
2. **Dual-Line Segmenter Optimization:** Implement a dynamic layout classification branch within the rectification stage to route single-line and dual-line plates into dedicated sequence decoders.
3. **Physical Hardware Benchmarking:** Deploy the compiled TensorRT INT8/FP16 execution graphs on physical NVIDIA Jetson Orin Nano hardware to capture empirical frame-to-frame latency via hardware timers, measure real-world thermal wattage under prolonged thermal throttling, and record raw Character Error Rates against live multi-lane CCTV feeds.

---

## References

[1] C. N. Anagnostopoulos, I. E. Anagnostopoulos, I. D. Psoroulas, V. Loumos, and E. Kayafas, "License Plate Recognition From Still Images and Video Sequences: A Survey," *IEEE Transactions on Intelligent Transportation Systems*, vol. 9, no. 3, pp. 377-391, 2008.

[2] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28, 2015.

[3] R. Laroca, E. Severo, L. A. Zanlorensi, L. S. Oliveira, G. R. Gonçalves, and W. R. Schwartz, "A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector," in *International Joint Conference on Neural Networks (IJCNN)*, 2018, pp. 1-10.

[4] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, "YOLOv10: Real-Time End-to-End Object Detection," *arXiv preprint arXiv:2405.14458*, 2024.

[5] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, "Spatial Transformer Networks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28, 2015.

[6] B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 39, no. 11, pp. 2298-2304, 2016.

[7] Ministry of Road Transport and Highways (MoRTH), "High Security Registration Plates (HSRP) Standards and Implementation Guidelines," *The Gazette of India, Central Motor Vehicles Rules (CMVR)*, 2019.

[8] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, "SSD: Single Shot MultiBox Detector," *European Conference on Computer Vision (ECCV)*, pp. 21-37, 2016.

[9] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," *GitHub repository*, https://github.com/ultralytics/ultralytics, 2023.

[10] R. Smith, "An Overview of the Tesseract OCR Engine," in *Ninth International Conference on Document Analysis and Recognition (ICDAR)*, vol. 2, 2007, pp. 629-633.

[11] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, "Character Region Awareness for Text Detection (CRAFT)," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 9365-9374.

