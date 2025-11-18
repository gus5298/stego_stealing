# *Steganographic Data Exfiltration for Model Stealing: A Case Study on Energy Critical Infrastructure IEC 61850 Datasets*

This repository contains the implementation used in our IEEE BigData 2025 paper on steganographic model exfiltration attacks in image-based network intrusion detection systems (NIDS).  
As ML-based NIDS are increasingly converting raw network traffic into image representations, they inadvertently enable a covert channel: embedding sensitive ML model data inside those images.

## Problem Overview  
Machine learning (ML)-based NIDS convert packet streams into image form (e.g., grayscale images) to leverage computer-vision architectures. These images, though benign in appearance, can become carriers for steganographic payloads. Traditional Data Loss Prevention (DLP) and Intrusion Detection Systems (IDS) do not reliably detect such concealed channels.

## Research Contributions  
Stego-Stealing implements and evaluates three embedding pipelines for hiding model secrets (architecture, hyperparameters, weights, optimizer state) inside NIDS images:

- **LSB Replacement**: straightforward embedding in the least significant bits of image pixels.  
- **PRNG-Controlled Dual-Layer Embedding**: pseudorandom pixel selection + hashing to increase capacity and reduce detectability.  
- **Saliency-Guided Layered Embedding**: using explainable-AI (XAI) techniques to locate high-saliency pixels and embed payloads there; explores the notion of steganographic adversarial examples.

We built a CNN-based NIDS testbed on grayscale packet-derived images to evaluate the embedding pipelines, measuring image distortion, classifier accuracy impact, embedding capacity, and detectability under channel transformations.

## Key Findings  
- The LSB replacement method remains visually imperceptible and does not degrade classifier accuracy for moderate payloads.  
- The PRNG-controlled method supports larger payloads (e.g., stolen model weights) with minimal distortion and impact on the classifier.  
- The saliency-guided embedding method, while innovative, suffers from lower capacity and negatively impacts classifier accuracy.  
- Existing DLP/IDS controls in image-based workflows fail to detect these steganographic channels.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{sanchez2025stegostealing,
  author    = {Gustavo Sánchez Collado},
  title     = {Steganographic Data Exfiltration for Model Stealing: A Case Study on Energy Critical Infrastructure IEC 61850 Datasets},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2025}
}
