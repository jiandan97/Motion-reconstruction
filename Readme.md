# Asymmetric Architecture toward Waveform Encoding for Motion Perception in Perovskite Photodetectors

## Abstract

Accurately perceiving motion from visual signals is fundamental to intelligent perception in dynamic environments. Conventional optical flow estimation relies on algorithmic processing of image sequences under the assumptions of brightness constancy and spatial coherence, which often fail in real-world conditions such as illumination fluctuation or occlusion. To address these challenges, we propose a strategy that enables motion sensing through in-sensor encoding. In this work, the introduction of a high dielectric constant aluminum oxide layer into perovskite photodiodes couples the polarization behavior to the carrier dynamics. Furthermore, the adoption of an asymmetric pixel architecture enhances motion sensitivity, thereby significantly enriching the motion information encoded in the waveform edges during the light–dark transition process. This strategy enables the detection of three-dimensional information—light intensity, motion speed, and direction—without the need for any additional optical or electrical components, a capability not achieved in previous studies. Leveraging machine learning algorithms, the device achieves almost ideal recognition accuracy over a broad speed range (0.2-30 m s-1), angular span (0-180°) and a range of light intensities. Furthermore, imaging simulations of a sensor array were performed to validate the feasibility of the proposed in-sensor encoding strategy for optical flow estimation. The array accurately estimated object motion under dynamic illumination and occlusion scenarios. This work offers a valuable perspective for motion perception through in-sensor encoding strategies.

## Schematic illustration of dataset and network architecture

![image-20251004160432686](functions/explain.png)

## Requirements

PyTorch ≥ 1.8

NumPy ≥ 1.20

Pandas ≥ 1.3

Matplotlib ≥ 3.4

PyWavelets (pywt) ≥ 1.1
