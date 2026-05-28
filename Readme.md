# Asymmetric Architecture toward Waveform Encoding for Motion Perception in Perovskite Photodetectors

## Abstract

Reliable motion perception from visual signals is essential for intelligent systems, but conventional optical-flow methods often degrade under changing illumination or occlusion. Here we present an in-sensor encoding strategy that extracts motion information directly from perovskite photodiodes. By inserting a high-dielectric-constant aluminum oxide (Al₂O₃) layer, polarization behavior is coupled with carrier dynamics, and an asymmetric pixel design further strengthens motion-dependent signals at light-dark transitions. These devices encode light intensity, motion speed, and direction in a single photocurrent waveform without extra optical or electrical components. With machine-learning algorithms, the device achieves almost ideal recognition accuracy over a broad speed range (0.2-30 m s-1), angular span (0-180°) and a range of light intensities. We also demonstrated real-time human-machine interaction by converting finger-motion photocurrents into commands for a block-stacking game. A 5 × 5 array and large-array simulations further confirmed scalable motion estimation under dynamic illumination and occlusion. This work provides a practical route toward compact visual motion sensors.

## Framework

![image-20251004160432686](functions/explain.png)

## Performance

![Performance](functions/fig.png)

## Environment

PyTorch ≥ 1.8

NumPy ≥ 1.20

Pandas ≥ 1.3

Matplotlib ≥ 3.4

PyWavelets (pywt) ≥ 1.1
