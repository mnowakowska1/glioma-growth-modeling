# Computational Modeling of Glioma Growth

## Overview

This project explores the computational modeling of glioma growth through mathematical models, numerical simulations and medical imaging data.

It originally started as a university coursework assignment, but my interest in mathematical modeling, machine learning and their applications in medicine motivated me to continue developing it independently.

For now, it is a personal learning project driven by curiosity and experimentation, with the potential to become the foundation of my future engineering thesis.

## Current implementation

The current version is based on the Fisher–Kolmogorov reaction-diffusion equation and uses multimodal MRI data to simulate the spatial growth of glioma.

The implementation includes:

- Two-dimensional tumor growth simulation
- MRI-based diffusion and proliferation maps using FLAIR, ADC and ASL-CBF images
- Finite difference discretization and explicit Euler time integration
- Parameter optimization using Grid Search and Hill Climbing
- Evaluation using the Dice similarity coefficient
- Visualization of simulated and reference tumor regions

At this stage, the model focuses on reconstructing the tumor extent on a selected MRI slice rather than predicting its future progression.

## Future directions

In the future, I would like to explore how mathematical modeling and machine learning can be combined to predict glioma progression based on historical patient imaging data.

I am particularly interested in the possibility of developing personalized predictions that can be updated and refined as new MRI scans become available.

Potential directions include three-dimensional simulations, longitudinal medical image analysis, patient-specific parameter estimation and the integration of machine learning with mathematical tumor growth models.

The exact methodology is still open and will evolve as I gain more knowledge, explore scientific literature and experiment with different approaches.

## Technologies

Python · NumPy · Matplotlib · NiBabel

## Project status

**Work in progress — Personal research and learning project.**

This project is intended for educational and exploratory research purposes only and is not clinically validated.
