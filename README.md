# CYP2D6_MultiMod

Title: A Multimodal Deep Learning Approach for Predicting Drug Metabolism According to the CYP2D6 Genetic Variation

Authors: Yeabean Na, Hyunho Kim, Junho Kim, Myung-Gyun Kang and Sunyong Yoo

## Description

We present CYP2D6-MultiMod, a multimodal deep learning framework that predicts drug-specific metabolic phenotypes by integrating CYP2D6 genetic variants and molecular structure representations.
This repository contains the data preprocessing pipeline, model architectures (CNN + GCN fusion), and training / evaluation scripts used in our study.
Our goal is to provide an interpretable and extensible AI framework for understanding the relationship between genotype, drug structure, and metabolic activity, ultimately contributing to precision medicine and drug development.

- [Data](https://github.com/naybean/CYP2D6_MultiMod/tree/main/data/main%20dataset)
- [onehot_encoded_matrix](https://github.com/naybean/CYP2D6_Multimodal/tree/main/data/onehot_7)

## Dependency

`Python == 3.7`
`torch == 1.13.1+cu117`
`torch_geometric == 2.3.1`
`scikit-learn == 1.0.2`
`RDKit == 2023.03.2`
