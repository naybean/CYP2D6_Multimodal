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

## How to use
**1. Code usage**
- If you want to use the project as Python scripts, please refer to the source codes in the **'src'** folder.
- If you prefer working with Jupyter Notebooks, please use the notebooks provided in the **'model'** folder.

**2. Training and Test Dataset**
- The datasets used for model training and evaluation are located under **'data/main_dataset/.'**
    - **'train_473'** : training data
    - **'test_149'** : test data
    - The numbers in the filenames indicate the number of samples in each dataset.
      
**3. One-hot Encoded Variant Features**
- One-hot encoded matrices for CYP2D6 variants are provided in CSV format.
- Each variant is encoded using 7 feature types:
    - A
    - T
    - G
    - C
    - SNP
    - INDEL
    - EXON/INTRON
- These files are located in **'data/main_dataset/onehot_7.'**

**4. Drug Input Representation and Unseen Drug Evaluation**
- Drug information is provided as **SMILES** strings, which are stored in the **'isomeric'** column of the dataset.
- To evaluate the model on external drugs, you can use the **'unseen drug'** file located at **'data/unseen.'**
- These unseen drugs were not used during training and are provided for evaluating the generalization performance of the model.

# Contact

If you have any questions or comments, please feel free to create an issue on github here, or email us:

- naybean990209@gmail.com


