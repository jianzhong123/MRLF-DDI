# KG-CLDDI

## Overview

This repository contains source code for our paper "MGFF-DDI: A Multi-granularity Feature Fusion Framework for Drug-Drug Interaction Event Prediction via Multi-Relational GNN and Cross Strategy".

In this paper, we propose a new framework named MGFF-DDI, which integrates multi-granularity features derived from individual drugs, drug-drug pairs, and the DDI event graph to improve the accuracy of DDI event prediction. 

## Requirements

* Python == 3.7.3
* Pytorch == 1.8.1
* CUDNN == 11.1
* pandas == 1.3.0
* scikit-learn == 0.23.2
* rdkit-pypi == 2022.9.5
  
## Datasets
We preprocess the dataset collected by Ryu et al. (Ryu, J.Y., Kim, H.U., Lee, S.Y.: Deep learning improves prediction of drug–
drug and drug–food interactions. Proceedings of the national academy of sciences
115(18), E4304–E4311 (2018)), which includes 1,710 approved drugs and 192,284 DDIs associated with 86 DDI events. By selecting drugs that interact with other drugs and possess all four multimodal structural features, we retain 1,559 drugs and 171,141 DDIs associated with the 86 DDI events.
. The datasets are placed in the ./data/86datasets folder. The details are as follows:
* 86smiles.csv：includes the drug ids, DeepDDI index, index.
* DDI_event.csv:includes all types of the ddi events.
* S0 folder: includes the training and validation datasets based on 5-fold cross-validation split in the S0 setting.
* S1 folder: includes the training and validation datasets based on 5-fold cross-validation split in the S1 setting.
* S2 folder: includes the training and validation datasets based on 5-fold cross-validation split in the S2 setting.

In addition, we extract four types of multimodal structural features for individual drugs, including traditional Morgan and PubChem fingerprints, features extracted by the MolFormer model, and atomic-level features. These multimodal structural features are placed in the ./data/drugfeatures folder. The details are as follows:
* drugs1559_morgen86.npy: Morgan fingerprint features.
* drugs1559_pubchem86.npy: PubChem fingerprint features.
* drugs1559_mol86.npy: Features extracted by the MolFormer model.
* S0align_cross.npy, S1align_cross.npy, and S2align_cross.npy: Atomic-level features.
  
## Files:
The source code files are in the ./codes folder. The details are as follows:
* cross5_S0_data_helper.py: constructs the DDI event graph using the training dataset in the S0 setting.
* cross5_S1_data_helper.py, gridsearch_S2_data_helper.py： construct the DDI event graph using the training dataset in the S1 and S2 settings.
* gridsearch_crossModel.py: extracts the cross-features base on the drug-drug pairs.
* gridsearch_SEmodel.py: extracts the graph-based features from the DDI event graph.
  
## Running the code

The parameters are already set in the code files. You can run the following command to re-implement our work:

* > python Gridsearch_cross5_final_train_S0_86.py #Gridsearch_cross5_final_train_S1_86.py, Gridsearch_cross5_final_train_S2_86.py

## Contact

If you have any questions or suggestions with the code, please let us know. Contact Zhong Jian at jianzhong@csu.edu.cn
