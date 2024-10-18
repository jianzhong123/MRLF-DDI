# KG-CLDDI

## Overview

This repository contains source code for our paper "MGFF-DDI: A Multi-granularity Feature Fusion Framework for Drug-Drug Interaction Event Prediction via Multi-Relational GNN and Cross Strategy".

In this paper, we propose a new framework named MGFF-DDI, which fuses multi-granularity features by constructing a multi-relational network for DDI and introducing a cross-strategy. 

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
* OurDDI Dataset.csv：The DDI Dataset.
* S0_drug1559_train86.csv, S0_drug1559_valid86.csv, and S0_drug1559_test86.csv: The training, validation, and test datasets for the S0 setting.
* S12_drug1559_train86.csv:The training dataset for the S1 and S2 settings.
* S1_drug1559_similar_valid86.csv, S1_drug1559_similar_test86.csv: The validation, and test datasets for the S1 setting.
* S2_drug1559_similar_valid86.csv, S2_drug1559_similar_test86.csv: The validation, and test datasets for the S2 setting. 

In addition, we extract four types of multimodal structural features for individual drugs, including traditional Morgan and PubChem fingerprints, features extracted by the MolFormer model, and atomic-level features. These multimodal structural features are placed in the ./data/drugfeatures folder. The details are as follows:
* drugs1559_morgen86.npy: Morgan fingerprint features.
* drugs1559_pubchem86.npy: PubChem fingerprint features.
* drugs1559_mol86.npy: Features extracted by the MolFormer model.
* S0_drugs1559_alignn86.npy, S12_drugs1559_alignn86.npy:Atomic-level features.
  
## Files:
The source code files are in the ./codes folder. The details are as follows:
* S0data_helper.py: Construct the DDI event graph using the training dataset in the S0 setting.
* data_helper.py： Construct the DDI event graph using the training dataset in the S1 and S2 settings.
* dataloader.py: Knowledge graph, training and test data sets are loaded and parsed, and a drug-drug bipartite graph is constructed.
* crossModel.py: The implementation of thecross strategy base on the drug pairs.
* SEmodel.py: The implementation of SE-GNN based on the DDI event graph.
  
## Running the code

The parameters are already set in the code files. You can run the following command to re-implement our work:

* > python train_S0_86.py #train_S1_86.py,train_S2_86.py

## Contact

If you have any questions or suggestions with the code, please let us know. Contact Zhong Jian at jianzhong@csu.edu.cn
