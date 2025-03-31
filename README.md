# MRLF-DDI

## Overview

This repository contains source code for our paper "MRLF-DDI: A Multi-view Representation Learning Framework for Drug-Drug Interaction Event Prediction".

In this paper, we propose MRLF-DDI, a multi-view representation learning framework for DDI event prediction. MRLF-DDI extracts features from three distinct views: individual drugs, drug–drug pairs, and the DDI graph. Notably, we introduce atomic-level drug features combined with bond-angle information for the first time in DDI event prediction.  

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
. The datasets are placed in the data folder. The details are as follows:
* 86smiles.csv：includes the drug ids, DeepDDI index, index of Ryu's dataset.
* DDIMDL_drug_smiles569.csv：includes the drug ids, DeepDDI index, index of Deng's dataset.
* Drugbank_DDI_event.csv: includes all types of Ryu's dataset.
* DDIMDL_DDI_event.csv:  includes all types of Deng's dataset.
* cross5datasets: includes the training and test datasets of Ryu's based on 5-fold cross-validation split in the S0, S1, and S2 setting.
* Dengcross5datasets: includes the training and test datasets of Deng's based on 5-fold cross-validation split in the S0, S1, and S2 setting.

In addition, we extract four types of multimodal structural features for individual drugs, including traditional Morgan and PubChem fingerprints, features extracted by the MolFormer model, and atomic-level features. These multimodal structural features are placed in the ./data/drugfeatures folder.
  
## Files:
The source code files are in the ./codes folder. The details are as follows:
* cross5_S0_data_helper.py: constructs the DDI event graph using the training dataset in the S0 setting.
* cross5_S1_data_helper.py, gridsearch_S2_data_helper.py： construct the DDI event graph using the training dataset in the S1 and S2 settings.
* gridsearch_crossModel.py: extracts the cross-features base on the drug-drug pairs.
* gridsearch_SEmodel.py: extracts the graph-based features from the DDI event graph.
  
## Running the code

The parameters are already set in the code files. You can run the following command to re-implement our work based on Ryu's dataset:

* > python Gridsearch_cross5_final_train_S0_86.py
* > python Gridsearch_cross5_final_train_S1_86.py
* > python  Gridsearch_cross5_final_train_S2_86.py
  
You can run the following command to re-implement our work based on Deng's dataset:
* > python Gridsearch_cross5_final_train_S0_Deng.py
* > python Gridsearch_cross5_final_train_S1_Deng.py
* > python Gridsearch_cross5_final_train_S2_Deng.py

## Contact

If you have any questions or suggestions with the code, please let us know. Contact Zhong Jian at jianzhong@csu.edu.cn
