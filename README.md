# SeizureDetection-TransformerModel
A seizure detection model built using the Transformer model archietecture and EEGNet to identify seiz/bckg based on EEG data from the TUH EEG seizure corpus.

# Overview
This work was done as part of NSF undergraduate research with the goal of modifying code based on a previous seizure detection model from Yuanda Zhu and Prof. May D. Wang [(their works)](https://github.com/UnitedHolmes/seizure_detection_EEGs_transformer_BHI_2023) to be compatible with the newest data in the TUH EEG seizure corpus, Version 2.0.3.

# Data
The EEG data used to train and test the model can be found [here](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) from the Temple University Hospital EEG Seizure Corpus which is one of the largest accessible datasets for EEG seizure data. To request and gain access to the data, the initial steps on their page can be followed, and then the rsync command at the end of the page can be used to download the desired version of the data.

# Workflow
After downloading the EEG seizure data, create and activate the conda environment provided by the environment.yml file using the commands:

```bash
conda env create -f environment.yml```

```bash
conda activate eeg```
