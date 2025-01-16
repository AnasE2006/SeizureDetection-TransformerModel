# SeizureDetection-TransformerModel
A seizure detection model built using the Transformer model archietecture and EEGNet to identify seiz/bckg based on EEG data from the TUH EEG seizure corpus.

# Overview
This work was done as part of NSF undergraduate research with the goal of modifying code based on a previous seizure detection model from Yuanda Zhu and Prof. May D. Wang [(their works)](https://github.com/UnitedHolmes/seizure_detection_EEGs_transformer_BHI_2023) to be compatible with the newest data in the TUH EEG seizure corpus, Version 2.0.3.

# Data
The EEG data used to train and test the model can be found [here](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) from the Temple University Hospital EEG Seizure Corpus which is one of the largest accessible datasets for EEG seizure data. To request and gain access to the data, the initial steps on their page can be followed, and then the rsync command at the end of the page can be used to download the desired version of the data.

# Workflow
## Environment
After downloading the EEG seizure data, create and activate the conda environment provided by the environment.yml file using the commands:

```bash
conda env create -f environment.yml
conda activate eeg
```

## Preprocessing
The preprocessing work primarily takes place in dataPreprocessor.py which relies on functions in dataReader.py. To get it working set up the proper file paths to the downloaded EEG seizure data and where the preprocessed data will be saved inside of dataPreprocessor.py. Then run the command ```bash python3 dataPreprocessor.py```

## Training
To train the model, a run file is provided in run_eeg where the desired hyperparameters to pass into the model can be altered otherwise the command ```python3 model.py --eegnet_kernel_size x --eegnet_f1 x  --eegnet_D x --num_heads x``` can be used where the x's are replaced with hyperparameters.
