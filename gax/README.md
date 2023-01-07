This github repository contains all the codes required to replicate the results in the paper <b>Improving Deep Neural Network Classification Confidence using Heatmap-based eXplainable AI</b>. Link: [arxiv version](https://arxiv.org/abs/2201.00009).

<img src="https://drive.google.com/uc?export=view&id=1N4IDRJepmcK0-PkaqpSBDJdPYZRA7QLh" width="640"></img>

Summary. Given a classification model $net$, input $x$ and feature attribution $h=attr(net,x)$ (e.g. heatmap from Class Activation Mapping):
1. Augmentative eXplanation (AX) process is introduced, the basic form being $net(x+h)$
2. For some heatmap-based XAI methods, with AX process, there is a gap of the distribution of CO scores between the correct and false classification.
3. Overall accuracy improvement across samples in a dataset is not significant. Some methods cause severe degradation of accuracy.
4. Generative AX (GAX) is the direct optimization of CO score.

# Version 2.
The streamlined version of version 1 codes and additional experiments are now available.

## Installation
We use conda environment. The env.yml is provided.

The pytorch used here is torch==2.0.0.dev20221226+cu117, with torchvision installed using the following
```
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```
pytorch captum has been packaged into pytorch 2.0.

## Results
To replicate our experiment, simply follow the commands in misc/commands.txt.

To obtain our existing results, go to https://drive.google.com/drive/folders/1CEVOVmW3yJI7u1AQzJUpq-e9VndLIvTp?usp=share_link

## Data
The data we use are publicly available:
1. ImageNet
2. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
4. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
5. https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset

Extra notes:
For chest-xray COVID data, we fix the following irregularities manually:
COVID-3615.png (both images and masks)

# Version 1.
Everything related to version 1 code can all be found in legacy/v1 folder. 
