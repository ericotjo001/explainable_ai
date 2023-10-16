# Enhancing the Confidence of Deep Learning Classifiers via Interpretable Saliency Maps
This github repository contains all the codes required to replicate the results in the paper with the above title. Link: [arxiv version](https://arxiv.org/abs/2201.00009). The paper has also been published in the journal [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231223009487)!

<img src="https://drive.google.com/uc?export=view&id=1N4IDRJepmcK0-PkaqpSBDJdPYZRA7QLh" width="640"></img>

Figure 1. *Left*: user interface for GAX. *Right*. CO scores shown in box plots for each saliency-based XAI method; correct and wrong predictions are plotted separately. 

In the example below, we run the entire pipeline from training to XAI evaluation on a COVID dataset (only Saliency and InputXGradient are shown for the AX process). The full instructions can be found in misc/commands.txt.
```
python main.py --data chestxray_covid --mode visdata --CHEST_XRAY_COVID_DATA_DIR C:/data/COVID-19_Radiography_Dataset

python main.py --data chestxray_covid --mode trainval --CHEST_XRAY_COVID_DATA_DIR C:/data/COVID-19_Radiography_Dataset --n_epochs 256 --VAL_TARGET 0.85 --label_name project01
python main.py --data chestxray_covid --mode test --CHEST_XRAY_COVID_DATA_DIR C:/data/COVID-19_Radiography_Dataset --label_name project01

python main.py --data chestxray_covid --mode test_ax --ax_method Saliency --CHEST_XRAY_COVID_DATA_DIR C:/data/COVID-19_Radiography_Dataset --label_name project01
python main.py --data chestxray_covid --mode test_ax --ax_method InputXGradient --CHEST_XRAY_COVID_DATA_DIR C:/data/COVID-19_Radiography_Dataset --label_name project01

python main.py --data chestxray_covid --mode vis_co_score --label_name project01
```

The outline of the core mechanisms are shown below. In essence, CO score compares net(x+attr) and net(x).
```
normalize = get_transform(transformtype='one_channel')
ax = AugEplanation(method=ax_method, model=model)

y_baseline = net(x)
attr = ax.get_attribution_by_method(net, x)
attr = ax.normalize_attr(attr)
y = net(x + attr)

co_score = compute_co_score(y_baseline, y, y0, n_class)
```

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


Previous titles of this paper: (1) Augmentative eXplanation and the Distributional Gap of Confidence Optimization Score (2) Improving Deep Neural Network Classification Confidence using Heatmap-based eXplainable AI
