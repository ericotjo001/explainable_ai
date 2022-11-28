!!! This project is currently due to be updated. Current version will be deprecated soon !!!

## Augmentative eXplanation and the Distributional Gap of Confidence Optimization Score

This folder contains the codes for [Augmentative eXplanation and the Distributional Gap of Confidence Optimization Score](https://arxiv.org/abs/2201.00009).

This project Augmentative Explanation and Confidence Optimization score to use heatmap/saliency explanations for improving predictive probability. This project also introduces GAX as a heatmap generation methods as a study on eXplainable AI. Confidence Optimization score is optimized through GAX, thus our heatmaps can be used to increase predictive probability.

Examples of commands to reproduce our results can be found in _quick_start folder. Running sequence.py files will generate quickcommands.txt which contains the direct python commands. Settings can be changed in sequence.py according to users' preference.

We provide brief descriptions of the commands using Chest X-Ray example. 

***Main files***. main_pneu.py is main script to run GAX on Chest X-Ray pneumonia dataset, main_imgnet.py for ImageNet. Assume we run every scripts from the gax folder, where these two main files are found.


Use the following to reshuffle data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, whose validation data folder is not optimal. Assume you extract train, val and test data into data/chest_xray folder.
```
python main_pneu.py --mode data_reshuffle
```

Fine-tuning Resnet34 models and evaluations for Pneumonia dataset:
```
python main_pneu.py --mode train --PROJECT_ID pneu256n_1 --model resnet34 --n_iter 64 --batch_size 4 --realtime_print 1 --min_iter 10 --n_debug 16
python main_pneu.py --mode evaluate --model resnet34 --PROJECT_ID pneu256n_1  --n_debug 0
```

Collecting CO scores for existing XAI methods, such as Saliency, for train, validation and test data respectively.
```
python main_pneu.py --mode xai_collect --model resnet34 --PROJECT_ID pneu256n_1 --method Saliency --split train --realtime_print 1 --n_debug 0
python main_pneu.py --mode xai_collect --model resnet34 --PROJECT_ID pneu256n_1 --method Saliency --split val --realtime_print 1 --n_debug 0
python main_pneu.py --mode xai_collect --model resnet34 --PROJECT_ID pneu256n_1 --method Saliency --split test --realtime_print 1 --n_debug 0
```

Collected CO scores for training data samples are shown as an individiual histogram.
```
python main_pneu.py --mode xai_display_collection --model resnet34 --PROJECT_ID pneu256n_1 --method Saliency --split train
```

Once data collection is done, CO scores across different XAI methods are shown together as boxplots.
```
python main_pneu.py --mode xai_display_boxplot --PROJECT_ID pneu256n_1
```

Running GAX for the first 100 data samples that are correctly predicted by the fine-tuned models. --label NORMAL and PNEUMONIA are for images stored in the respective folders, where Chest X-Ray images indicate whether the patients are healthy or suffer from pneumonia.
```
python main_pneu.py --mode gax --PROJECT_ID pneu256n_1 --model resnet34 --label NORMAL --split test --first_n_correct 100 --target_co 48 --gax_learning_rate 0.1
python main_pneu.py --mode gax --PROJECT_ID pneu256n_1 --model resnet34 --label PNEUMONIA --split test --first_n_correct 100 --target_co 48 --gax_learning_rate 0.1
```

Viewing GAX results! You will see a pop up with slider, like 

<img src="_quick_start/demo_img.PNG" width="700px">

```
python main_pneu.py --mode gax_display --PROJECT_ID pneu256n_1 --img_name IM-0009-0001.jpeg --label NORMAL --split test --submethod sum
python main_pneu.py --mode gax_display --PROJECT_ID pneu256n_1 --img_name person1_virus_6.jpeg --label PNEUMONIA --split test --submethod sum
```
