This subrepo contains the code for [**Evaluating Weakly Supervised Object Localization Methods Right? A Study on Heatmap-based XAI and Neural Backed Decision Tree**](https://www.researchgate.net/publication/360609079_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_A_Study_on_Heatmap-based_XAI_and_Neural_Backed_Decision_Tree). The project has been discontinued.


**Important**. Make sure your setup run on both of the following github repos. 
1. https://github.com/alvinwan/neural-backed-decision-trees
2. https://github.com/clovaai/wsolevaluation
Our code structures mostly follow the two above repos.

General flow: 
1. Run WSOL measurements before NBDT (see xquickruns_resnet50.py)
2. Train model with NBDT process (see x_nbdt_main.py)
3. Re-run WSOL measurements, but for model trained after NBDT (again see xquickruns_resnet50.py)
4. display results

We illustrate this with the XAI method Saliency (and Resnet50)
Step 1. WSOL measurements. 
```
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency \
    --scoremap_mode saliency --scoremap_submode layer3  --debug_toggles 100000  
```

Step 2. NBDT training
```
python3 x_nbdt_main.py --mode inducegraph --arch ResNet50CAM 
python3 x_nbdt_main.py --mode train --arch ResNet50CAM --hierarchy induced-ResNet50CAM --epochs 2 --print_every 1 --batch-size 4 --resume 0 --eval 0 --lr 0.0001 --loss SoftTreeSupLoss --debug_toggles 0100000
python3 x_nbdt_main.py --mode train --arch ResNet50CAM --hierarchy induced-ResNet50CAM --epochs 2 --print_every 1 --batch-size 4 --resume 1 --eval 1 --lr 0.0001 --loss SoftTreeSupLoss --debug_toggles 0100000
```

Step 3. WSOL measurements on NBDT trained model.
```
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1 \
    --scoremap_mode saliency --scoremap_submode input  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer1  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer2  --debug_toggles 100000  
python3 xquickruns_resnet50.py   --scoremap_root=xresearchlog/resnet50_saliency_NBDT --NBDT 1\
    --scoremap_mode saliency --scoremap_submode layer3  --debug_toggles 100000  
```


Finally, some results are displayed with 
```
python3 xquickruns_resnet50.py --mode collate_results --scoremap_root xresearchlog.resnet50.nscc.1
python3 xquickruns_resnet50.py --mode collate_results --scoremap_root xresearchlog.resnet50.nscc.2 --NBDT 1
python xquickruns_resnet50.py --mode compare_results --scoremap_root xresearchlog.resnet50.nscc.1 --scoremap_root_compare xresearchlog.resnet50.nscc.

```


The experiment is conducted on pytorch 1.08, Ubuntu Red Hat 6.10 (Santiago). 
