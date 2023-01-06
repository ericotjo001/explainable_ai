DATA_DIR = "D:/shigoto/ImageNet/ILSVRC"

model =  'resnet34'   # 'resnet34'  , 'alexnet' 
# 2. existing XAI 
n_debug_imagennet = 10000
# 3.
img_index = 2
first_n_correct = 10
target_co = 15
gax_learning_rate = 0.1

# 1. training
PROJECT_ID = 'imgnet256_%s'%(str(model))
cmd = "python main_imgnet.py --mode train --PROJECT_ID %s --model %s --DATA_DIR %s --n_iter 120\n"%(str(PROJECT_ID), str(model),str(DATA_DIR))

# 2. existing XAI 
# METHODS = ['Saliency', 'InputXGradient']
METHODS = ['Saliency', 'InputXGradient', 'LayerGradCam', 'Deconvolution', 'GuidedBackprop', 'DeepLift'] # ALL AVAILABLE

cmd+='\n'
for method in METHODS:
    cmd += 'python main_imgnet.py --mode xai_collect --PROJECT_ID %s --model %s --DATA_DIR %s --split val --method %s --n_debug_imagenet %s\n'%(
        str(PROJECT_ID),str(model),str(DATA_DIR),str(method),str(n_debug_imagennet))
    cmd += 'python main_imgnet.py --mode xai_display_collection --PROJECT_ID %s --method %s\n'%(str(PROJECT_ID),str(method))
cmd+='\n'

cmd += "python main_imgnet.py --mode xai_display_boxplot --PROJECT_ID %s --DATA_DIR %s --model %s"%(str(PROJECT_ID), str(DATA_DIR), str(model))
cmd +='\n\n'


# 3. GAX
submethod = 'sum'
cmd += "python main_imgnet.py --mode gax --model %s --PROJECT_ID %s --DATA_DIR %s --submethod %s --first_n_correct %s --target_co %s --gax_learning_rate %s \n"%(
        str(model),str(PROJECT_ID), str(DATA_DIR), str(submethod), str(first_n_correct), str(target_co), str(gax_learning_rate))
cmd+='\n'
submethod = 'sum'
cmd+='python main_imgnet.py --mode gax_display --PROJECT_ID %s --model %s --DATA_DIR %s --submethod %s --img_index %s\n'%(
    str(PROJECT_ID),str(model),str(DATA_DIR), str(submethod),str(img_index),)
cmd+='\n'
cmd+='python main_imgnet.py --mode gax_coplot --PROJECT_IDs imgnet256_resnet34 imgnet256_alexnet --DATA_DIR %s --submethod %s \n'%(
    str(DATA_DIR), str(submethod),)

cmd += "python main_imgnet.py --mode xai_display_boxplot --PROJECT_ID %s --DATA_DIR %s --model %s"%(str(PROJECT_ID), str(DATA_DIR), str(model))
cmd +='\n\n'


# 4. GAX2
METHODS2 = ['LayerGradCam.conv1','LayerGradCam.layer1','LayerGradCam.layer2',
    'LayerGradCam.layer3','LayerGradCam.layer4'] # ALL AVAILABLE

cmd+='\n'
for method in METHODS2:
    cmd += 'python main_imgnet.py --mode xai_collect2 --PROJECT_ID %s --model %s --DATA_DIR %s --split val --method %s --n_debug_imagenet %s\n'%(
        str(PROJECT_ID),str(model),str(DATA_DIR),str(method),str(n_debug_imagennet))
cmd+='\n'
cmd += "python main_imgnet.py --mode xai_display_boxplot2 --PROJECT_ID %s --DATA_DIR %s --model %s"%(str(PROJECT_ID), str(DATA_DIR), str(model))
cmd +='\n\n'



txt = open('quickcommands.txt','w')
txt.write(cmd)
txt.close()