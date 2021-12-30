PROJECT_ID = 'pneu256n_1'
realtime_print = 1

model = 'resnet34' # 'resnet34', 'alexnet'
# 1. training
train_iter = 64
train_batch_size = 4
train_n_debug = 16 # set to 0 for non-debug
min_iter = 10 # 2400

# 2. evaluation
eval_n_debug = 0 # 24

# 3. collect explanation gain (co score)
# METHODS = ['Saliency', 'LayerGradCam']
METHODS = ['Saliency', 'InputXGradient', 'LayerGradCam', 'Deconvolution', 'GuidedBackprop', 'DeepLift'] # ALL AVAILABLE

# 4. GAX
GAX_SAMPLES = [
    {'img_name':'IM-0009-0001.jpeg',
        'label':'NORMAL', 
        'split':'test' , 
        'gax_learning_rate': 0.001,
    },
    {'img_name':'person1_virus_6.jpeg',
        'label':'PNEUMONIA', 
        'split':'test' ,
        'gax_learning_rate': 0.001,
    },
]
target_co = 48
first_n_correct = 100
gax_learning_rage = 0.1

# 1. training
this_str = "python main_pneu.py --mode train --PROJECT_ID %s --model %s --n_iter %s --batch_size %s --realtime_print %s"%(
    str(PROJECT_ID), str(model), str(train_iter), str(train_batch_size), str(realtime_print))
this_str += " --min_iter %s --n_debug %s"%(str(min_iter),str(train_n_debug))

# 2. evaluation
this_str += "\npython main_pneu.py --mode evaluate --model %s --PROJECT_ID %s "%(str(model),str(PROJECT_ID))
this_str += " --n_debug %s"%(str(eval_n_debug))
this_str += '\n'

# 3. collect explanation gain (co score)
for METHOD in METHODS:
    for split in ['train','val','test']:
        this_str += "\npython main_pneu.py --mode xai_collect --model %s "%(str(model))
        this_str += "--PROJECT_ID %s --method %s --split %s --realtime_print %s --n_debug %s"%(
            str(PROJECT_ID),str(METHOD), str(split), str(realtime_print), str(eval_n_debug))
for METHOD in METHODS:
    for split in ['train','val','test']:
        this_str += "\npython main_pneu.py --mode xai_display_collection --model %s "%(str(model))
        this_str += "--PROJECT_ID %s --method %s --split %s"%(str(PROJECT_ID), str(METHOD), str(split))
this_str += "\n\npython main_pneu.py --mode xai_display_boxplot --PROJECT_ID %s"%(str(PROJECT_ID))
this_str += '\n'

# 4. GAX

for label in ['NORMAL','PNEUMONIA']:
    this_str+='\npython main_pneu.py --mode gax --PROJECT_ID %s --model %s --label %s --split %s --first_n_correct %s --target_co %s --gax_learning_rate %s'%(
        str(PROJECT_ID),str(model),str(label),str('test'),str(first_n_correct), str(target_co) , str(gax_learning_rage))

this_str+='\n'
for this_sample in GAX_SAMPLES:
    for submethod in ['sum']:
        this_str += "\npython main_pneu.py --mode gax_display "%()
        this_str += "--PROJECT_ID %s "%(str(PROJECT_ID), )
        this_str += "--img_name %s --label %s --split %s --submethod %s"%(str(this_sample['img_name']),
            str(this_sample['label']), str(this_sample['split']) ,str(submethod))

this_str += '\n'

def extend_string_for_gax(this_str):
    this_str += "\npython main_pneu.py --mode gax "%()
    this_str += "--PROJECT_ID %s "%(str(PROJECT_ID), )
    this_str += "--img_name %s --label %s --split %s "%(str(this_sample['img_name']),
        str(this_sample['label']), str(this_sample['split']) )
    this_str += "--target_co %s --gax_learning_rate %s "%(str(target_co),
        str(this_sample['gax_learning_rate']),)
    return this_str

this_str+='\nOPTIONAL: testing optimization for individual images'
for this_sample in GAX_SAMPLES:
    this_str = extend_string_for_gax(this_str)
this_str += '\n'


txt = open('quickcommands.txt','w')
txt.write(this_str)
txt.close()