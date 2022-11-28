EVAL_DESCRIPTION= """
python main.py --mode evaluation
python main.py --mode evaluation --mode2 resnet34_ten_classes
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 branch_validation_info
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 xai
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 unpack_and_pointwise_process
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 view_gallery
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 roc
python main.py --mode evaluation --mode2 resnet34_ten_classes --mode3 xai_singleton_scope 

python main.py --mode evaluation --mode2 alexnet_ten_classes
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 branch_validation_info
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 xai
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 unpack_and_pointwise_process
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 view_gallery
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 roc
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 xai_singleton_scope 

python main.py --mode evaluation --mode2 vgg_ten_classes
python main.py --mode evaluation --mode2 vgg_ten_classes --mode3 branch_validation_info
python main.py --mode evaluation --mode2 vgg_ten_classes --mode3 xai
python main.py --mode evaluation --mode2 vgg_ten_classes --mode3 unpack_and_pointwise_process
python main.py --mode evaluation --mode2 vgg_ten_classes --mode3 roc
python main.py --mode evaluation --mode2 vgg_ten_classes --mode3 view_gallery
"""