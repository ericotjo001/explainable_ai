

def select_example(args, dargs,parser, TOGGLES):
    print('selecting example...')
    # parser is an argparse.ArgumentParser object
    

    if dargs['data'] == 'donut':
        print('donut example')
        
        """
        python main.py --mode example --data donut --submode train --kwidth 8 --show_fig_and_exit 0 --redir_id 0
        python main.py --mode example --data donut --submode eval_train --show_fig_and_exit 0 --redir_id 0
        python main.py --mode example --data donut --submode eval --show_fig_and_exit 0

        python main.py --mode example --data donut --submode showcase --classes 0 1 2 0 2 2 --idx 0 0 0 11 6 32
        python main.py --mode example --data donut --submode find --find_what activated_nodes --classes 0 0 0 2  --idx 0 7 18 32
        python main.py --mode example --data donut --submode find --find_what activated_nodes --input freeinput
        python main.py --mode example --data donut --submode find --find_what node_info --layers 1 2 2  --idx 5 0 6
        """

        from .donut.donut_example import first_donut_example
        first_donut_example(args, dargs, parser, TOGGLES)
    elif dargs['data'] == 'bigdonut':
        print('big donut example')
        """
        python main.py --mode example --data bigdonut --submode train --kwidth 64 --show_fig_and_exit 0 --redir_id 0
        python main.py --mode example --data bigdonut --submode eval_train --kwidth 64 --show_fig_and_exit 0 --redir_id 0
        python main.py --mode example --data bigdonut --submode eval --show_fig_and_exit 0

        python main.py --mode example --data bigdonut --submode showcase --classes 0 1 2 0 1 2 3 4 --idx 0 0 0 1 2 3 59 89
        python main.py --mode example --data bigdonut --submode find --find_what activated_nodes --classes 0 0 0 2  --idx 0 7 18 32
        python main.py --mode example --data bigdonut --submode find --find_what activated_nodes --input freeinput
        python main.py --mode example --data bigdonut --submode find --find_what node_info --layers 1 2 2  --idx 5 0 6
        """

        from .donut.bigdonut_example import big_donut_example
        big_donut_example(args, dargs, parser, TOGGLES)

    elif dargs['data'] == 'mnist':
        print('mnist')
        """
        python main.py --mode example --data mnist --submode train_dnn --epoch 4 --batch_size 16
        python main.py --mode example --data mnist --submode ces --classes 4 9 --firstn 240 240
        python main.py --mode example --data mnist --submode ces --classes 4 9 --firstn 240 240 --assess 4 0  4 130 4 152 4 777 9 0  9 239 9 555
        python main.py --mode example --data mnist --submode ces --classes 4 9 --firstn 240 240 --assess freeinput
        python main.py --mode example --data mnist --submode hyper

        python main.py --mode example --data mnist --submode train --kwidth 16 
        python main.py --mode example --data mnist --submode eval_train --kwidth 16 
        python main.py --mode example --data mnist --submode eval --kwidth 16  
        python main.py --mode example --data mnist --submode result --kwidth 16  

        python main.py --mode example --data mnist --submode showcase --kwidth 16 --classes 0 1 2 3 4 5 --idx 0 0 0 0 0 0  
        """

        from .mnist.mnist_example import first_mnist_example
        first_mnist_example(args, dargs, parser, TOGGLES)

    elif dargs['data'] == 'cifar':
        print('cifar')
        """
        python main.py --mode example --data cifar --submode train_dnn --epoch 4 --batch_size 16
        python main.py --mode example --data cifar --submode ces --classes 4 9 --firstn 240 240
        python main.py --mode example --data cifar --submode ces --classes 4 9 --firstn 240 240 --assess 4 0 4 24 4 43  4 555  9 0  9 210 9 666
        python main.py --mode example --data cifar --submode hyper

        python main.py --mode example --data cifar --submode train --kwidth 16 
        python main.py --mode example --data cifar --submode eval_train --kwidth 16  
        python main.py --mode example --data cifar --submode eval --kwidth 16  
        python main.py --mode example --data cifar --submode result --kwidth 16 

        python main.py --mode example --data cifar --submode showcase --kwidth 16 --classes 0 1 2 3 4 5 --idx 0 0 0 0 0 0  
        """
        from .cifar.cifar_example import run_cifar_example
        run_cifar_example(args, dargs, parser, )

    elif dargs['data'] == 'imagenet':
        print('imagenet')
        """
        python main.py --mode example --data imagenet --submode ces --classes 4 30 --kwidth 32 --firstn 900 900
        python main.py --mode example --data imagenet --submode ces --classes 4 30 --kwidth 32  --firstn 900 900 --assess 4 0  4 43  4 59  4 999 30 0  30 507  30 999
        # after rearranging problematic data.
        python main.py --mode example --data imagenet --submode ces --classes 4 30 --kwidth 32  --firstn 900 900 --assess 4 0  4 43  4 59  4 998 30 0  30 506  30 998
        """
        from .imagenet.imagenet_example import run_imagenet_example
        run_imagenet_example(args, dargs, parser)

    elif dargs['data'] == 'imagenetv2':
        print('imagenetv2')
        WARNING_MSG = """ 
        python main.py --mode example --data imagenetv2 --submode hyper 
          >>> WARNING! The above command is highly experimental. It seems like having too many classes at once
          is not yet feasible for kaBEDONN. For now, to use kaBEDONN on datasets with very large no of classes (>=1000),
          only use --submode ces, as demonstrated with 'imagenet' above <<<        
        """

        import warnings
        warnings.warn(WARNING_MSG)
        from .imagenetv2.imagenetv2_example import run_imagenetv2_example
        run_imagenetv2_example(args, dargs, parser)

    else:
        raise NotImplementedError('data choice not available')


""" Our experiments EXTENDED
mnist 
python main.py --mode example --data mnist --submode train --kwidth 32
python main.py --mode example --data mnist --submode eval_train --kwidth 32 
python main.py --mode example --data mnist --submode eval --kwidth 32
python main.py --mode example --data mnist --submode result --kwidth 32  


python main.py --mode example --data mnist --submode train --kwidth 48
python main.py --mode example --data mnist --submode eval_train --kwidth 48 
python main.py --mode example --data mnist --submode eval --kwidth 48
python main.py --mode example --data mnist --submode result --kwidth 48


python main.py --mode example --data mnist --submode train --kwidth 64
python main.py --mode example --data mnist --submode eval_train --kwidth 64 
python main.py --mode example --data mnist --submode eval --kwidth 64
python main.py --mode example --data mnist --submode result --kwidth 64  

cifar
python main.py --mode example --data cifar --submode train --kwidth 32
python main.py --mode example --data cifar --submode eval_train --kwidth 32  
python main.py --mode example --data cifar --submode eval --kwidth 32
python main.py --mode example --data cifar --submode result --kwidth 32

python main.py --mode example --data cifar --submode train --kwidth 48
python main.py --mode example --data cifar --submode eval_train --kwidth 48  
python main.py --mode example --data cifar --submode eval --kwidth 48
python main.py --mode example --data cifar --submode result --kwidth 48

python main.py --mode example --data cifar --submode train --kwidth 64
python main.py --mode example --data cifar --submode eval_train --kwidth 64  
python main.py --mode example --data cifar --submode eval --kwidth 64
python main.py --mode example --data cifar --submode result --kwidth 64
"""

def postprocessing(dargs):
    print('postprocessing entry')

    if dargs['mode'] == 'hyperboxplots':
        from .post_processing_utils.boxplots import arrange_boxplots_of_hyper_results 
        """
        python postprocessing.py --mode hyperboxplots
        """
        arrange_boxplots_of_hyper_results(dargs)
    else:
        raise NotImplementedError()