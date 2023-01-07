LEGACY_MODE  ="""
######################## Important! ########################
# Legacy mode info

For ImageNet and Chest X Ray Pneumonia data,
our original experiments are run using v1 legacy code.
Please visit legacy/v1 and follow the instructions there.

Some difference between v2 and v1 include
the use of pytorch 2.0 and streamlined codebase 
############################################################
"""

def entry_imagenet(parser):
    print('entry_imagenet...')

    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--model', default=None, type=str, help=None)
    parser.add_argument('--IMGNET_DATA_DIR', default=None, type=str, help=None)

    parser.add_argument('--ax_method', default=None, type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary    

    if dargs['mode'] == 'val':
        print('val!')
        from .axpackage.ax_imagenet import imagenet_val
        imagenet_val(dargs, ax_method=None)
    elif dargs['mode'] == 'val_ax':
        print('val_ax!')
        from .axpackage.ax_imagenet import imagenet_val
        assert(dargs['ax_method'] is not None)
        imagenet_val(dargs, ax_method=dargs['ax_method'])   
    elif dargs['mode'] == 'vis_co_score':        
        from .axpackage.ax_imagenet import vis_co_score
        vis_co_score(dargs)
    else:
        print(LEGACY_MODE)


def entry_chestxray(parser):
    print('entry_chestxray...')

    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--model', default=None, type=str, help=None)
    parser.add_argument('--CHEST_XRAY_PNEUMONIA_DATA_DIR', default=None, type=str, help=None)

    parser.add_argument('--ax_method', default=None, type=str, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   

    if dargs['mode'] == 'test':
        print('chestxray_test!')
        from .axpackage.ax_chestxray_pneu import chestxray_test
        chestxray_test(dargs, ax_method=None)
    elif dargs['mode'] == 'test_ax':
        print('chestxray_test_ax!')
        from .axpackage.ax_chestxray_pneu import chestxray_test
        assert(dargs['ax_method'] is not None)
        chestxray_test(dargs, ax_method=dargs['ax_method'])    
    elif dargs['mode'] == 'vis_co_score':      
        from .axpackage.ax_chestxray_pneu import vis_co_score
        vis_co_score(dargs)
    else:
        print(LEGACY_MODE)



def entry_chestxray_covid(parser):
    print('entry_chestxray_covid...')

    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--CHEST_XRAY_COVID_DATA_DIR', default=None, type=str, help=None)

    parser.add_argument('--ax_method', default=None, type=str, help=None)
    parser.add_argument('--label_name', default='debug', type=str, help=None)

    
    # trainval
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--n_epochs', default=1, type=int, help=None)
    parser.add_argument('--VAL_TARGET', default=0.8, type=float, help=None)


    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   


    if dargs['mode'] == 'visdata':
        from .trainval import chestxray_covid_visdata
        chestxray_covid_visdata(dargs)
    elif dargs['mode'] == 'trainval':
        from .trainval import chestxray_covid_trainval
        chestxray_covid_trainval(dargs)
    elif dargs['mode'] == 'test': 
        from .axpackage.ax_chestxray_covid import chestxray_covid_test
        chestxray_covid_test(dargs, ax_method=None) 
    elif dargs['mode'] == 'test_ax': 
        from .axpackage.ax_chestxray_covid import chestxray_covid_test
        assert(dargs['ax_method'] is not None)
        chestxray_covid_test(dargs, ax_method=dargs['ax_method']) 
    elif dargs['mode'] == 'vis_co_score':      
        from .axpackage.ax_chestxray_covid import vis_co_score
        vis_co_score(dargs)
    else:
        raise NotImplementedError()


def entry_creditcardfraud(parser):
    print('entry_creditcardfraud...')

    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--CREDIT_FRAUD_DATA_DIR', default=None, type=str, help=None)

    parser.add_argument('--ax_method', default=None, type=str, help=None)
    parser.add_argument('--label_name', default='debug', type=str, help=None)

    # reconstruct data
    parser.add_argument('--n_split_negative', default=(2500,2500,2500), help=None)
    parser.add_argument('--n_mult_positive', default=5, type=int, help=None)


    # trainval
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--n_epochs', default=1, type=int, help=None)
    parser.add_argument('--VAL_TARGET', default=0.8, type=float, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   
    
    if dargs['mode'] == 'visdata':
        from .trainval import creditcardfraud_vis_data
        creditcardfraud_vis_data(dargs)
    elif dargs['mode'] == 'reconstruct_data':
        from .trainval import creditcardfraud_reconstruct_data
        creditcardfraud_reconstruct_data(dargs)
    elif dargs['mode'] == 'trainval':
        from .trainval import creditcardfraud_trainval
        creditcardfraud_trainval(dargs)

    elif dargs['mode'] == 'test': 
        from .axpackage.ax_creditcardfraud import ax_creditcardfraud_test
        ax_creditcardfraud_test(dargs, ax_method=None) 
    elif dargs['mode'] == 'test_ax': 
        from .axpackage.ax_creditcardfraud import ax_creditcardfraud_test
        assert(dargs['ax_method'] is not None)
        ax_creditcardfraud_test(dargs, ax_method=dargs['ax_method']) 
    elif dargs['mode'] == 'vis_co_score':      
        from .axpackage.ax_creditcardfraud import vis_co_score
        vis_co_score(dargs)
    elif dargs['mode'] == 'test_remaining':
        from .axpackage.ax_creditcardfraud import ax_creditcardfraud_test_remaining
        ax_creditcardfraud_test_remaining(dargs)
    else:
        raise NotImplementedError()

def entry_drybean(parser):
    print('entry_drybean...')
    parser.add_argument('--mode', default=None, type=str, help=None)
    parser.add_argument('--DRYBEAN_DATA_DIR', default=None, type=str, help=None)

    parser.add_argument('--ax_method', default=None, type=str, help=None)    
    parser.add_argument('--label_name', default='debug', type=str, help=None)

    # trainval
    parser.add_argument('--batch_size', default=16, type=int, help=None)
    parser.add_argument('--n_epochs', default=1, type=int, help=None)
    parser.add_argument('--VAL_TARGET', default=0.8, type=float, help=None)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary   

    if dargs['mode'] == 'visdata':
        from .trainval import drybean_vis_data
        drybean_vis_data(dargs)
    elif dargs['mode'] == 'reconstruct_data':
        from .trainval import drybean_reconstruct_data
        drybean_reconstruct_data(dargs)
    elif dargs['mode'] == 'trainval':
        from .trainval import drybean_trainval
        drybean_trainval(dargs)

    elif dargs['mode'] == 'test': 
        from .axpackage.ax_drybean import ax_drybean_test
        ax_drybean_test(dargs, ax_method=None) 
    elif dargs['mode'] == 'test_ax': 
        from .axpackage.ax_drybean import ax_drybean_test
        assert(dargs['ax_method'] is not None)
        ax_drybean_test(dargs, ax_method=dargs['ax_method']) 
    elif dargs['mode'] == 'vis_co_score':      
        from .axpackage.ax_drybean import vis_co_score
        vis_co_score(dargs)
    else:
        raise NotImplementedError()
