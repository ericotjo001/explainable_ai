    DEBUG_MODE = 1
    FULL_DATA_MODE = 0
    MACHINE_OPTION = None # 'NSCC'

    DEBUG_MODE = 0
    FULL_DATA_MODE = 1
    MACHINE_OPTION = None # 'NSCC'


    TOGGLE = {'DATA':1,
        'PART1':0, 'PART1.2':0,
        'PART2':0, 'PART2.2':0,
        'PART3':1, 'PART3.2':1, 'PART3.3':1,
    }

    TOGGLE = {'DATA':1,
        'PART1':1, 'PART1.2':1,
        'PART2':1, 'PART2.2':1,
        'PART3':1, 'PART3.2':1, 'PART3.3':1,
    }


# NSCC MODE
    DEBUG_MODE = 0
    FULL_DATA_MODE = 1
    MACHINE_OPTION = 'NSCC'
    TOGGLE = {'DATA':0,
        'PART1':0, 'PART1.2':0,
        'PART2':0, 'PART2.2':0,
        'PART3':1, 'PART3.2':1, 'PART3.3':1,
    }

    # Testing with resnet34
    XAI_MODES = [ 
        'Saliency', 
        # 'IntegratedGradients', 
        'InputXGradient', 
        'DeepLift', 
        'GuidedBackprop', 
        'GuidedGradCam',
        'Deconvolution', 

        'GradientShap',
        'DeepLiftShap',
        ]

    BRANCHES_PER_MODEL = [1,2,3,4,5] # each number will become a new optimized model [PART 2]