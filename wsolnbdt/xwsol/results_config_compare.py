compare_config = {
    'saliency': {
        'normal':['resnet50_saliency',
            'resnet50_saliency_layer1',
            'resnet50_saliency_layer2',
            'resnet50_saliency_layer3',],
        'nbdt':[ 'resnet50_saliency_NBDT',
            'resnet50_saliency_NBDT_layer1',
            'resnet50_saliency_NBDT_layer2',
            'resnet50_saliency_NBDT_layer3',],
    },
    'deeplift':{
        'normal': [ 
            'resnet50_deeplift',    
            'resnet50_deeplift_layer1', 
            'resnet50_deeplift_layer2', 
            'resnet50_deeplift_layer3', 
            'resnet50_deeplift_layer4',],
        'nbdt': [            
            'resnet50_deeplift_NBDT',    
            'resnet50_deeplift_NBDT_layer1', 
            'resnet50_deeplift_NBDT_layer2', 
            'resnet50_deeplift_NBDT_layer3', 
            'resnet50_deeplift_NBDT_layer4',],
    },
    'gradcam':{
        'normal': [ 'resnet50_gradcam', 
            'resnet50_gradcam_layer1',
            'resnet50_gradcam_layer2',
            'resnet50_gradcam_layer3',
            'resnet50_gradcam_layer4',],
        'nbdt':[ 'resnet50_gradcam_NBDT', 
            'resnet50_gradcam_NBDT_layer1',
            'resnet50_gradcam_NBDT_layer2',
            'resnet50_gradcam_NBDT_layer3',
            'resnet50_gradcam_NBDT_layer4',],
    },
    'gbp':{
        'normal': ['resnet50_gbp',
            'resnet50_gbp_layer1',
            'resnet50_gbp_layer2',
            'resnet50_gbp_layer3',],
        'nbdt':[ 'resnet50_gbp_NBDT',
            'resnet50_gbp_NBDT_layer1',
            'resnet50_gbp_NBDT_layer2',
            'resnet50_gbp_NBDT_layer3',],
    }

}