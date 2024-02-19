result_config = {
    'METHOD_LIST': [
        [
            'resnet50_cam',
            'resnet50_saliency_NBDT',
            'resnet50_saliency_NBDT_layer1',
            'resnet50_saliency_NBDT_layer2',
            'resnet50_saliency_NBDT_layer3',
        ],

        [
            'resnet50_cam',
            'resnet50_deeplift_NBDT',    
            'resnet50_deeplift_NBDT_layer1', 
            'resnet50_deeplift_NBDT_layer2', 
            'resnet50_deeplift_NBDT_layer3', 
            'resnet50_deeplift_NBDT_layer4',
        ],
        
        [
            'resnet50_cam',
            'resnet50_gradcam_NBDT', 
            'resnet50_gradcam_NBDT_layer1',
            'resnet50_gradcam_NBDT_layer2',
            'resnet50_gradcam_NBDT_layer3',
            'resnet50_gradcam_NBDT_layer4',
        ],
        
        [
            'resnet50_cam',
            'resnet50_gbp_NBDT',
            'resnet50_gbp_NBDT_layer1',
            'resnet50_gbp_NBDT_layer2',
            'resnet50_gbp_NBDT_layer3',
        ],
    ]
}