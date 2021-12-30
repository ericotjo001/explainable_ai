import os
from ..utils import create_if_not_exists, FastPickleClient

def manage_directories(args, ):
    DIRS = {}
    DIRS['ROOT_DIR'] = args['ROOT_DIR']
    DIRS['CHECKPOINT_DIR'] = 'checkpoint' 
    create_if_not_exists(DIRS['CHECKPOINT_DIR'])
    DIRS['PROJECT_DIR'] = os.path.join(DIRS['CHECKPOINT_DIR'],args['PROJECT_ID'])
    create_if_not_exists(DIRS['PROJECT_DIR'])
    DIRS['MODEL_DIR'] = os.path.join(DIRS['PROJECT_DIR'], 'main.model')
    # DIRS['SAVE_IMG_FOLDER'] = os.path.join(DIRS['PROJECT_DIR'],'existing_methods')
    # create_if_not_exists(DIRS['SAVE_IMG_FOLDER'])

    DIRS['RESULT_DIR'] = os.path.join(DIRS['PROJECT_DIR'],'test.result.json')

    if args['DATA_DIR'] is None:
        if args['mode']=='xai_display_collection':
            pass
        elif args['mode']=='eval_selected_image':
            pass
        else:
            raise RuntimeError('--DATA_DIR None is not available now. Set a proper directory')
    else:
        DIRS['DATA_DIR'] =  args['DATA_DIR']
    return DIRS



