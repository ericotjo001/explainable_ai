import utils.manage_console as mc
from utils.utils import check_default_aux_folders
from utils.descriptions.main_description import DESCRIPTION, WORKFLOW_DESCRIPTION, NULL_DESCRIPTION

if __name__ == '__main__':
    check_default_aux_folders() # assume it is run from the right directory, i.e. xai_basic
    mc3 = mc.ThreeTierConsoleModesManager()
    mcd = mc.DynamicConsole(['mode','mode2','mode3'])

    console_modes = mcd.arg_dict
    if console_modes['mode'] is None:
        print(console_modes)
        print(DESCRIPTION)
    elif console_modes['mode']=='data':
        import pipeline.data.entry as dt
        dt.select_data_mode(console_modes)
    elif console_modes['mode'] == 'training':
        import pipeline.training.entry as tr
        tr.select_training_mode(console_modes)
    elif console_modes['mode'] == 'evaluation':
        import pipeline.eval.entry as ev
        ev.select_evaluation_mode(console_modes)
    elif console_modes['mode'] == 'workflow':
        mode2 = console_modes['mode2']
        if mode2 is None:
            print(WORKFLOW_DESCRIPTION)
        elif mode2 == 'workflow1':
            from pipeline.workflow import *; workflow1()
        elif mode2 == 'workflow2':
            from pipeline.workflow2 import *; workflow2()
        elif mode2 == 'workflow3':
            from pipeline.workflow3 import *; workflow3()
        elif mode2 == 'workflow4':
            from pipeline.workflow4 import *; workflow4()
        else:
            print(NULL_DESCRIPTION+'\n'+WORKFLOW_DESCRIPTION)
    else:
        print(NULL_DESCRIPTION+'\n'+DESCRIPTION)
