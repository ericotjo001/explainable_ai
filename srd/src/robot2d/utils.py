import os

def create_folder_if_not_exists(this_dir):
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)
        
def only_odd_map_size(args):
    try:
        H, W = args['map_size']
        H_vis, W_vis = args['vision_size']
        for x in [H,W,H_vis, W_vis]:
            assert(x%2==1)
    except:
        print('For now, we only allow odd map and vision sizes.')