import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_activation(x, epsilon=1e-4): 
    return epsilon/(epsilon+x**2)

def numpy_to_pytorch_tensor(x):
    # x is (H,W) shape
    return torch.tensor([[x]]).to(torch.float)

def sort_dictionary_by_values(d, reverse=False):
    dsorted = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    return dsorted

device = None# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Robot2NN(nn.Module):
    def __init__(self, args):
        super(Robot2NN, self).__init__()
        self.args = args

        self.ACTIONS = ['RIGHT','DOWN','LEFT','UP']
        
        self.TILES_RAW = {
            'target':np.array((255,215,0)),
            'dirt': np.array((139,69,19)), 
            'grass':np.array((0,128,0)),
        }
        self.TILES = { x :y/255. for x,y in self.TILES_RAW.items()}  
        
        # Modules
        self.initiate_tile_based_modules()

        # others
        self.n_plans = args['n_plans']
        self.recognition_threshold = 1e-4
        self.tanh = nn.Tanh()


    def initiate_tile_based_modules(self):
        self.tile_modules = nn.ModuleDict()
        self.tile_modules['self'] = DeconvSeq(5,kernel_size=3, mid_val=1., side_val=0.1)
        self.tile_modules['target'] = DeconvSeq(5,kernel_size=3, mid_val=1., side_val=0.1)
        for tile_name in self.TILES:    
            if tile_name=='target':
                continue
            self.tile_modules[tile_name] = DeconvSeq(5,kernel_size=3, mid_val=1., side_val=0.1)
        
        if self.args['custom_tile_val']:
            self.tile_values = {
                'target': self.args['custom_target'],
                'dirt':self.args['custom_dirt'],
                'grass':self.args['custom_grass'], 
            }
        else:
            self.tile_values = {
                'target':10.,
                'dirt':0.2,
                'grass':-0.8, 
            }
        self.tile_values['self'] = -1.
        self.unkwown_avoidance = self.args['unknown_avoidance']


        
    def register_current_mental_representation(self, attn_map, w_self):
        self.x_attn = attn_map # assume it is normalized to [0,1]

        self.w_self = w_self # binary
        w_unknown = 1.
        for tile_name in self.TILES:    
            # self.w_target = self.get_ABA('target')
            aba = self.get_ABA(tile_name)
            w_unknown = w_unknown - aba 
            setattr(self, 'w_%s'%(str(tile_name)), aba)
        self.w_unknown = (w_unknown>self.recognition_threshold).astype(float) 

    def get_ABA(self, TILE_NAME):
        # ABA is approximate binary array
        this_tile = np.expand_dims(self.TILES[TILE_NAME], axis=1)
        this_tile = np.expand_dims(this_tile,axis=2)
        out = (self.x_attn - this_tile)**2
        out = np.sum(out,axis=0)/3.
        
        aba = selective_activation(out, epsilon=1e-4)
        return aba

    def compute_v1(self):
        v = {}
        v['self'] = self.tile_values['self'] * self.apply_deconvseq(self.w_self , self.tile_modules['self'])
        v['target'] = self.tile_values['target'] * self.apply_deconvseq(self.w_target , self.tile_modules['target'])

        # for every other tile that is not the target.
        # we want to compute the values of the tiles w.r.t target and position.
        # The main reason is because we want the gradient to still flow towards the target.
        for tile_name in self.TILES:   
            if tile_name == 'target':
                continue 
            w = getattr(self, 'w_%s'%(str(tile_name)))
            v[tile_name] = self.tile_values[tile_name] * self.apply_deconvseq( (self.w_target-self.w_self)*w , self.tile_modules[tile_name])   

        return v

    def compute_v_sigma(self, v):
        v_sigma = 0.
        for tile_name, vx in v.items():
            v_sigma = v_sigma + vx 

        v_max = torch.max(torch.abs(v_sigma.clone().detach()))
        w_unknown = numpy_to_pytorch_tensor(self.w_unknown).to(device=device)
        v_sigma = v_sigma + (v_max * numpy_to_pytorch_tensor(self.w_target).to(device=device) \
            - v_max * numpy_to_pytorch_tensor(self.w_self).to(device=device))

        v_max = torch.max(torch.abs(v_sigma.clone().detach()))  # actually not necessary
        v_sigma = v_sigma*(1-w_unknown) + w_unknown * -self.unkwown_avoidance*v_max

        return v_sigma

    def get_top_two_choices(self, v_sigma, idy, idx):
        action_values = {}
        for action in self.ACTIONS:
            out_of_bound = self.check_action_against_vision_bounds(idx, idy, action)
            if out_of_bound:
                action_values[action] = torch.tensor(-np.inf).to(device=device)
            else:
                idx_next, idy_next = self.get_next_pos_by_action(action, idx, idy)
                action_values[action] = v_sigma[0,0,idy_next,idx_next]
        # self.peek_self_attention(marker_coords='auto')

        sorted_action_values =  sort_dictionary_by_values(action_values, reverse=True)
        choices = {}
        for i,(action, value) in enumerate(sorted_action_values.items()):
            # if torch.isinf(value):
            #     raise RuntimeError('neg inf value')
            choices[i] = [action, value]
            if i>=1: break
        # print(choices)
        return choices

    def get_next_pos_by_action(self, action, idx, idy):
        if action == 'RIGHT':
            idy_next, idx_next = idy,idx+1
        elif action == 'DOWN':
            idy_next, idx_next = idy+1,idx 
        elif action == 'LEFT':
            idy_next, idx_next = idy,idx-1            
        elif action == 'UP':
            idy_next, idx_next = idy-1,idx
        else:
            raise RuntimeError('yess???')
        return idx_next, idy_next

    def make_a_plan(self):
        idys, idxs = self.get_mental_position_indices()
        idy, idx = idys[0], idxs[0]
        assert(len(idxs)==1) # robot only exists at one place

        plan, v_plan = ['ready'], 0.
        positions, tiles = [(idx,idy)], ['start']

        v = self.compute_v1()
        v_sigma = self.compute_v_sigma(v)
        
        for k in range(self.args['iter_limit']):
            # print(v_sigma)
            reached = self.reached_target(self.x_attn[:, idy, idx])
            if reached:
                k = k - 1
                break

            choices = self.get_top_two_choices(v_sigma, idy, idx)
            coin = np.random.choice([0,1],p=[0.9,0.1])
            action, value = choices[coin] # like ['LEFT', tensor(-0.8541, grad_fn=<SelectBackward>)]
            v_plan = v_plan + value

            # update for next iteration
            v_sigma, idx, idy = self.step_update(v_sigma, idx, idy, action)
            tile = self.tile_recognition(self.x_attn[:,idy,idx])

            tiles.append(tile)
            plan.append(action)            
            positions.append((idx,idy))
        k = k + 1
        v_plan = v_plan/k
        return plan, v_plan, positions, tiles, reached

    def tile_recognition(self, x):
        # x is array of shape  (3,)
        tile = 'unrecognized'

        for tile_name in self.TILES:
            out = (x-self.TILES[tile_name])**2
            out = np.sum(out,axis=0)/3.

            activated = selective_activation(out, epsilon=1e-4) > 1- self.recognition_threshold
            if activated:
                tile = tile_name
                break

        return tile

    def step_update(self, v_sigma, idx, idy, action):
        v_abs_max = torch.max(torch.abs(v_sigma.clone().detach())).item()
        v_sigma[0,0,idy,idx] = -0.9*v_abs_max # the max is reserved for negative 

        idx, idy = self.get_next_pos_by_action(action, idx, idy)
        return v_sigma, idx, idy

    def make_plans(self, ):
        plans ={}
        current_best_index = -1
        running_max = -999.

        for i in range(self.args['n_plans']):
            plan, v_plan, positions, tiles, reached = self.make_a_plan() 
            if v_plan.item()>running_max:
                current_best_index = i
            v_plan = self.normalization(v_plan) # always 1, for a novel way of loss computation
            plans[i] = [plan, v_plan, positions, tiles, reached]
        return plans, current_best_index

    def normalization(self, v_plan):
        abs_val = torch.abs(v_plan.clone().detach()).item()
        v_plan = v_plan/abs_val
        return v_plan

    ###############################
    # utils
    ###############################

    def reached_target(self, x):
        # x is array of shape  (3,)
        out = (x-self.TILES['target'])**2
        out = np.sum(out,axis=0)/3.

        reached = selective_activation(out, epsilon=1e-4)
        return reached >1- self.recognition_threshold
    
    def apply_deconvseq(self, aba, DCS_module):
        aba = numpy_to_pytorch_tensor(aba).to(device=device)
        w = DCS_module(aba)
        return w

    def check_action_against_vision_bounds(self,idx,idy, action):
        Hv, Wv = self.args['map_size']
        return np.any([
            (action == 'RIGHT') and (idx + 1>= Wv),
            (action == 'DOWN') and (idy + 1 >= Hv),
            (action == 'LEFT') and (idx - 1 < 0),
            (action == 'UP') and (idy-1 < 0),])

    ###############################
    # temp
    ###############################

    def get_mental_position_indices(self):
        current_pos = self.w_self
        idys, idxs = np.where(current_pos==np.max(current_pos))
        return idys, idxs    

    def peek_self_attention(self, marker_coords=None, new_attn=None, do_exit=False):
        # mainly for debug
        # put marker_coords as (idx,idy)
        
        if marker_coords =='auto':
            idy, idx = self.get_mental_position_indices()        
            marker_coords = idx, idy

        import matplotlib.pyplot as plt
        plt.figure()
        plt.gcf().add_subplot(121)
        plt.gca().imshow(self.x_attn.transpose(1,2,0))
        plt.gca().set_xlabel('x_attention')

        if marker_coords is not None:
            idx,idy= marker_coords
            print('(x,y):',idx,idy)
            plt.gca().scatter([idx],[idy], marker='x',c='b')

        plt.show()
        if do_exit:
            exit()        

class DeconvSeq(nn.Module):
    def __init__(self, n, kernel_size=3, mid_val=1., side_val=0.01):
        super(DeconvSeq, self).__init__()
        
        assert(kernel_size%2==1) # for now only odd kernel size

        self.n = n

        self.kernel = nn.ModuleList()
        for i in range(n):
            deconv = self.init_deconv(kernel_size, mid_val, side_val)
            self.kernel.append(deconv)

        self.tanh = nn.Tanh()

    def init_deconv(self, kernel_size, mid_val, side_val):
        mid = int((kernel_size-1)/2)
        padding = mid # yep

        deconv = nn.ConvTranspose2d(1,1,kernel_size,padding=padding,bias=False)
        deconv.weight.data = deconv.weight.data*0 + side_val
        deconv.weight.data[0,0,mid,mid] = mid_val
        return deconv

    def forward(self, x):
        for i in range(self.n):
            x = self.kernel[i](x)
            x = self.tanh(x)
            max_val = torch.max(torch.abs(x.clone().detach())).item()
            x = x/max_val
        return x

 
def get_position_indices_from_singleton_binary_array(pos_array):
    """
    assuming pos_array is binary, with only 1 entry of 1.
    x = np.zeros(shape=(5,7))
    idx,idy = np.random.choice(range(7),), np.random.choice(range(5),)
    print(idx,idy)
    x[idy,idx] = 1
    print(x)
    idy2,idx2 = get_position_indices_from_singleton_binary_array(x)
    print(idx2,idy2)

    """
    assert(np.sum(pos_array.reshape(-1))==1.)
    idy, idx = np.where(pos_array==np.max(pos_array))
    idy, idx = idy[0], idx[0]
    return idy, idx    

