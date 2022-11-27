from .random_simple_generator import SimpleRandomFetcher
import numpy as np


#######################################################################
#                            - Example 2 -
#######################################################################

class ThreeClassesRandomFetcher(SimpleRandomFetcher):
    def __init__(self):
        super(ThreeClassesRandomFetcher, self).__init__()

    # def setup0001(self, general_meta_setting, explanation_setting, s=None):
    #     self.s = s if s is not None else (512,512) # image shape

    def uniform_random_draw(self):
        y0 = np.random.randint(3)
        bg_rand = np.random.randint(3)
        return self.draw_one_sample(y0, bg_rand)


    def draw_one_sample(self, y0, bg_rand):
        # y0 is 0,1,2 
        # bg_rand is 0, 1, 2
        if y0 == 0:
            cobj, cimg, heatmap, variables = self.get_random_CCellPX()
        elif y0 == 1:
            cobj, cimg, heatmap, variables = self.get_random_RCellXB() # green
        elif y0 == 2:
            cobj, cimg, heatmap, variables = self.get_random_CCellTX8()
        # print('[%s] cimg.shape:%s, heatmap.shape:%s'%(str(y0), str(cimg.shape),str(heatmap.shape)))
                
        self.background_setting['type'] = bg_rand
        bg = self.generate_background()
        if bg is not None:
            ep = 1e-2
            pos = np.stack(((cimg[:,:,0]<ep),(cimg[:,:,1]<ep),(cimg[:,:,2]<ep))).transpose(1,2,0)
            cimg =  cimg + pos * bg 
        cimg = np.clip(cimg, a_min=0., a_max=1.)

        variables['y0'] = y0
        variables['bg_rand'] = bg_rand

        return cobj, cimg, heatmap, variables


import torch.utils.data as data

class ThreeClassesPyIO(data.Dataset, ThreeClassesRandomFetcher):
    def __init__(self,):
        super(ThreeClassesPyIO, self).__init__()
        self.x, self.y = [], [] 

    def __getitem__(self, index):
        return np.array(self.x[index]), np.array(self.y[index])

    def __len__(self):
        return self.data_size

    def setup_training_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        self.x, self.y = [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=(256,256))
        for i in range(data_size):
            if realtime_update:
                update_text = 'ThreeClassesPyIO.setup_training_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            cobj, cimg, heatmap, variables = self.uniform_random_draw()
            self.x.append(cimg.transpose(2,0,1))
            self.y.append(variables['y0'])
        print('%-64s'%('  data prepared.'))

    def setup_xai_evaluation_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        # self.h takes the value of [0, 0.4, 0.9]
        self.x, self.y, self.h, self.v = [], [], [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=(256,256))
        for i in range(data_size):
            if realtime_update:
                update_text = 'ThreeClassesPyIO.setup_xai_evaluation_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            cobj, cimg, heatmap, variables = self.uniform_random_draw()
            self.x.append(cimg.transpose(2,0,1))
            self.y.append(variables['y0'])
            self.h.append(heatmap)
            self.v.append(variables)
        print('%-64s'%('  data prepared.'))


class ThreeClassesPyIOwithHeatmap(ThreeClassesPyIO):
    def __init__(self, x,y,h, label_mapping={0:0., 1:0.4, 2:0.9}):
        """
        label_mapping is used to convert the values in heatmaps to labels
        Our standard values are 0,0.4,0.9 and we will label them as 0,1,2 respectively
        """
        super(ThreeClassesPyIOwithHeatmap, self).__init__()
        self.x = x
        self.y = y
        self.h = h
        self.data_size = len(self.x)
        self.label_mapping = label_mapping

    def __getitem__(self, index):   
        hx = np.array(self.h[index])
        h0 = (hx==self.label_mapping[0])*0 + (hx==self.label_mapping[1])*1 + (hx==self.label_mapping[2])*2

        return np.array(self.x[index]), np.array(self.y[index]), h0  