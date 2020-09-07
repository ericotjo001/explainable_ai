from pipeline.objgen.random_simple_generator import SimpleRandomFetcher
import numpy as np

class TenClassesRandomFetcher(SimpleRandomFetcher):
    def __init__(self):
        super(TenClassesRandomFetcher, self).__init__()

    # def setup0001(self, general_meta_setting, explanation_setting, s=None):
    #     self.s = s if s is not None else (512,512) # image shape

    def uniform_random_draw(self):
        y0 = np.random.randint(10)
        bg_rand = np.random.randint(3)
        return self.draw_one_sample(y0, bg_rand)


    def draw_one_sample(self, y0, bg_rand):
        # y0 is 0,1,...,9 
        # bg_rand is 0, 1, 2
        if y0 == 0:
            cobj, cimg, heatmap, variables = self.get_random_CCellX()
        elif y0 == 1:
            cobj, cimg, heatmap, variables = self.get_random_CCellMX()
        elif y0 == 2:
            cobj, cimg, heatmap, variables = self.get_random_CCellPX()
        elif y0 == 3:
            cobj, cimg, heatmap, variables = self.get_random_RCellX() # red
        elif y0 == 4:
            cobj, cimg, heatmap, variables = self.get_random_RCellXB() # green
        elif y0 == 5:
            cobj, cimg, heatmap, variables = self.get_random_RCellXC() # blue
        elif y0 == 6:
            tfraction = 0.
            while tfraction < 0.3:
                cobj, cimg, heatmap, variables = self.get_random_CCellTX()
                tailpos = cobj.parts['tailpos'].reshape(-1)
                tfraction = np.round(np.sum(tailpos)/len(tailpos)*100.,2)
            #     if tfraction < 0.3: print('tfraction (reject)',tfraction)
            # print('tfraction',tfraction)
        elif y0 == 7:
            cobj, cimg, heatmap, variables = self.get_random_CCellTX3()
        elif y0 == 8:
            cobj, cimg, heatmap, variables = self.get_random_CCellTX8()
        elif y0 == 9:
            cobj, cimg, heatmap, variables = None, np.zeros(self.s + (3,)), np.zeros(self.s[:2]), {'type': 'NOISE'}
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
from utils.utils import FastPickleClient
class TenClassesPyIO(data.Dataset, TenClassesRandomFetcher, FastPickleClient):
    def __init__(self,):
        super(TenClassesPyIO, self).__init__()
        self.x, self.y = [], []

        # FastPickleClient
        self.save_text = 'Saving pytorch dataset via TenClassesPyIO(FastPickleClient)...'
        self.load_text = 'Loading pytorch dataset via TenClassesPyIO(FastPickleClient)...'

    def __getitem__(self, index):
        return np.array(self.x[index]), np.array(self.y[index])

    def __len__(self):
        return self.data_size

    def setup_training_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        self.x, self.y = [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=None)
        for i in range(data_size):
            if realtime_update:
                update_text = 'TenClassesPyIO.setup_training_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            cobj, cimg, heatmap, variables = self.uniform_random_draw()
            self.x.append(cimg.transpose(2,0,1))
            self.y.append(variables['y0'])
        print('%-64s'%('  data prepared.'))

    def setup_xai_evaluation_0001(self, general_meta_setting=None, explanation_setting=None, data_size=12, 
        realtime_update=False):
        self.x, self.y, self.h, self.v = [], [], [], []
        self.data_size = data_size

        self.setup0001(general_meta_setting, explanation_setting, s=None)
        for i in range(data_size):
            if realtime_update:
                update_text = 'TenClassesPyIO.setup_xai_evaluation_0001() progress %s/%s'%(str(i+1),str(data_size))
                print('%-64s'%(update_text),end='\r')
            cobj, cimg, heatmap, variables = self.uniform_random_draw()
            self.x.append(cimg.transpose(2,0,1))
            self.y.append(variables['y0'])
            self.h.append(heatmap)
            self.v.append(variables)
        print('%-64s'%('  data prepared.'))