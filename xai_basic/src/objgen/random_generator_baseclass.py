import numpy as np
from .cellsX import CCellX, CCellMX, CCellPX, RCellX

class SettingBuilder(object):
    def __init__(self, ):
        super(SettingBuilder, self).__init__()
    
    def generate_setting(self, meta_setting):
        """
        ('uniform',min, max)
        ('normal', mean, sd)
        """
        setting = {}
        if meta_setting is None:
            # default example
            meta_setting = {
                'attr1': ('uniform',0.,1.),
                'attr2': ('normal',1.,0.2),
                'attr3': ('constant',[0.1,1.2]),
                'attr4': ('array',('uniform',0.7,0.8),('normal',0.8,0.1),('uniform',0.2,0.1))
            }

        for attr, attr_sttg in meta_setting.items():
            attr_random_mode = attr_sttg[0]
            if attr_random_mode == 'uniform':
                lower_bound, upper_bound = attr_sttg[1], attr_sttg[2]
                setting[attr] = np.random.uniform(lower_bound, upper_bound)
            elif attr_random_mode == 'normal':
                mean, sd = attr_sttg[1], attr_sttg[2]
                setting[attr] = np.random.normal(mean,sd)
            elif attr_random_mode == 'array':
                setting[attr] = []
                for i in range(1,len(attr_sttg)):
                    attr_random_submode = attr_sttg[i][0]
                    if attr_random_submode == 'uniform':
                        lower_bound, upper_bound = attr_sttg[i][1],attr_sttg[i][2]
                        setting[attr].append(np.random.uniform(lower_bound, upper_bound))
                    elif attr_random_submode == 'normal':
                        mean, sd = attr_sttg[i][1], attr_sttg[i][2]
                        setting[attr].append(np.random.normal(mean,sd))
                    else:
                        raise Exception('attr_random_submode is not valid')     
                setting[attr] = np.array(setting[attr])
            elif attr_random_mode == 'constant':
                setting[attr] = attr_sttg[1]
            elif attr_random_mode == 'manual':
                setting[attr] = None
            else:
                raise Exception('attr_random_mode is not valid')
        return setting  

    def print_setting(self, setting, header=None):
        if header: strg = ''
        vals = ''
        for x, y in setting.items():
            if header: strg = strg + '%-12s'%(str(x))
            if isinstance(y,float): y = round(y,3)
            if isinstance(y, np.ndarray): y = np.round(y,3)
            vals = vals + '%-12s'%(str(y))
        if header: print(strg)
        print(vals)

class RandomFetcher(object):
    def __init__(self):
        super(RandomFetcher, self).__init__()
        self.sb = SettingBuilder()

    def get_random_center_position(self, mode='quarter_of_max'):
        if mode == 'quarter_of_max':
            x0 = np.random.uniform( -0.25*self.s[0], 0.25*self.s[0])
            y0 = np.random.uniform( -0.25*self.s[1], 0.25*self.s[1])  
        elif mode == 'one_third_of_max':
            x0 = np.random.uniform( -0.33*self.s[0], 0.33*self.s[0])
            y0 = np.random.uniform( -0.33*self.s[1], 0.33*self.s[1])           
        else:
            raise RuntimeError('Invalid random center mode.')
        return int(x0), int(y0)
    
    def get_random_rotation_angle(self,):
        tmin, tmax = self.general_meta_setting['rotation range']
        theta = np.random.uniform(tmin, tmax)
        return theta

    def get_simple_affine_transformation(self, center_mode='quarter_of_max'):
        x0,y0 = self.get_random_center_position(mode=center_mode)
        theta = self.get_random_rotation_angle()
        return x0,y0, theta
    
    def get_random_CCellX(self, meta_setting):
        c = CCellX()
        # Implement as the following
        # __generate_all_settings__
        # c.setup_explanation_attributes(explanation_setting)
        # c.setup_basic_ball(color_setting, shape_setting)
        # circ, explanation = c.make_basic_ball(centerpos=(100,0))      
        return c

    # implement other types as well, for example
    # def get_random_CCellMX(self, meta_setting): return None
    # def get_random_CCellPX(self, meta_setting): return None
    # def get_random_RCellX(self, meta_setting): return None