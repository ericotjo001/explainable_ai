import numpy as np
from pipeline.objgen.baseclass import ObjetOutline2D
from pipeline.objgen.cellsX import CCellX, CCellMX, CCellPX, RCellX, CCellTX,CCellTsX
from pipeline.objgen.random_generator_baseclass import RandomFetcher
from pipeline.objgen.random_simple_generator_metasettings import *
from skimage.transform import resize

class SimpleRandomFetcher(RandomFetcher):
    def __init__(self):
        super(SimpleRandomFetcher, self).__init__()
        self.obj2d = ObjetOutline2D()
        
        # inherited
        # self.sb = SettingBuilder()

    def setup0001(self, general_meta_setting, explanation_setting, background_setting=None, s=None):
        self.s = s if s is not None else (512,512) # image shape
        self.c = (int(self.s[0]/2),int(self.s[1]/2)) # image center coord

        if general_meta_setting is None:
            # the following are metasettings for the arguments of 
            #   pipeline.objgen.random_generator_baseclass.SettingBuilder
            # see the metasettings initiated in pipeline.objgen.random_simple_generator_metasettings
            self.general_meta_setting = {
                'random position':'one_third_of_max',
                'rotation range':rotate_angle_range,
                'CCellX': CCellX_metasetting,
                'CCellMX': CCellMX_metasetting,
                'CCellPX': CCellPX_metasetting,
                'RCellX': RCellX_metasetting,
                'CCellTX': CCellTX_metasetting,
            }
        if explanation_setting is None:
            # this setting is peculiar to the implementation of explanations in objgen.cellsX
            self.explanation_setting = { 
                'localization_threshold' : 0.05,
                'localization_value' : 0.4,
                'discriminative_feature_threshold' : 0.05,
                'discriminative_feature_value' : 0.9
            }
        else:
            self.explanation_setting = explanation_setting

        if background_setting is None:
            self.background_setting = {
                'type': 0,
            }
        else:
            self.background_setting = background_setting

    def get_setting_from_meta_setting(self, name_type, alternate_type=None):
        shape_meta_setting = self.general_meta_setting[name_type]['shape_meta_setting'].copy()
        color_meta_setting = self.general_meta_setting[name_type]['color_meta_setting'].copy()

        if alternate_type == 'RCellXB':
            name_type = alternate_type
            color_meta_setting['border'] = ('array', ('uniform',0.,0.2),('uniform',0.8-0.2,0.8+0.1), ('uniform',0.,0.2))
        elif alternate_type == 'RCellXC':
            name_type = alternate_type
            color_meta_setting['border'] = ('array', ('uniform',0.,0.2), ('uniform',0.,0.2) ,('uniform',0.8-0.2,0.8+0.1))

        # recall self.sb = SettingBuilder()
        shape_setting = self.sb.generate_setting(meta_setting=shape_meta_setting)
        color_setting = self.sb.generate_setting(meta_setting=color_meta_setting)
        return shape_setting, color_setting

    def collect_setting_in_variables_dictionary(self, name_type, x0,y0,theta,shape_setting, color_setting):
        variables = {
            'type': name_type,
            'center': (x0,y0),
            'theta': theta,
            'shape_setting': shape_setting, 
            'color_setting': color_setting}
        return variables

    ################################################
    # different types of cells to fetch
    ################################################

    # 1.
    def get_random_CCellX(self):
        name_type = 'CCellX'
        shape_setting, color_setting = self.get_setting_from_meta_setting(name_type)
        
        x0, y0, theta = self.get_simple_affine_transformation(center_mode=self.general_meta_setting['random position'])
        cobj = CCellX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting)
        cimg, heatmap = cobj.make_basic_ball(centerpos=(x0,y0), rotate_angle=theta)     
        variables = self.collect_setting_in_variables_dictionary(name_type,x0,y0,theta,shape_setting, color_setting)
        return cobj, cimg, heatmap, variables

    # 2.
    def get_random_CCellMX(self):
        name_type = 'CCellMX'
        shape_setting, color_setting = self.get_setting_from_meta_setting(name_type)
        color_setting['bar'] = color_setting['border']

        x0, y0, theta = self.get_simple_affine_transformation(center_mode=self.general_meta_setting['random position'])
        cobj = CCellMX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting) 
        cimg, heatmap = cobj.make_ccell(centerpos=(x0,y0),rotate_angle=theta)       
        variables = self.collect_setting_in_variables_dictionary(name_type, x0,y0,theta,shape_setting, color_setting)
        return cobj, cimg, heatmap, variables

    # 3.
    def get_random_CCellPX(self):
        name_type = 'CCellPX'
        shape_setting, color_setting = self.get_setting_from_meta_setting(name_type)
        color_setting['bar'] = color_setting['border']
        
        x0, y0, theta = self.get_simple_affine_transformation(center_mode=self.general_meta_setting['random position'])
        cobj = CCellPX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting)
        cimg, heatmap = cobj.make_ccell(centerpos=(x0,y0),rotate_angle=theta)        
        variables = self.collect_setting_in_variables_dictionary(name_type, x0,y0,theta,shape_setting, color_setting)
        return cobj, cimg, heatmap, variables

    # 4.
    def get_random_RCellX(self, alternate_type=None):
        name_type = 'RCellX'
        shape_setting, color_setting = self.get_setting_from_meta_setting(name_type, alternate_type=alternate_type)

        x0, y0, theta = self.get_simple_affine_transformation(center_mode=self.general_meta_setting['random position'])
        cobj = RCellX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting)
        cimg, heatmap = cobj.make_basic_rect(centerpos=(x0,y0),rotate_angle=theta)
        variables = self.collect_setting_in_variables_dictionary(name_type, x0,y0,theta,shape_setting, color_setting)
        return cobj, cimg, heatmap, variables

    # 5.
    def get_random_RCellXB(self): 
        return self.get_random_RCellX(alternate_type='RCellXB')

    # 6.
    def get_random_RCellXC(self):
        return self.get_random_RCellX(alternate_type='RCellXC')

    # 7.
    def get_random_CCellTX(self, alternate_type=None):
        name_type = 'CCellTX'
        shape_setting, color_setting = self.get_setting_from_meta_setting(name_type, alternate_type=alternate_type)
        
        if alternate_type == 'CCellTX3':
            shape_setting['tail_angles'] = np.array([0., 120., 240.])
        elif alternate_type == 'CCellTX8':
            shape_setting['tail_angles'] = np.array([45.*i for i in range(8)])

        color_setting['tail'] =  np.clip(color_setting['border'] * 0.75,a_max=1.0,a_min=0.0)

        x0, y0, theta = self.get_simple_affine_transformation(center_mode=self.general_meta_setting['random position'])
        if alternate_type is None: 
            cobj = CCellTX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting)
        else:
            cobj = CCellTsX(fast_init=True, array_shape=self.s, color_setting=color_setting, shape_setting=shape_setting, explanation_setting=self.explanation_setting)
        cimg, heatmap = cobj.make_tcell(centerpos=(x0,y0),rotate_angle=theta)

        variables = self.collect_setting_in_variables_dictionary(name_type, x0,y0,theta,shape_setting, color_setting)
        return cobj, cimg, heatmap, variables

    # 8.
    def get_random_CCellTX3(self, alternate_type='CCellTX3'):
        return self.get_random_CCellTX(alternate_type=alternate_type)

    # 9.
    def get_random_CCellTX8(self, alternate_type='CCellTX8'):
        return self.get_random_CCellTX(alternate_type=alternate_type)


    ################################################
    # background generator
    ################################################

    def generate_background(self):
        x, y = self.obj2d.get_lattice_coord()
        s = x.shape
        x0 = np.random.randint(int(s[0]/4),int(s[0]*3/4))
        y0 = np.random.randint(int(s[1]/4),int(s[1]*3/4))
        d = ((x-x0)**2 + (y-y0)**2)**0.5
        bg_type = self.background_setting['type']

        if bg_type == 1:
            bg = self.background0001(x, y, d)
        elif bg_type == 2:
            bg = self.background0002(x, y, d)
        else:
            bg = None

        if bg is not None:
            bg = np.clip(bg, a_max=1.0, a_min=0.)

        # full_shape = self.s + (3,)
        return bg

    def background0001(self, x, y, d):
        bg = []
        for _ in range(3):
            d = d*0.
            n = resize(np.random.uniform(0.2,0.7, size=[int(dn/64) for dn in d.shape]), d.shape)
            bg.append(d + n)
        bg = np.stack(tuple(bg)).transpose(1,2,0)
        return bg

    def background0002(self, x, y, d):
        bg = []
        for _ in range(3):
            n = resize(np.random.uniform(-0.3,0.3, size=[int(dn/32) for dn in d.shape]), d.shape)           
            stretch = np.random.uniform(1e-4,1e-3)
            bgn = 0.5*np.sin(stretch*(d+n*d)*180/np.pi)+0.5
            bg.append(0.5*bgn)
        bg = np.stack(tuple(bg)).transpose(1,2,0)
        return bg

