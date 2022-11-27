from .cells import CCell
import numpy as np
from skimage.transform import resize, rotate
from scipy.ndimage import shift

class CCellM(CCell):
    def __init__(self):
        super(CCellM, self).__init__()

    def setup_ccell(self, color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8]),
                'bar': np.array([0.8,0.8,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'radius': 100,
                'vertical_stretch':1.0, # to roughly make ellipse
                'thickness': 20,
                'bar_thickness':20,
                'd_noise':5.,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }           
        else:
            self.shape_setting = shape_setting
    
    def make_ccell(self, centerpos=(0,0), rotate_angle=None):
        r = self.shape_setting['radius']
        d_noise = self.shape_setting['d_noise']
        thickness = self.shape_setting['thickness']
        noise_roughness = self.shape_setting['noise_roughness']
        vertical_stretch = self.shape_setting['vertical_stretch']
        border_color = self.color_setting['border']
        inner_color = self.color_setting['inner'] 
        
        bar_thickness = self.shape_setting['bar_thickness']
        bar_color = self.color_setting['bar']

        return self.build_ccell_body(r, thickness,bar_thickness, border_color, bar_color, inner_color, vertical_stretch=vertical_stretch,
            d_noise=d_noise, noise_roughness=noise_roughness, centerpos=centerpos, rotate_angle=rotate_angle)    
    
    def build_ccell_body(self,r, thickness,bar_thickness, border_color, bar_color, inner_color, vertical_stretch=1.0,
        d_noise=None, noise_roughness=None, centerpos=(0,0), rotate_angle=None):
        
        self.build_basic_ball_body(r, thickness, border_color, inner_color, vertical_stretch=vertical_stretch,
            d_noise=d_noise, noise_roughness=noise_roughness, centerpos=(0,0), do_make_explanation=False)
        
        x,y,d = self.parts['xyd']
        x0,y0 = self.parts['centerpos']
        ball= self.parts['ball']
        r = self.shape_setting['radius'] 
        pole_thickness = self.shape_setting['pole_thickness'] if 'pole_thickness' in self.shape_setting else None
        n = self.parts['noise']
        
        if d_noise is not None:
            n = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            x = x + n
            n2 = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            y = y + n2
            

        bar, barpos = self.create_skeleton(x,y, x0,y0,r, bar_thickness,bar_color, 
            vertical_stretch=vertical_stretch, pole_thickness=pole_thickness)
        ball = ball*(1-barpos) + bar

        ############### AFFINE TRANSFORMATION ###############
        # rotate
        if rotate_angle is not None:
            self.parts['inner_ball'] = rotate(self.parts['inner_ball'], rotate_angle)
            self.parts['border'] = rotate(self.parts['border'] , rotate_angle)
            bar = rotate(bar, rotate_angle)
            ball = rotate(ball, rotate_angle)
        # translate
        self.parts['inner_ball'] = shift(self.parts['inner_ball'], centerpos + (0,))
        self.parts['border'] = shift(self.parts['border'] , centerpos + (0,))
        bar = shift(bar, centerpos + (0,))
        ball = shift(ball, centerpos + (0,))

        ball = np.clip(ball, a_min=0., a_max=1.)

        ############### EXPLANATION ZONE ###################
        self.parts['ball'] = ball
        self.parts['bar'] = np.clip(bar, a_min=0., a_max=1.) 
        self.parts['inner_ball'] = np.clip(self.parts['inner_ball'], a_min=0., a_max=1.)   
        self.parts['border'] = np.clip(self.parts['border'], a_min=0., a_max=1.)
        heatmap = self.make_explanation()
        
        return ball, heatmap

    def create_skeleton(self,x,y, x0,y0,r,bar_thickness,bar_color, vertical_stretch=None, pole_thickness=None):
        bar = (x - x0<=r)*(x-x0>=-r)*(y-y0<=int(bar_thickness/2))*(y-y0>=-int(bar_thickness/2))
        barpos = self.stack_color(bar>0., [1.,1.,1.]) # make binary index for bar in 3 channel space
        barpos = barpos.transpose(1,2,0)
        bar = self.stack_color(bar, bar_color)
        bar = bar.transpose(1,2,0)
        return bar, barpos

class CCellP(CCellM):
    def __init__(self):
        super(CCellP, self).__init__()
        """
        bar is the horizontal wall dividing the cell when it is upright.
        pole is the vertical wall.
        For convenience, now both pole and bar are collectively also called bar.
        """

    def setup_ccell(self, color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8]),
                'bar': np.array([0.8,0.8,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'radius': 100,
                'vertical_stretch':1.0, # to roughly make ellipse
                'thickness': 20,
                'bar_thickness':20,
                'pole_thickness':10,
                'd_noise':5.,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }           
        else:
            self.shape_setting = shape_setting

    def create_skeleton(self,x,y, x0,y0,r,bar_thickness, bar_color, vertical_stretch=1.0, pole_thickness=None):
        if pole_thickness is None:
            pole_thickness = bar_thickness
        bar = (x - x0<=r)*(x-x0>=-r)*(y-y0<=int(bar_thickness/2))*(y-y0>=-int(bar_thickness/2))
        
        pole = (y - y0<= r*vertical_stretch)*(y-y0>=-r* vertical_stretch)*(x-x0<=int(pole_thickness/2))*(x-x0>=-int(pole_thickness/2))
        polepos = (pole>0.)*(1 - bar>0)
        bar = bar + pole*polepos

        barpos = self.stack_color(bar>0., [1.,1.,1.]) # make binary index for bar in 3 channel space
        barpos = barpos.transpose(1,2,0)
        bar = self.stack_color(bar, bar_color)
        bar = bar.transpose(1,2,0)
        return bar, barpos