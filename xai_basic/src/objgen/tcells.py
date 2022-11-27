from .cells import CCell
import numpy as np
from skimage.transform import resize, rotate
from scipy.ndimage import shift

class CCellT(CCell):
    """CCell with a tail"""
    def __init__(self):
        super(CCellT, self).__init__()

    def setup_tcell(self, color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8]),
                'tail': np.array([0.8,0.8,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'radius': 100,
                'vertical_stretch':1.0, # to roughly make ellipse
                'thickness': 20,
                'tail_thickness':20,
                'tail_ratio_to_radius':2.,
                'd_noise':5.,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }           
        else:
            self.shape_setting = shape_setting

    def make_tcell(self, centerpos=(0,0), rotate_angle=None):
        r = self.shape_setting['radius']
        d_noise = self.shape_setting['d_noise']
        thickness = self.shape_setting['thickness']
        noise_roughness = self.shape_setting['noise_roughness']
        vertical_stretch = self.shape_setting['vertical_stretch']
        border_color = self.color_setting['border']
        inner_color = self.color_setting['inner'] 
        
        tail_angles = self.shape_setting['tail_angles'] if 'tail_angles' in self.shape_setting else None
        tail_thickness = self.shape_setting['tail_thickness']
        tail_color = self.color_setting['tail']
        tail_ratio_to_radius= self.shape_setting['tail_ratio_to_radius']

        return self.build_tcell_body(r, thickness, tail_thickness, border_color, tail_color, inner_color, 
            vertical_stretch=vertical_stretch, tail_angles=tail_angles, tail_ratio_to_radius=tail_ratio_to_radius,
            d_noise=d_noise, noise_roughness=noise_roughness, centerpos=centerpos, rotate_angle=rotate_angle)  

    def build_tcell_body(self,r, thickness, tail_thickness, border_color, tail_color, inner_color, 
        vertical_stretch=1.0, tail_angles=None, tail_ratio_to_radius=2,
        d_noise=None, noise_roughness=None, centerpos=(0,0), rotate_angle=None):
        
        self.build_basic_ball_body(r, thickness, border_color, inner_color, vertical_stretch=vertical_stretch,
            d_noise=d_noise, noise_roughness=noise_roughness, centerpos=(0,0), rotate_angle=None, 
            do_make_explanation=False)
        
        x,y,d = self.parts['xyd']
        x0,y0 = self.parts['centerpos']
        ball= self.parts['ball']
        r = self.shape_setting['radius'] 
        n = self.parts['noise'] 
        
        if d_noise is not None:
            n = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            x = x + n
            n2 = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            y = y + n2
            
        tail, tailpos = self.create_elongation(x,y,x0,y0, r, tail_thickness, tail_color, border_thickness=thickness,
            tail_ratio_to_radius=tail_ratio_to_radius,tail_angles=tail_angles, vertical_stretch=vertical_stretch)
        ball = ball*(1-tailpos) + tail

        ############### AFFINE TRANSFORMATION ###############
        # rotate
        if rotate_angle is not None:
            self.parts['inner_ball'] = rotate(self.parts['inner_ball'], rotate_angle)
            self.parts['border'] = rotate(self.parts['border'] , rotate_angle)
            tail = rotate(tail, rotate_angle)
            ball = rotate(ball, rotate_angle)
            tailpos = rotate(tailpos, rotate_angle)

        # translate
        self.parts['inner_ball'] = shift(self.parts['inner_ball'], centerpos+(0,))
        self.parts['border'] = shift(self.parts['border'], centerpos+(0,))
        tail = shift(tail, centerpos+(0,)) # the last dim for channel
        ball = shift(ball, centerpos+(0,))
        tailpos = shift( tailpos, centerpos+(0,))

        ball = np.clip(ball, a_min=0., a_max=1.)

        ############### EXPLANATION ZONE ###################
        self.parts['tailpos'] = tailpos
        self.parts['tail'] = np.clip(tail, a_min=0., a_max=1.) 
        self.parts['inner_ball'] = np.clip(self.parts['inner_ball'], a_min=0., a_max=1.)   
        self.parts['border'] = np.clip(self.parts['border'], a_min=0., a_max=1.)
        self.parts['ball'] = ball
        heatmap = self.make_explanation()
        
        return ball, heatmap

    def create_elongation(self,x,y, x0,y0, radius, tail_thickness, tail_color, border_thickness,
        tail_ratio_to_radius=2., tail_angles=None, vertical_stretch=1.):
        tail = (x-x0<=tail_thickness/2)*(x-x0>=-tail_thickness/2)*(y-y0 >= 0.8*vertical_stretch*radius)*(y-y0 <= tail_ratio_to_radius*vertical_stretch*radius + border_thickness)
        tailpos = self.stack_color(tail>0., [1.,1.,1.]) # make binary index for bar in 3 channel space
        tailpos = tailpos.transpose(1,2,0)
        tail = self.stack_color(tail, tail_color)
        tail = tail.transpose(1,2,0)
        return tail, tailpos


class CCellTs(CCellT):
    """CCellT but with many tails"""
    def __init__(self):
        super(CCellTs, self).__init__()

    def setup_tcell(self, color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8]),
                'tail': np.array([0.8,0.8,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'radius': 100,
                'vertical_stretch':1.0, # to roughly make ellipse
                'thickness': 20,
                'tail_thickness':20,
                'tail_angles':[0.,90.],
                'tail_ratio_to_radius':2.,
                'd_noise':5.,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }           
        else:
            self.shape_setting = shape_setting

    def create_elongation(self,x,y, x0,y0, radius, tail_thickness, tail_color, border_thickness, 
        tail_ratio_to_radius=1.5, tail_angles=[0.,180.], vertical_stretch=1.):        
        tail = None
        for theta in tail_angles:
            theta = theta*np.pi/180.
            x1 = np.cos(theta) * (x-x0) - np.sin(theta) * (y-y0)
            y1 = np.sin(theta) * (x-x0) + np.cos(theta) * (y-y0)
            t = (x1<=tail_thickness/2)*(x1>=-tail_thickness/2)*(y1 >= 0.8*vertical_stretch*radius)*(y1<= tail_ratio_to_radius * radius * vertical_stretch + border_thickness)
            if tail is None:
                tail = t
            else:
                tail = tail + t

        tailpos = self.stack_color(tail>0., [1.,1.,1.]) # make binary index for bar in 3 channel space
        tailpos = tailpos.transpose(1,2,0)
        tail = self.stack_color(tail, tail_color)
        tail = tail.transpose(1,2,0)
        return tail, tailpos