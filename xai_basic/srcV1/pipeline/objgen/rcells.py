from pipeline.objgen.baseclass import ObjetOutline2D
import numpy as np
from skimage.transform import resize, rotate

class RCell(ObjetOutline2D):
    def __init__(self):
        super(RCell, self).__init__()
        
    def setup_rectangle(self,color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'h':200,
                'w':100,
                'thickness': 0, # border thickness
                'd_noise':5,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }
        else:
            self.shape_setting = shape_setting

    def make_basic_rect(self, centerpos=(0,0), rotate_angle=None):
        h = self.shape_setting['h']
        w = self.shape_setting['w']
        d_noise = self.shape_setting['d_noise']
        border_thickness = self.shape_setting['thickness']
        noise_roughness = self.shape_setting['noise_roughness']
        border_color = self.color_setting['border']
        inner_color = self.color_setting['inner']   
        
        return self.build_basic_rectangular_body(h, w, border_thickness, border_color, inner_color,
                                    d_noise=d_noise, noise_roughness=noise_roughness, centerpos=centerpos, 
                                    rotate_angle=rotate_angle)
            
    def build_basic_rectangular_body(self, h, w, border_thickness, border_color, inner_color,
                                    d_noise=None, noise_roughness=None, centerpos=(0,0), rotate_angle=None):
        x, y = self.get_lattice_coord()
        x0,y0 = centerpos
        d = ((x-x0) **2 + (y-y0)**2)**0.5
        
        if d_noise is not None:
            n = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            x = x + n
            n2 = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            y = y + n
            
        border = (x-x0<=int(w/2))*(x-x0>=-int(w/2))*(y-y0>=-int(h/2))*(y-y0<=int(h/2))
        inner_rect = (x-x0<=int(w/2)-border_thickness)*(x-x0>=-int(w/2)+border_thickness)*\
            (y-y0>=-int(h/2)+border_thickness)*(y-y0<=int(h/2)-border_thickness)
        border = np.clip(border.astype(np.float)-inner_rect.astype(np.float), a_max=1.0, a_min=0.)
        
        border = self.stack_color(border, border_color, channel_CHW=True)        
        inner_rect = self.stack_color(inner_rect, inner_color, channel_CHW=True)
        
        rect = border + inner_rect

        if rotate_angle is not None:
            rect = rotate(rect, rotate_angle)
            border = rotate(border, rotate_angle)
            inner_rect = rotate(inner_rect, rotate_angle)
        rect = np.clip(rect, a_min=0., a_max=1.)

        ########### EXPLANATION ZONE ###########
        self.parts = {
            'xyd' :(x,y,d),
            'centerpos':(x0,y0),
            'noise':n,
            'border': np.clip(border, a_min=0., a_max=1.),
            'inner_rect': np.clip(inner_rect, a_min=0., a_max=1.),
        }
        heatmap = self.make_explanation()

        return rect, heatmap