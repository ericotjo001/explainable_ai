ccell_meta_color_border = ('array',('uniform',0.8-0.3,0.8+0.1),
                        ('uniform',0.8-0.3,0.8+0.1),
                        ('uniform',0.8-0.3,0.8+0.1))
ccell_meta_color_inner = ('array',('uniform',0.2-0.1,0.2+0.1),
                        ('uniform',0.2-0.1,0.2+0.1),
                        ('uniform',0.4-0.1,0.4+0.1))
ccell_meta_radius = ('uniform',100.-10.,100.+10.)
ccell_meta_thickness = ('uniform',20-10,20+10)
ccell_meta_vertical_stretch = ('uniform',1-0.5,1+0.5)

rcell_meta_dim = ('uniform',100.,200.)

rotate_angle_range = [-60,60]
d_noise = ('uniform',6, 12)
noise_roughness = ('uniform',32.-1,64)

CCellX_metasetting = {
    'color_meta_setting':{
        'border': ccell_meta_color_border, 
        'inner': ccell_meta_color_inner,
    },
    'shape_meta_setting':{
        'radius': ccell_meta_radius,
        'vertical_stretch': ccell_meta_vertical_stretch, # to roughly make ellipse
        'thickness': ccell_meta_thickness,
        'd_noise': d_noise,
        'noise_roughness': noise_roughness, # 1 is very fine, pixel level noise                     
    },
}

CCellMX_metasetting = {
        'color_meta_setting':{
            'border': ccell_meta_color_border, 
            'inner': ccell_meta_color_inner,
            'bar': ('manual',)
        },
        'shape_meta_setting':{
            'radius': ccell_meta_radius,
            'vertical_stretch': ccell_meta_vertical_stretch, # to roughly make ellipse
            'thickness': ccell_meta_thickness,
            'bar_thickness': ccell_meta_thickness,
            'd_noise': d_noise,
            'noise_roughness':  noise_roughness,                 
        }
    }

CCellPX_metasetting = {
    'color_meta_setting':{
        'border': ccell_meta_color_border, 
        'inner': ccell_meta_color_inner,
        'bar': ccell_meta_color_border
    },
    'shape_meta_setting':{
        'radius': ccell_meta_radius,
        'vertical_stretch': ccell_meta_vertical_stretch, # to roughly make ellipse
        'thickness': ccell_meta_thickness,
        'bar_thickness':ccell_meta_thickness,
        'pole_thickness':('uniform',20-15,20+20),
        'd_noise':d_noise,
        'noise_roughness':  noise_roughness,
    }
}

RCellX_metasetting = {
    'color_meta_setting' : {
        'border': ('array',('uniform',0.8-0.2,0.8+0.1),
                    ('uniform',0.,0.2),
                    ('uniform',0.,0.2)),
        'inner': ccell_meta_color_inner
    } ,
    'shape_meta_setting': {
        'h':rcell_meta_dim,
        'w':rcell_meta_dim,
        'thickness': ccell_meta_thickness, # border thickness
        'd_noise':d_noise,
        'noise_roughness': noise_roughness # 1 is very fine, pixel level noise                    
    }                  
}

CCellTX_metasetting = {
    'color_meta_setting' : {
        'border': ccell_meta_color_border, 
        'inner': ccell_meta_color_inner,
        'tail': ccell_meta_color_border
    },
    'shape_meta_setting' : {
        'radius': ccell_meta_radius,
        'vertical_stretch': ccell_meta_vertical_stretch, # to roughly make ellipse
        'thickness': ccell_meta_thickness,
        'tail_thickness':ccell_meta_thickness,
        'tail_angles': ('constant', None),
        'tail_ratio_to_radius': ('constant',1.5),
        'd_noise':d_noise,
        'noise_roughness': noise_roughness,     
    }
   
}
