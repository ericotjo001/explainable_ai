import numpy as np

class Interpolator(object):
    """
    Abstract interpolator class for KABEDONN
    Need implementations downstream
    """
    def __init__(self):
        super(Interpolator, self).__init__()
        self.interp_buffer = {}
        self.interp_package = None
        
    def init_interpolator(self, settings):
        if settings is None:
            self.mode = 'max_activation'
        else:
            raise NotImplementedError()
        
    def interpolator_signal_collection(self, interp_buffer):
        # interp_buffer = {'layer': layer_,'act': act,'act_pre': act_pre,}
        self.interp_buffer = interp_buffer
        
    def interpolator_processing(self):
        if self.mode == 'max_activation':
            self.interpolate_by_max_activation()
        else:
            raise NotImplementedError()

    def interpolator_output(self, x):
        if self.mode == 'max_activation':
            return self.predict_by_max_activation(x)
        else:
            raise NotImplementedError()

    ###############################
    # first type of interpolator
    ###############################
    def predict_by_max_activation(self, x):
        layer = self.interp_package['layer']
        act_idx = self.interp_package['max_act_idx']
        
        activated_node = self.layers[layer].node_list[act_idx]
        
        act_pre = self.activate_layer_l(x, layer-1, filter_mode=False)
        y_pred, NODE_INFO = activated_node.forward(act_pre)

        OUTPUT_INFO = {
            'output_mode': 'interpolation',
            'layer':layer,
            'activated_node': activated_node,
            'act_idx': act_idx,
            'activated_node':activated_node, 
            'NODE_INFO': NODE_INFO}
        return y_pred, OUTPUT_INFO

    def interpolate_by_max_activation(self):
        act = self.interp_buffer['act']
        max_act, max_act_idx = np.max(act), np.argmax(act)

        update_package = False
        if self.interp_package is None:
            update_package = True
        elif max_act > self.interp_package['max_act']:
            update_package = True

        if update_package:
            self.interp_package = {
                'max_act': max_act,
                'max_act_idx': max_act_idx,
                'layer':self.interp_buffer['layer'],
            }

            