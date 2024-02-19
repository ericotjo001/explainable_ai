
LIMIT_FACTOR = 48
BALANCE_ERROR ="""Length of altset exceeds %s* kwidth.
This probably means too many new samples are activating existing nodes 
i.e. admission threshold too low? Try adjusting the parameters.
e.g increase kwidth, admission_threshold and activation_threshold.
"""%(str(LIMIT_FACTOR))

class MetaTrainer(object):
    def __init__(self, ):
        super(MetaTrainer, self).__init__()
        
        self.metaconfig = {
            'balance_type': 'standard',
        }

    def check_altset_balance(self, altset):
        if self.metaconfig['balance_type'] == 'standard':
            if len(altset)> LIMIT_FACTOR *self.ix.kwidth:
                print('\n\nLENGTH OF ALTSET:',len(altset))
                raise RuntimeError(BALANCE_ERROR)
        else:
            raise NotImplementedError()

    def set_metaconfig(self, metaconfig):
        self.metaconfig = metaconfig