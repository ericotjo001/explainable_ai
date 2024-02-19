
class EvaluationResult(object):
    def __init__(self, **kwargs):
        super(EvaluationResult, self).__init__()
        
        if kwargs['mode'] == 'simple_accuracy':
            self.setup_simple_accuracy(**kwargs)
        else:
            raise NotImplementedError()

    def setup_simple_accuracy(self, **kwargs):
        self.correct = kwargs['correct']
        self.total = kwargs['total']
        self.header_text = kwargs['header_text']
        if 'indices_wrong_data' in kwargs:
            self.indices_wrong_data = kwargs['indices_wrong_data']

    def print_accuracy(self):
        if self.header_text is not None:
            print(self.header_text)

        c,t = self.correct, self.total
        print('acc = %s/%s=%s'%(str(c),str(t),str(c/t)))


def evaluate_on_test_data(net, **eval_settings):
    print('evaluate_on_test_data...')
    
    kwidth = 10
    from src.model.indexer import DataIndexer
    ix = DataIndexer(eval_settings['DIRS']['TEST_DATA_DIR'], 
        eval_settings['folder_to_class_mapping'], 
        kwidth, 
        eval_settings['data_fetcher'], 
        init_new=True)
    q = ix.create_queue_one_runthrough()

    total, correct = 0, 0
    elasticset = []
    indices_wrong_data = [] # for wrongly predicted data
    while q.qsize()>0 or len(elasticset)>0:
        elasticset, q, kFetcherInfo = ix.mould_elasticset_by_queue(elasticset, q)
        x_batch, y0_batch = ix.fetch_data_by_elastic_set(elasticset)

        for i, (x,y0) in enumerate(zip(x_batch, y0_batch)):
            y, activated_node = net.forward(x)

            if int(y)==int(y0):
                correct+=1
            else:
                indices_wrong_data.append(elasticset[i])
            total+=1   
            if total%100==0:
                update_text = 'eval qsize: [%s]'%(str(q.qsize()))
                print('%-64ss'%(str(update_text)), end='\r') 
        elasticset = []

    evalargs = {
        'mode': 'simple_accuracy',
        'correct': correct, 'total': total, 
        'header_text': 'test accuracy:',
        'indices_wrong_data':indices_wrong_data,
    }         
    EVAL_RESULT = EvaluationResult(**evalargs)
    return EVAL_RESULT