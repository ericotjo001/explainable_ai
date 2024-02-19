
def redirect_for_debugtests(dargs, **kwargs):
    print('redirect_for_debugtests...')

    if dargs['redir_id'] == 0:
        return
    elif dargs['redir_id']==1:
        test_first_nqueue(dargs,**kwargs)
    exit()


def test_first_nqueue(dargs, **kwargs):
    print('test_first_nqueue...')

    config = kwargs['fitting_config']    
    q = kwargs['net'].ix.create_queue_one_runthrough(config=config['qconfig'])
    while q.qsize()>0:
        x,y = q.get()
        print('%s [%s] %s [%s]'%(str(x), type(x), str(y), type(y)))