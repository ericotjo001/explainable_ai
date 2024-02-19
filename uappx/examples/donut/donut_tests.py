

def redirect_for_testing(dargs, datax, folder_to_class_mapping, DIRS, net):
    rid = dargs['redir_id']
    if rid>0:
        print('redirect_for_testing...')
    else:
        return

    if rid==1:
        test_queue(net)
    elif rid==2:
        test_elastic_set(dargs, net )
    else:
        raise NotImplementedError('what tests?')


def test_queue(net):
    print('test_queue')
    q = net.ix.create_queue_one_runthrough()
    i=0
    while q.qsize()>0:
        i+=1
        print('{%-3s}'%(str(i)),q.get(False))
    if q.empty():
        print('yes empty')

def test_elastic_set(dargs, net ):
    print('test_elastic_set')

    ########### replace this with main function ##########
    def filter(elasticset):
        newelasticset, setaside = [], []
        for i,j in elasticset:
            if (i+j)%7==0:
                setaside.append((i,j))
            else:
                newelasticset.append((i,j))
        return newelasticset, setaside

    ############## MAIN PROCESS ####################
    q = net.ix.create_queue_one_runthrough()
    elasticset, layers, sides = [], [], []
    while q.qsize()>0:
        elasticset, q, kFetcherInfo = net.ix.mould_elasticset_by_queue(elasticset, q)
        elasticset, setaside = filter(elasticset)
        sides.extend(setaside)

        if len(elasticset)>= net.ix.kwidth:
            layers.append(elasticset)
            elasticset = []

        if kFetcherInfo['ALL_USED_UP']:
            if len(elasticset)>0:
                layers.append(elasticset)
    ################################################

    print('\nsides:\n',sides)
    collectors = {}
    n_total = 0
    for l,layer_nodes in enumerate(layers):
        print('layer:',l, layer_nodes)
        n_total+=len(layer_nodes)
        for i,j in layer_nodes:
            if i not in collectors:
                collectors[i] = [j]
            else:
                collectors[i].append(j)
    for i,j in sides:
        collectors[i].append(j)

    print('\ndouble checking!')
    for i, jlist in collectors.items():
        print(i, sorted(jlist))

    print('\nfinal report')
    print('n_taken:', n_total)
    print('n_aside:', len(sides))
    print('total:', n_total+len(sides))    
