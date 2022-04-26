import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from OANN.src.models.utils import binarize

def parse_bool_from_string(bool_string):
    # assume bool_string is either 0 or 1 (str)
    if str(bool_string)=='1': return True
    elif str(bool_string)=='0': return False
    else: raise RuntimeError('parse_bool_from_string only accepts 0 or 1.')
strbool_description = 'bool by string 1 or 0 (avoid store_true problem)'

def readjust_bools(args, dargs, BOOLS):
    for bkey in BOOLS:
        adjusted_bool = parse_bool_from_string(dargs[bkey])
        setattr(args, bkey, adjusted_bool)
        dargs[bkey] = adjusted_bool
        # print(bkey, 'true') if dargs[bkey] else print(bkey, 'false')
        # print(bkey, 'true') if getattr(args,bkey) else print(bkey, 'false')
    return args, dargs

def create_folder_if_not_exist(folder_dir):
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)

def manage_dir(args):
    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    CKPT_DIR = os.path.join(args['ROOT_DIR'], 'Checkpoint')
    create_folder_if_not_exist(CKPT_DIR)

    args['CKPT_DIR'] = CKPT_DIR
    return args


def simple_evaluation(X,Y, net, header_text='', output_mode='continuous', verbose=20):
    print('simple_evaluation()')

    if output_mode=='continuous':
        simple_evaluation_cont_var(X,Y, net, header_text=header_text,  verbose=verbose)
    elif output_mode=='discrete':
        simple_evaluation_discrete_var(X,Y, net, header_text=header_text,verbose=verbose)

def simple_evaluation_discrete_var(X,Y, net, header_text='',verbose=20):
    print('simple_evaluation_discrete_var...')
    if verbose>=20: 
        print(header_text)

    N,D = X.shape
    cumulated_errors = 0.
    N_MISSED_ACTIVATIONS, N_large_error = 0, 0

    # Compute error excluding missed activations.
    cumulated_exclusive_errors = 0.
    N_ex = 0.

    epsilon = 1e-7
    if verbose>=100:
        print('%-6s %-2s   %-10s %-32s '%(str(' %s'%('i')),str('Layer'), 'abs error', '|y-y0|', ))
    for i in range(N):
        y, act, ACTIVATION_STATUS, layer_k = net.SQANN_propagation(X[i,:])

        if ACTIVATION_STATUS == 'MISS':
            N_MISSED_ACTIVATIONS+=1
            print_error = '%-6s MISSED'%(str('[%s]'%(str(i))))
            error = 1.
        else:
            error = np.mean((y- binarize(Y[i],net.n_class))**2)
            N_ex+=1
            cumulated_exclusive_errors += error

            error_text = '|(%-5s) - (%-5s))|'%(str(np.round(y,3)),str(np.round(Y[i],3)))
            print_error = '%-6s L=%-8s %-10s %-32s  '%(str('[%s]'%(str(i))),str(layer_k),str(np.round(error,3)),error_text)

        if verbose>=100:
            print( '%-48s'%(str(print_error)), )

        if error>0.1: 
            N_large_error+=1
        cumulated_errors += error

    print('N_MISSED_ACTIVATIONS:%s, N_large_error (>0.1):%s'%(str(N_MISSED_ACTIVATIONS), str(N_large_error)))
    print('avg error          : %7s'%(
        str(np.round(cumulated_errors/N,5)),))
    print('avg exclusive error: %7s'%(
        str(np.round(cumulated_exclusive_errors/N_ex,5))))


def simple_evaluation_cont_var(X,Y, net, header_text='',verbose=20):
    print('simple_evaluation_cont_var...')
    if verbose>=20: 
        print(header_text)

    N,D = X.shape
    cumulated_errors, cumulated_frac_err = 0., 0.
    N_MISSED_ACTIVATIONS, N_large_error = 0, 0

    # Compute error excluding missed activations.
    cumulated_exclusive_errors, cumulated_exclusive_frac_errors = 0., 0.
    N_ex = 0.

    epsilon = 1e-7
    if verbose>=100:
        print('%-6s %-2s   %-10s %-32s '%(str(' %s'%('i')),str('Layer'), 'abs error', '|y-y0|', ))
    for i in range(N):
        y, act, ACTIVATION_STATUS, layer_k = net.SQANN_propagation(X[i,:])

        if ACTIVATION_STATUS == 'MISS':
            N_MISSED_ACTIVATIONS+=1
            print_error = '%-6s MISSED'%(str('[%s]'%(str(i))))
            error = 1.
        else:
            error = np.mean((y-Y[i])**2)
            N_ex+=1
            cumulated_exclusive_errors += error
            cumulated_exclusive_frac_errors +=  (error+epsilon)/(np.mean(Y[i]**2)+epsilon)

            error_text = '|(%-5s) - (%-5s))|'%(str(np.round(y,3)),str(np.round(Y[i],3)))
            print_error = '%-6s L=%-8s %-10s %-32s  '%(str('[%s]'%(str(i))),str(layer_k),str(np.round(error,3)),error_text)

        if verbose>=100:
            print( '%-48s'%(str(print_error)), )

        if error>0.1: 
            N_large_error+=1
        cumulated_errors += error
        cumulated_frac_err += (error+epsilon)/(np.mean(Y[i]**2)+epsilon)
    print('N_MISSED_ACTIVATIONS:%s, N_large_error (>0.1):%s'%(str(N_MISSED_ACTIVATIONS), str(N_large_error)))
    print('avg error          : %7s, avg_frac_error          : %7s '%(
        str(np.round(cumulated_errors/N,5)), str(np.round(cumulated_frac_err/N,5)),))
    print('avg exclusive error: %7s, avg exclusive frac error: %7s'%(
        str(np.round(cumulated_exclusive_errors/N_ex,5)),str(np.round(cumulated_exclusive_frac_errors/N_ex,5))))


def ood_searcher(X,Y,net,N, error_th):
    # OOD: OUT OF DISTRIBUTIONS
    # Find all indices of data that give large errors. We want to include them in the construction
    OOD_INDICES_COLLECTION = []
    for i in range(N):
        y, act, ACTIVATION_STATUS, info_ = net.SQANN_propagation(X[i,:], ALLOW_INTERPOLATION=True)

        error = np.abs(y-Y[i])
        if error>error_th:
            OOD_INDICES_COLLECTION.append(i)

    return OOD_INDICES_COLLECTION    

def standard_evaluation(X, Y, net, get_interp_indices=False, output_mode='continuous',verbose=100):
    if output_mode=='continuous':
        return standard_evaluation_cont_var(X,Y,net,get_interp_indices=get_interp_indices, verbose=verbose)
    elif output_mode=='discrete':
        return standard_evaluation_discrete_var(X,Y,net,get_interp_indices=get_interp_indices, verbose=verbose)

def standard_evaluation_cont_var(X,Y,net,get_interp_indices=False, verbose=100):
    INTERP_INDICES = []

    N,D = X.shape
    cumulated_errors, cumulated_frac_err = 0., 0.
    N_INTERPOLATED, N_large_error = 0, 0

    # Compute error excluding missed activations.
    cumulated_exclusive_errors, cumulated_exclusive_frac_errors = 0., 0.
    N_ex = 0.

    epsilon = 1e-7
    if verbose>=100:
        print('%-6s %-6s %s'%(str(' %s'%('i')),str('Layer'), 'rms error'))    
    for i in range(N):
        y, act, ACTIVATION_STATUS, info_ = net.SQANN_propagation(X[i,:], ALLOW_INTERPOLATION=True)
        
        error = np.mean((y-Y[i])**2)**0.5

        if ACTIVATION_STATUS == 'INTERPOLATE':
            interp_info = info_
            layers = [y[1] for _,y in interp_info.items()]
            N_INTERPOLATED+=1
            print_error = '%-6s L=%-2s  '%(str('[%s]'%(str(i))),str('%s'%(str(layers))))

            if get_interp_indices:
                INTERP_INDICES.append(i)

        elif 'HIT':
            layer_k = info_
            N_ex+=1
            cumulated_exclusive_errors += error
            cumulated_exclusive_frac_errors +=  (error+epsilon)/(np.mean(Y[i]**2)+epsilon)
            print_error = '%-6s L=%-6s '%(str('[%s]'%(str(i))),str(layer_k),)
        else:
            raise NotImplementedError('What activation status is this?')
        if verbose>=100:
            print( '%-16s %5s      %-s'%(str(print_error), np.round(error,3), str(ACTIVATION_STATUS)))

        if error>0.1: 
            N_large_error+=1
        cumulated_errors += error
        cumulated_frac_err += (error+epsilon)/(np.mean(Y[i]**2)**0.5+epsilon)

    mean_error = np.round(cumulated_errors/N,5)
    mean_frac_error = np.round(cumulated_frac_err/N, 5)
    mean_ex_error = np.round(cumulated_exclusive_errors/N_ex,5) if N_ex>0 else 'N.A.'
    mean_ex_frac_error = np.round(cumulated_exclusive_frac_errors/N_ex,5) if N_ex>0 else 'N.A.'
    if verbose>=100:
        print('N_INTERPOLATED:%s, N_large_error (>0.1):%s'%(str(N_INTERPOLATED), str(N_large_error)))
        print('avg error          : %7s, avg_frac_error          : %7s '%(
            str(mean_error), str(mean_frac_error),))
        print('avg exclusive error: %7s, avg exclusive frac error: %7s'%(
            str(mean_ex_error),str(mean_frac_error)))


    RESULTS = {
        'N_INTERPOLATED': N_INTERPOLATED,
        'mean_error': mean_error,
        'mean_frac_error': mean_frac_error,
        'mean_ex_error': mean_ex_error,
        'mean_ex_frac_error': mean_ex_frac_error,
    }

    if get_interp_indices:
        return RESULTS, INTERP_INDICES
    return RESULTS

def standard_evaluation_discrete_var(X,Y,net,get_interp_indices=False, verbose=100):
    INTERP_INDICES = []

    N,D = X.shape
    cumulated_errors = 0.
    N_INTERPOLATED, N_large_error = 0, 0

    # Compute error excluding missed activations.
    cumulated_exclusive_errors = 0.
    N_ex = 0.

    epsilon = 1e-7
    if verbose>=100:
        print('%-6s %-6s %s'%(str(' %s'%('i')),str('Layer'), 'rms error'))    
    for i in range(N):
        y, act, ACTIVATION_STATUS, info_ = net.SQANN_propagation(X[i,:], ALLOW_INTERPOLATION=True)
        
        error = np.mean((y- binarize(Y[i],net.n_class) )**2)**0.5

        if ACTIVATION_STATUS == 'INTERPOLATE':
            interp_info = info_
            layers = [y[1] for _,y in interp_info.items()]
            N_INTERPOLATED+=1
            print_error = '%-6s L=%-2s  '%(str('[%s]'%(str(i))),str('%s'%(str(layers))))

            if get_interp_indices:
                INTERP_INDICES.append(i)

        elif 'HIT':
            layer_k = info_
            N_ex+=1
            cumulated_exclusive_errors += error
            print_error = '%-6s L=%-6s '%(str('[%s]'%(str(i))),str(layer_k),)
        else:
            raise NotImplementedError('What activation status is this?')
        if verbose>=100:
            print( '%-16s %5s      %-s'%(str(print_error), np.round(error,3), str(ACTIVATION_STATUS)))

        if error>0.1: 
            N_large_error+=1
        cumulated_errors += error

    mean_error = np.round(cumulated_errors/N,5)
    mean_ex_error = np.round(cumulated_exclusive_errors/N_ex,5) if N_ex>0 else 'N.A.'
    if verbose>=100:
        print('N_INTERPOLATED:%s, N_large_error (>0.1):%s'%(str(N_INTERPOLATED), str(N_large_error)))
        print('avg error          : %7s'%(str(mean_error), ))
        print('avg exclusive error: %7s'%(str(mean_ex_error)))


    RESULTS = {
        'N_INTERPOLATED': N_INTERPOLATED,
        'mean_error': mean_error,
        'mean_ex_error': mean_ex_error,
    }

    if get_interp_indices:
        return RESULTS, INTERP_INDICES
    return RESULTS

def pickle_save(data, DIR):
    output = open(DIR, 'wb')
    pickle.dump(data, output)
    output.close()

def pickle_load(DIR):
    pkl_file = open(DIR, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data
        

def off_ticks(i,j,nj):
    if j+1<nj:
        plt.gca().set_xticks([])
    if i+1>1:
        plt.gca().set_yticks([])


def get_file_extension(THIS_DIR):
    segments = THIS_DIR.split('.')
    return segments[-1]

