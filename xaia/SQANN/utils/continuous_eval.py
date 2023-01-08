
import numpy as np

MSE = {}
def compute_discrete_error(y_pred,y0):
    return np.mean((y_pred-y0)>0.)

def standard_evaluation_cont(X, Y, net, get_interp_indices=False, verbose=100):
    INTERP_INDICES = []

    N,D = X.shape
    cumulated_errors, cumulated_frac_err = 0., 0.
    N_INTERPOLATED, N_large_error = 0, 0

    # Compute error excluding missed activations.
    cumulated_exclusive_errors, cumulated_exclusive_frac_errors = 0., 0.
    N_ex = 0.

    if verbose>=100:
        print('%-6s %-6s    %-32s %s'%(str(' %s'%('i')),str('Layer'),'|y-y0|', 'abs error'))    
    for i in range(N):
        y, act, ACTIVATION_STATUS, info_ = net.SQANN_propagation(X[i,:], ALLOW_INTERPOLATION=True)

        error = 1. if np.abs(y-Y[i])>0. else 0.
        error_text = '|(%-5s) - (%-5s))|'%(str(np.round(y,3)),str(np.round(Y[i],3)))

        if ACTIVATION_STATUS == 'INTERPOLATE':
            interp_info = info_
            layers = [y[1] for _,y in interp_info.items()]
            N_INTERPOLATED+=1
            print_error = '%-6s L=%-2s    %-32s  '%(str('[%s]'%(str(i))),str('%s'%(str(layers))),error_text)

            if get_interp_indices:
                INTERP_INDICES.append(i)

        elif 'HIT':
            layer_k = info_
            N_ex+=1
            cumulated_exclusive_errors += error
            print_error = '%-6s L=%-6s    %-32s  '%(str('[%s]'%(str(i))),str(layer_k),error_text)
        else:
            raise NotImplementedError('What activation status is this?')
        if verbose>=100:
            print( '%-48s %5s      %-s'%(str(print_error), np.round(error,3), str(ACTIVATION_STATUS)))

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
        'mean_ex_error': mean_ex_error,
    }

    if get_interp_indices:
        return RESULTS, INTERP_INDICES
    return RESULTS