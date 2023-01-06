import pandas as pd
import numpy as np

# this is the prototype of cross_projective_augmentation
# Concept: let x be n vectors with shape d. 
#   Assume each vector is associated with some label y0 (an integer)
#   We want to augment x, for example because n is too small compared 
#   to the entire dataset (imbalanced dataset)
#   Then we use x_ref (n2 vectors of shape d), each with a label 0,1,..., y0-1,
#   to agument x, as the following
def cross_projective_augmentation(x, x_ref, cross_factor=5, dev=0.95):
    """ 
    x     : shape (n, d)
    x_ref : shape (n2, d)
    cross_factor: int
    """
    n,d = x.shape
    n2,d2 = x_ref.shape
    assert(d==d2) # vectors of the same dimension
    assert(n2 >= n*cross_factor)

    x_aug = []
    for i in range(n):
        for j in range(cross_factor):
            delta_x = x[i] - x_ref[i*cross_factor+j]

            cross_projection = x_ref[i*cross_factor+j] + dev * delta_x

            x_aug.append(cross_projection)
    return np.array(x_aug)

def cp_augmentation_type_dfc(df, df_ref, cross_factor=5, dev=0.95):
    # dfc : dataframe + class label
    # print(df.shape) # (n,d+1) # Extra 1 column for class label
    # print(df_ref.shape) # (n2,d2+1)

    n,d = df.shape
    n2,d2 = df_ref.shape
    assert(d==d2) # vectors of the same dimension
    assert(n2 >= n*cross_factor)

    x_aug = []
    for i in range(n):
        for j in range(cross_factor):

            # delta_x = df.loc[i] - df_ref[i*cross_factor+j]
            thisrow = np.array(df.loc[i])
            # print(thisrow) # [-1.20279735  0.63248526  1.31965896  2.        ]

            x = thisrow[:-1]
            y = thisrow[-1]
            thisxrow_ref = np.array(df_ref.loc[i*cross_factor+j])
            x_ref = thisxrow_ref[:-1]
            y_ref = thisxrow_ref[-1]
            # print(x,y) # [ 0.21629346 -0.50364941 -0.80736698] 2.0
            # print(x_ref, y_ref) # [-1.69474536 -0.2102739   0.06041401] 1.0
            
            delta_x = x - x_ref
            newrow = np.concatenate(( x_ref + dev * delta_x, [y]))
            newrow = pd.DataFrame([newrow], columns=df.columns )

            x_aug.append(newrow)

    x_aug = pd.concat(x_aug,join="inner")
    return x_aug



if __name__ == '__main__':
    print('cross project augmentation\n')
    np.set_printoptions(precision=3)

    x = np.random.normal(0,1, size=(4, 3)) # small sample to augment
    x_ref = np.random.normal(-0.5,1, size=(25, 3))
    x_aug = cross_projective_augmentation(x, x_ref, cross_factor=5)
    print('===== numpy array =====')
    print('  reference vectors shape:\n', x_ref.shape)
    print('  original vectors  :\n', x)
    print('  augmented vectors :\n',x_aug)

    def get_random_dataframe(n, c=None):
        if c is None:
            c = np.random.randint(0,2, size=(n,))

        df = pd.DataFrame({
            'V1': np.random.normal(0,1., size=(n,)),
            'V2': np.random.normal(0,1., size=(n,)),
            'V3': np.random.normal(0,1., size=(n,)),
            'Class': c,
            })
        return df

    print('\n===== data frame =====')
    nx = 4
    df = get_random_dataframe(nx, c=2+np.zeros(shape=(nx,), dtype=int) )
    print('df:\n', df)
    df_ref = get_random_dataframe(25)
    print('\ndf_ref:', df_ref.shape)
    print(np.array(df_ref['Class']))  #  [0 0 0 1 0 0 1 0 1 0 0 0 0 1 1 1 1 1 1 1 1 0 1 1 1]

    df_aug = cp_augmentation_type_dfc(df, df_ref, cross_factor=5, dev=0.95)
    print('\ndf_aug:', df_aug)