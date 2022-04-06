import numpy as np
import timeit
import time


# https://gist.github.com/tinybike/d9ff1dad515b66cc0d87
def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    # AG check for nans --------------
    valid_indices = np.logical_and(~ np.isnan(data), ~ np.isnan(weights))
    if np.sum(valid_indices)==0:
        return np.nan
    elif np.sum(valid_indices)==1:
        return data[0]
    data = data[valid_indices]
    weights = weights[valid_indices]

    # ---------------------------------

    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median

def image_stack_weighted_median(data, weights, axis=0):
    if not data.ndim == 3:
        raise ValueError('data and weights must be of size (image_instances, image_height, image_width)')
    if not axis == 0:
        raise ValueError('Only axis=0 implemented')

    A = data
    W = weights
    size = A.shape
    # se pasan las imagenes a 1D.
    # queda AA y WW de tamaño (image_instances, image_height*image_width)
    AA = np.reshape(A, (size[0], size[1] * size[2]))
    WW = np.reshape(W, (size[0], size[1] * size[2]))
    # weighted median columna a columa
    WWMM = np.array([weighted_median(AA[:, i], WW[:, i]) for i in range(AA.shape[1])])
    # reshape para volver a (image_instances, image_height*image_width)
    WM = np.reshape(WWMM, size[1:])
    return WM


def test_weighted_median():
    data = [
        [333],
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10],
        [7, 1, 2, 4, 10, 15],
        [1, 2, 4, 7, 10, 15],
        [0, 10, 20, 30],
        [1, 2, 3, 4, 5],
        [30, 40, 50, 60, 35],
        [2, 0.6, 1.3, 0.3, 0.3, 1.7, 0.7, 1.7, 0.4],
    ]
    weights = [
        [1],
        [1, 1 / 3, 1 / 3, 1 / 3, 1],
        [1, 1, 1, 1, 1],
        [1, 1 / 3, 1 / 3, 1 / 3, 1, 1],
        [1 / 3, 1 / 3, 1 / 3, 1, 1, 1],
        [30, 191, 9, 0],
        [10, 1, 1, 1, 9],
        [1, 3, 5, 4, 2],
        [2, 2, 0, 1, 2, 2, 1, 6, 0],
    ]
    answers = [333,7, 4, 8.5, 8.5, 10, 2.5, 50, 1.7]
    for datum, weight, answer in zip(data, weights, answers):
        assert (weighted_median(datum, weight) == answer)




# TODO: Pasar los tests a una funcion
# def test_ndarray_weighted_median():
#     np.random.seed(1234)
#
#     # coincide con la mediana cuando los pesos son iguales ?
#     size = np.random.randint(3,10,())
#     A = np.random.randint(1, 10, size).astype(np.double)
#     W = np.ones_like(A) * np.random.rand(1)
#     assert(image_stack_weighted_median(A,W) == np.nanmedian(A))


if __name__ == "__main__":
    # TEST ORIGINAL PARA ARRAYS 1D
    #test_weighted_median()

    size = (30, 512, 512) # 15 segundos aprox
    size = (5, 3,4)


    A = np.random.randint(1, 10, size).astype(np.double)
    W = np.random.rand(*size)
    # W = np.ones_like(A)*.1  # para comparar con la mediana

    A[:, 0, 0] = np.nan
    W[0, 0, 0] = np.nan

    AA = np.reshape(A, (size[0], size[1] * size[2]))
    WW = np.reshape(W, (size[0], size[1] * size[2]))

    print('--------------')
    print(A)
    print('--------------')
    print(W)

    print('--------------')
    print(AA)
    print('--------------')
    print(WW)
    print('--------------')


    start = time.time()
    WM = image_stack_weighted_median(A, W)
    print('--------------')
    print('Weighted Median')
    print(WM)
    print('--------------')
    print('Median')
    print(np.nanmedian(A, axis=0))
    print('--------------')
    print('Size: {}'.format(size))
    print('Elapsed time: {}'.format(time.time() - start))


