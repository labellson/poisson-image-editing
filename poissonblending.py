from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np


def poisson_blending(source, target, mask, position):
    """
    Apply Poisson blending following the method described in:
    http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf

    Poisson blending involves solving a linear system Ax = b. We store a sparse
    matrix with the unknowns and solve it by Cholesky decomposition

    :param source: input 3 channel blended image (foreground)
    :param target: input 3 channel background image
    :param mask: mask indicating what pixels of source will be blended
    :param position: (i, j) upper left corner where the source image will be
        blended over the target

    :return: blend image with shape of target
    """
    assert len(source.shape) == 3, "source must have 3 channels"
    assert len(target.shape) == 3, "target must have 3 channels"
    assert (source.shape[0] <= target.shape[0] and source.shape[1]
            <= target.shape[1]), "source size must be smaller than target"
    assert source.shape[:2] == mask.shape[:2], "source and mask must have same shape"
    assert (position[0] + source.shape[0] <= target.shape[0] and
            position[1] + source.shape[1] <= target.shape[1]), "mask is out of source bounds"

    hm, wm = mask.shape[:2]

    source = source / 255.
    target = target / 255.
    blend = target.copy()
    blend_roi = blend[position[0]:position[0] + hm,
                      position[1]:position[1] + wm]

    # If mask has 3 channels assume all channels has same values
    if len(mask.shape) > 2:
        mask = mask[..., 1]

    # Every pixel involved will have an unknown variable, so the first pixel of
    # the mask will be variable 0, second 1, ... use only the non zero entries
    # of the mask
    unknown_index = np.argwhere(mask == 255)
    var_num = dict(zip(map(tuple, unknown_index),
                       np.arange(len(unknown_index))))

    # Fill the sparse matrix like in the (7) equation of the paper
    # var_num dict has ordered variables from 0 to len(unknown_index)
    A = sparse.identity(len(unknown_index), format='lil')
    b = np.zeros((len(unknown_index), 3))
    for (i, j), p in zip(var_num.keys(), var_num.values()):
        A[p, p] = 4
        f_star = np.zeros(3)

        # Look for neighboors inside the mask or compute the gradient for b
        if (i, j + 1) in var_num:
            A[p, var_num[(i, j + 1)]] = -1
        else:
            f_star += blend_roi[i, j + 1]

        if (i, j - 1) in var_num:
            A[p, var_num[(i, j - 1)]] = -1
        else:
            f_star += blend_roi[i, j - 1]

        if (i + 1, j) in var_num:
            A[p, var_num[(i + 1, j)]] = -1
        else:
            f_star += blend_roi[i + 1, j]

        if (i - 1, j) in var_num:
            A[p, var_num[(i - 1, j)]] = -1
        else:
            f_star += blend_roi[i - 1, j]

        # f_star + sum(v_pq) gradient; v_pq = g_p - g_q
        b[p] = f_star + 4 * source[i, j] - source[i, j + 1] \
               - source[i, j - 1] - source[i + 1, j] - source[i - 1, j]

    # Solve the lineal system for each channel using scipy spsolve
    A = A.tocsr()
    x = np.zeros_like(b)
    for c in range(3):
        x[:, c] = spsolve(A, b[:, c])
        x[x < 0] = 0
        x[x > 1] = 1

    blend_roi[unknown_index[:, 0], unknown_index[:, 1]] = x
    return (blend * 255).astype(np.uint8)


if __name__ == '__main__':
    import cv2
    source = cv2.imread('./poisson_data/fg.jpg')
    mask = cv2.imread('./poisson_data/mask.jpg', cv2.IMREAD_GRAYSCALE)
    target = cv2.imread('./poisson_data/bg.jpg')
    #pos = (108, 462)
    pos = (100, 100)

    hm, wm = mask.shape

    # Just paste the source
    mask_indices = np.where(mask > 0)
    paste = target.copy()
    paste_roi = paste[pos[0]:pos[0] + hm, pos[1]:pos[1] + wm]
    paste_roi[mask_indices] = source[mask_indices]
    cv2.imshow('Paste', paste)
    cv2.waitKey(0)

    # Poisson blending
    blend = poisson_blending(source, target, mask, pos)
    cv2.imshow('Poisson', blend)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
