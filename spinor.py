# @Author: Jianzhou Zhao <jzzhao>
# @Date:   2018-04-04T16:35:30+02:00
# @Email:  jianzhou.zhao@gmail.com
# @Last modified by:   jzzhao
# @Last modified time: 2019-03-19T00:56:41+01:00

import numpy as np

def euler_to_rmat(alpha, beta, gamma):
    """
    Given Euler angle: alpha, beta, gamma, generate the rotation matrix R

    Parameters
    ----------
    alpha:  Euler angle, in radian, [0,2*pi]
    beta:   Euler angle, in radian, [0,  pi]
    gamma:  Euler angle, in radian, [0,2*pi]
   
    Returns
    ----------
    rmat:   The rotation matrix
    """

    rmat=np.zeros((3,3), dtype=float)
    rmat[0,0] =  np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma)
    rmat[0,1] = -np.sin(gamma) * np.cos(alpha) * np.cos(beta) - np.sin(alpha) * np.cos(gamma)
    rmat[0,2] =  np.cos(alpha) * np.sin(beta)
    rmat[1,0] =  np.sin(alpha) * np.cos(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma)
    rmat[1,1] = -np.sin(gamma) * np.sin(alpha) * np.cos(beta) + np.cos(alpha) * np.cos(gamma)
    rmat[1,2] =  np.sin(alpha) * np.sin(beta)
    rmat[2,0] = -np.cos(gamma) * np.sin(beta)
    rmat[2,1] =  np.sin(beta) * np.sin(gamma)
    rmat[2,2] =  np.cos(beta)
    return rmat

def rmat_to_euler(rmat):
    """
    Given rotation matrix, return the Euler angle

    Parameters
    ----------
    rmat:   3*3 matrix

    Returns
    ---------
    alpha:  Euler angle, in radian, [0,2*pi]
    beta:   Euler angle, in radian, [0,  pi]
    gamma:  Euler angle, in radian, [0,2*pi]
   
    """

    if np.abs(rmat[2,2]) < 1.0:
        beta = np.arccos(rmat[2,2]) 

        cos_gamma = -rmat[2,0] / np.sin(beta)
        sin_gamma =  rmat[2,1] / np.sin(beta)
        gamma = where_is_angle(sin_gamma, cos_gamma)
        
        cos_alpha = rmat[0,2] / np.sin(beta)
        sin_alpha = rmat[1,2] / np.sin(beta)
        alpha = where_is_angle(sin_alpha, cos_alpha)
    else:
        # Gimbal lock (sin(beta)=0): only alpha+gamma (beta=0) or alpha-gamma
        # (beta=pi) is defined. Fix gamma=0 and recover alpha from cos/sin so the
        # sign is kept (arccos alone folds alpha into [0,pi] and drops it).
        gamma = 0.0
        if rmat[2,2] > 0:      # beta = 0,  R = Rz(alpha):        [0,0]=cos, [1,0]=sin
            beta = 0.0
            alpha = where_is_angle(rmat[1,0], rmat[0,0])
        else:                  # beta = pi, R = Rz(alpha)Ry(pi): [0,0]=-cos, [1,0]=-sin
            beta = np.pi
            alpha = where_is_angle(-rmat[1,0], -rmat[0,0])
    return alpha, beta, gamma

def where_is_angle(sina, cosa):
    """
    Given sin and cos of an angle, return the angle range from [0,2*pi]

    Parameters
    ----------
    sina:   sin(alpha)
    cosa:   cos(alpha)

    Returns
    ----------
    alpha:  range from [0,2*pi]
    """
    if cosa > 1.0:
        cosa =1.0
    elif cosa < -1.0:
        cosa = -1.0
    alpha = np.arccos(cosa)
    if sina < 0.0:
        alpha = 2.0 * np.pi - alpha
    return alpha

def dmat_spinor(alpha,beta,gamma):
    """
    Given three Euler angle alpha, beta, gamma, return the transformation
    matrix for 1/2 spinor
   
    Parameters
    ----------
    alpha:   Euler angle [0,2*pi]
    beta :   Euler angle [0,  pi]
    gamma:   Euler angle [0,2*pi]
    
    Returns
    ----------
    dmat:    np.ndarray((2,2), dtype=np.complex128)
    """
    dmat = np.zeros((2,2), dtype=complex)
    dmat[0,0] =  np.exp(-(alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0) 
    dmat[0,1] = -np.exp(-(alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0) 
    dmat[1,0] =  np.exp( (alpha-gamma)/2.0 * 1j) * np.sin(beta/2.0) 
    dmat[1,1] =  np.exp( (alpha+gamma)/2.0 * 1j) * np.cos(beta/2.0) 
    return dmat

def zx_to_rmat(z,x):
    """
    Given z vector and x vector, return y vector satisfy right-hand coordinate,
    and normalize them if needed

    Parameters
    ----------
    z:    array-like, shape (3,)
    x:    array-like, shape (3,), orthogonal to z
    
    Returns
    ---------
    rmat:   [x,y,z] np.ndarray((3,3), dtype=np.float64); a rotation up to the
            1e-6 orthogonality tolerance (not exactly SO(3) for slightly
            non-orthogonal input).  Raises ValueError for zero, non-finite or
            non-orthogonal axes.
    """
    z = np.array(z, dtype=float) 
    x = np.array(x, dtype=float) 
    # the result is used as a rotation (Euler angles, D matrices): reject axes
    # that are zero, non-finite or not orthogonal within 1e-6 (as run.py does)
    if z.shape != (3,) or x.shape != (3,) or not (np.isfinite(z).all() and np.isfinite(x).all()):
        raise ValueError(f"zx_to_rmat: z and x must be finite 3-vectors, got {z!r}, {x!r}")
    nz, nx = np.sqrt(np.dot(z,z)), np.sqrt(np.dot(x,x))
    if not (np.isfinite(nz) and np.isfinite(nx)) or nz < 1e-12 or nx < 1e-12:
        raise ValueError(f"zx_to_rmat: z and x must be non-zero with finite length, got {z!r}, {x!r}")
    cos = np.dot(z,x) / (nz*nx)
    if abs(cos) > 1e-6:
        raise ValueError(f"zx_to_rmat: z and x must be orthogonal within 1e-6 (cos angle = {cos:.2e})")
    xx = x / np.sqrt(np.dot(x,x))
    zz = z / np.sqrt(np.dot(z,z))
    yy = np.cross(zz,xx)

    rmat = np.zeros((3,3), dtype=float)
    rmat[:,0] = xx
    rmat[:,1] = yy
    rmat[:,2] = zz

    return rmat

def rotate_spinor(zdir, xdir):

    # Get rotation matrix from Z and X diration
    xyz = zx_to_rmat(zdir, xdir)
    # Get Euler angle from Rotantion matrix
    angle = rmat_to_euler(xyz)
    # Get transformation matrix for spin-1/2 from Euler angle
    dmat = dmat_spinor(*angle)

    return dmat
