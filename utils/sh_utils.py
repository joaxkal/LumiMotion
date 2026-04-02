import torch
import numpy as np


def RGB2SH(rgb):
    c1 = 0.282095
    return (rgb - 0.5) / c1

def SH2RGB(sh):
    c1 = 0.282095
    return sh *c1 + 0.5

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]


    Redundant, the same in eval_sh_point. TODO: remove one fuction
    """

    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def eval_sh_point(n, env, deg = 2):
    '''
    Computes lighting from eq.(3) in:
    An Efficient Representation for Irradiance Environment Maps by Ravi Ramamoorthi
    https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf.

    This function evaluates the SH value for a point, used to lit an object with a single point from the environment map.

    Args:
       n - Normal vector, [B, 3] or [W, H, 3]
       env - Vector with SH coefficients, [3x9] or [B, 3, 9]

    Outputs:
       Evaluated SH value for the given normal vector, [B, 3] or [W, H, 3]
    '''

    c = env
    if len(c.shape)==1:
        c=c.unsqueeze(0) #expand for batch
    if c.shape[1]!=(deg+1)**2: # by default we need transpose, but just a check
        c=torch.transpose(c, -1, -2)

    x, y, z = n[..., 0, None], n[..., 1, None], n[..., 2, None]


    # # Convention from pink env map
    # c1 = 0.282095
    # c2 = 0.488603
    # c3 = 1.092548
    # c4 = 0.315392
    # c5 = 0.546274

    # irradiance = (
    #     c[:,0] * c1 +
    #     c[:,1] * c2*y +
    #     c[:,2] * c2*z +
    #     c[:,3] * c2*x +
    #     c[:,4] * c3*x*y +
    #     c[:,5] * c3*y*z +
    #     c[:,6] * c4*(3*z*z-1) +
    #     c[:,7] * c3*x*z +
    #     c[:,8] * c5*(x*x-y*y)
    # )


    # # Convention from 3DGS - the same below but lets keep it because its tested
    # C0 = 0.28209479177387814
    # C1 = 0.4886025119029199
    # C2 = [
    #     1.0925484305920792,
    #     -1.0925484305920792,
    #     0.31539156525252005,
    #     -1.0925484305920792,
    #     0.5462742152960396
    # ]

    # irradiance = (
    #     c[:, 0] * C0 -
    #     c[:, 1] * C1 * y +
    #     c[:, 2] * C1 * z -
    #     c[:, 3] * C1 * x +
    #     c[:, 4] * C2[0] * (x * y) +
    #     c[:, 5] * C2[1] * (y * z) +
    #     c[:, 6] * C2[2] * (2.0 * (z * z) - (x * x) - (y * y)) +
    #     c[:, 7] * C2[3] * (x * z) +
    #     c[:, 8] * C2[4] * ((x * x) - (y * y))
    # )

    # From tensoRF - higher order

    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    irradiance = C0 * c[:, 0]
    if deg > 0:
        irradiance = (irradiance -
                C1 * y * c[:, 1] +
                C1 * z * c[:, 2] -
                C1 * x * c[:, 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            irradiance = (irradiance +
                    C2[0] * xy * c[:, 4] +
                    C2[1] * yz * c[:, 5] +
                    C2[2] * (2.0 * zz - xx - yy) * c[:, 6] +
                    C2[3] * xz * c[:, 7] +
                    C2[4] * (xx - yy) * c[:, 8])

            if deg > 2:
                irradiance = (irradiance +
                        C3[0] * y * (3 * xx - yy) * c[:, 9] +
                        C3[1] * xy * z * c[:, 10] +
                        C3[2] * y * (4 * zz - xx - yy)* c[:, 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[:, 12] +
                        C3[4] * x * (4 * zz - xx - yy) * c[:, 13] +
                        C3[5] * z * (xx - yy) * c[:, 14] +
                        C3[6] * x * (xx - 3 * yy) * c[:, 15])
                if deg > 3:
                    irradiance = (irradiance + C4[0] * xy * (xx - yy) * c[:, 16] +
                            C4[1] * yz * (3 * xx - yy) * c[:, 17] +
                            C4[2] * xy * (7 * zz - 1) * c[:, 18] +
                            C4[3] * yz * (7 * zz - 3) * c[:, 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * c[:, 20] +
                            C4[5] * xz * (7 * zz - 3) * c[:, 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * c[:, 22] +
                            C4[7] * xz * (xx - 3 * yy) * c[:, 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * c[:, 24])

    return irradiance