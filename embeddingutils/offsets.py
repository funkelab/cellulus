import numpy as np


# offset samplers

def uniform_radius_offset_sampler(n, r_min, r_max, z_scale=.1, return_int=True, eps=1e-10):
    def sampler():
        # sample radii
        radii = np.random.uniform(r_min, r_max, size=n)
        # sample directions on unit sphere
        dirs = np.random.uniform(eps, 1, size=(n, 3)) * np.random.choice([-1, 1], (n, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        # multiply radii with directions to get offsets
        result = radii[:, None] * dirs
        result[:, 0] *= z_scale
        if return_int:
            result = np.round(result).astype(int)
        return result
    return sampler


# offset construction

def U(alpha):
    if not isinstance(alpha, np.ndarray):
        alpha = np.array(alpha)
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def add_int_casting(func):
    def decorated(*args, as_int=True, **kwargs):
        result = func(*args, **kwargs)
        if as_int:
            return result.round().astype(int)
        else:
            return result

    return decorated


@add_int_casting
def offset_ring(n=8, radius=1, rot_offset=False, halved=True):
    return radius * np.stack([U(alpha) for alpha in np.arange(0, (2 - int(halved)) * np.pi, 2 * np.pi / n) + (
    np.pi / n if rot_offset else 0)])


@add_int_casting
def concentric_offset_rings(radii, pts_per_ring=8, alternating=False, rot_offset=False, **ring_kwargs):
    if isinstance(pts_per_ring, int):
        pts_per_ring = (pts_per_ring,) * len(radii)
    return np.concatenate(
        [offset_ring(n, r, as_int=False, rot_offset=((alternating and (i % 2 == 1)) != rot_offset), **ring_kwargs)
         for i, (n, r) in enumerate(zip(pts_per_ring, radii))], axis=0)


def stack_rings(offset_rings):
    return np.concatenate([np.concatenate([np.full((len(ring), 1), z), ring], axis=1)
                           for z, ring in enumerate(offset_rings)])