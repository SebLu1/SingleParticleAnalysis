import numpy as np
from scipy import interpolate
import mrcfile as mrc

class SigmoidFsc:
    def __init__(self, resolution, width):
        a = 1/resolution - width * np.log(6)
        fsc_x = np.arange(1e-4, 1.0, 0.01)
        fsc_y = 0.95 / (1 + np.exp((fsc_x-a)/width)) + 0.05
        fsc_x = 1 / fsc_x
        self.fsc_inter = interpolate.interp1d(fsc_x, fsc_y, fill_value=(0.05, 1), bounds_error=False)

    def __call__(self, grid, voxelsize, noise_factor=0.2):

        fgrid = np.fft.rfft(grid)
        f_amp = np.abs(fgrid)
        (fiz, fiy, fix) = fgrid.shape
        max_r2 = (fix - 1) ** 2

        Z, Y, X = np.meshgrid(np.linspace(-fiz // 2, fiz // 2 - 1, fiz),
                              np.linspace(-fiy // 2, fiy // 2 - 1, fiy),
                              np.linspace(0, fix - 1, fix))

        R2 = X ** 2 + Y ** 2 + Z ** 2
        R2[R2 > max_r2] = 0

        res = np.zeros(R2.shape)
        res[R2 > 0] = max_r2 / R2[R2 > 0] * voxelsize
        res[fiz // 2, fiz // 2, 0] = 1e3
        amp = self.fsc_inter(res)

        amp = np.fft.ifftshift(amp, axes=0)
        amp = np.fft.ifftshift(amp, axes=1)
        fgrid *= amp

        r = np.random.normal(0, 1, size=fgrid.shape)
        i = np.random.normal(0, 1, size=fgrid.shape)
        fgrid += (r + i * 1j) * (f_amp - np.abs(fgrid)) * noise_factor

        return np.fft.irfft(fgrid)


def apply_fsc(density, in_voxel_sz,  res, width, noise=0):

    (iz, iy, ix) = np.shape(density)
    assert ix == iy == iz and iz % 2 == 0

    f = np.fft.rfftn(density)
    (fiz, fiy, fix) = np.shape(f)
    favg = np.mean(np.abs(f))

    b = width
    a = 1/res - b * np.log(6)

    fsc_x = np.arange(1e-4, 0.5, 0.01)
    fsc_y = 1 / (1 + np.exp((fsc_x-a)/b))
    fsc_x = 1 / fsc_x
    fsc_inter = interpolate.interp1d(fsc_x, fsc_y, fill_value=(0, 1), bounds_error=False)

    '''
    plt.plot(1/fsc_x_, fsc_y_, "b")
    x = np.arange(1e-4, 0.5, 0.001)
    plt.plot(x, fsc_inter(1/x), "r")
    plt.plot([1/fsc_x[0], 1/fsc_x[-1]], [1./7, 1./7], 'k--')
    plt.xlabel("Resolution (1/Å)")
    plt.ylabel("FSC")
    plt.show()
    exit(0)
    '''

    f_amp = np.abs(f)

    max_r2 = (fix - 1) ** 2

    Z, Y, X = np.meshgrid(np.linspace(-fiz // 2, fiz // 2 - 1, fiz),
                          np.linspace(-fiy // 2, fiy // 2 - 1, fiy),
                          np.linspace(0, fix - 1, fix))

    R2 = X ** 2 + Y ** 2 + Z ** 2
    R2[R2 > max_r2] = 0

    res = np.zeros(R2.shape)
    res[R2 > 0] = max_r2 / R2[R2 > 0] * in_voxel_sz
    res[fiz // 2, fiz // 2, 0] = 1e3
    amp = fsc_inter(res)

    amp = np.fft.ifftshift(amp, axes=0)
    amp = np.fft.ifftshift(amp, axes=1)
    f *= amp

    r = np.random.normal(0, 1, size=f.shape)
    i = np.random.normal(0, 1, size=f.shape)
    f += (r + i * 1j) * (f_amp - np.abs(f)) * noise

    return np.fft.irfftn(f)


def grid_rot90(m, i):
    assert 0 <= i < 24
    if i == 0 : return m
    if i == 1 : return np.rot90(m,                      1, (0, 2))
    if i == 2 : return np.rot90(m,                      2, (0, 2))
    if i == 3 : return np.rot90(m,                      3, (0, 2))
    if i == 4 : return np.rot90(m,                      1, (1, 2))
    if i == 5 : return np.rot90(m,                      1, (2, 1))
    if i == 6 : return          np.rot90(m, 1, (0, 1))
    if i == 7 : return np.rot90(np.rot90(m, 1, (0, 1)), 1, (0, 2))
    if i == 8 : return np.rot90(np.rot90(m, 1, (0, 1)), 2, (0, 2))
    if i == 9 : return np.rot90(np.rot90(m, 1, (0, 1)), 3, (0, 2))
    if i == 10: return np.rot90(np.rot90(m, 1, (0, 1)), 1, (1, 2))
    if i == 11: return np.rot90(np.rot90(m, 1, (0, 1)), 1, (2, 1))
    if i == 12: return          np.rot90(m, 2, (0, 1))
    if i == 13: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (0, 2))
    if i == 14: return np.rot90(np.rot90(m, 2, (0, 1)), 2, (0, 2))
    if i == 15: return np.rot90(np.rot90(m, 2, (0, 1)), 3, (0, 2))
    if i == 16: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (1, 2))
    if i == 17: return np.rot90(np.rot90(m, 2, (0, 1)), 1, (2, 1))
    if i == 18: return          np.rot90(m, 3, (0, 1))
    if i == 19: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (0, 2))
    if i == 20: return np.rot90(np.rot90(m, 3, (0, 1)), 2, (0, 2))
    if i == 21: return np.rot90(np.rot90(m, 3, (0, 1)), 3, (0, 2))
    if i == 22: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (1, 2))
    if i == 23: return np.rot90(np.rot90(m, 3, (0, 1)), 1, (2, 1))



def save_mrc(box, voxel_size, origin, filename):
    (z, y, x) = box.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(box, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.flush()
    o.update_header_stats()
    o.close()


def rescale_fourier(f, out_sz):
    (fiz, fiy, fix) = np.shape(f)
    fox = int(out_sz / 2 + 1)
    foy = out_sz
    foz = out_sz

    of = np.zeros((foz, foy, fox)) + 0.j

    if fox > fix:
        max_r2 = (fix - 1) ** 2
        for k in range(fiz):
            if k < fix:
                kp = k
            else:
                kp = k - fiz
            for i in range(fiy):
                if i < fix:
                    ip = i
                else:
                    ip = i - fiy
                for j in range(fix):
                    if kp**2 + ip**2 + j**2 <= max_r2:
                        of[kp, ip, j] = f[kp, ip, j]
    else:
        for k in range(foz):
            if k < fox:
                kp = k
            else:
                kp = k - foz
            for i in range(foy):
                if i < fox:
                    ip = i
                else:
                    ip = i - foy
                for j in range(fox):
                    of[kp, ip, j] = f[kp, ip, j]

    return of


def rescale_real(density, out_sz):

    (iz, iy, ix) = np.shape(density)
    if out_sz != ix:
        in_mean = np.mean(density)
        f = np.fft.rfftn(density)
        f = rescale_fourier(f, out_sz)
        density = np.fft.irfftn(f)
        density *= in_mean / np.mean(density)

    return density


def normalize_voxel_size(density, in_voxel_sz):
    (iz, iy, ix) = np.shape(density)

    assert iz % 2 == 0 and iy % 2 == 0 and ix % 2 == 0
    assert ix == iy == iz

    in_sz = ix
    out_sz = int(round(in_sz * in_voxel_sz))
    if out_sz % 2 != 0:
        vs1 = in_voxel_sz * in_sz / (out_sz + 1)
        vs2 = in_voxel_sz * in_sz / (out_sz - 1)
        if np.abs(vs1 - 1) < np.abs(vs2 - 1):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = in_voxel_sz * in_sz / out_sz
    density = rescale_real(density, out_sz)

    return density, out_voxel_sz
