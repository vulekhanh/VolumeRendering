"""
GPU Volume Renderer (Vispy + OpenGL GLSL)

Single-file example implementing GPU raymarching (direct volume rendering)
for medical volumes (RAW, MHD/RAW pairs, or simple .raw binary files).

Features:
- Load .mhd/.raw pairs (via SimpleITK) and raw binary .raw files
- Upload 3D texture to GPU
- Raymarching fragment shader (front-to-back compositing)
- Transfer function (1D colormap texture) and basic Phong lighting
- Interactive controls: mouse rotate, scroll zoom, keys to adjust steps/opacity

Dependencies:
- Python 3.8+
- numpy
- vispy
- SimpleITK (for MHD/RAW reading)
- imageio (optional, for saving screenshots)

Run:
  python volume_renderer_vispy.py /path/to/volume.mhd
  or
  python volume_renderer_vispy.py /path/to/volume.raw --dims 512 512 200 --dtype uint16

Notes:
- For .mhd files SimpleITK will read spacing/origin and data cleanly.
- For raw you need to provide dimensions and dtype (use --dims and --dtype).
- The app creates a 3D texture; large volumes require enough GPU memory.

"""

import sys
import os
import argparse
import numpy as np
import math
import vispy.app
import vispy.gloo as gloo
from vispy.util.transforms import perspective, translate, rotate


# Try to import SimpleITK for medical file reading
try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False

# ---------------------- Utilities: volume IO ----------------------


def load_mhd(path):
    if not HAS_SITK:
        raise RuntimeError(
            "SimpleITK is required to load .mhd files. Install with `pip install SimpleITK`")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x with z as slice index
    # convert to z,x,y ordering for convenience (we'll use zyx -> z is depth)
    arr = arr.astype(np.float32)
    return arr, img.GetSpacing(), img.GetOrigin()


def load_raw(path, dims, dtype=np.uint16, endian='<'):
    # dims: (width, height, depth) or (x,y,z)
    dtype = np.dtype(dtype)
    dtype = dtype.newbyteorder(endian)
    data = np.fromfile(path, dtype=dtype)
    expected = int(dims[0]) * int(dims[1]) * int(dims[2])
    if data.size != expected:
        raise ValueError(
            f"Raw file size ({data.size}) doesn't match dims {dims} (expected {expected})")
    data = data.reshape((dims[2], dims[1], dims[0])).astype(np.float32)
    return data, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)

def load_shader(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
# ---------------------- GLSL Shaders ----------------------

VERTEX_SHADER = load_shader("shaders/vertex_shader.glsl")

FRAGMENT_SHADER = load_shader("shaders/fragment_shader.glsl")

# ---------------------- Vispy Canvas ----------------------


def safe_translate(M, offset):
    """Return a translated 4x4 matrix (avoids vispy's strict len(offset) check)."""
    offset = np.asarray(offset, dtype=np.float32).ravel()
    if offset.size != 3:
        raise ValueError("Offset must have exactly 3 elements")
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = offset
    return np.dot(M, T)


def safe_rotate(angle, axis, dtype=np.float32):
    """4x4 rotation matrix for rotation about a vector."""
    angle = np.radians(angle)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c,     cy * x - z * s, cz * x + y * s, 0.0],
                  [cx * y + z * s, cy * y + c,     cz * y - x * s, 0.0],
                  [cx * z - y * s, cy * z + x * s, cz * z + c,     0.0],
                  [0.0,            0.0,            0.0,            1.0]], dtype)
    return M.T


class VolumeCanvas(vispy.app.Canvas):
    def __init__(self, volume, spacing=(1, 1, 1)):
        vispy.app.Canvas.__init__(self, keys='interactive', size=(
            900, 700), title='GPU Volume Renderer')
        self.volume = volume
        self.spacing = np.array(spacing, dtype=np.float32)

        # normalize volume to 0..1
        vmin = float(np.min(volume))
        vmax = float(np.max(volume))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        vol_norm = (volume - vmin) / (vmax - vmin)
        vol_norm = vol_norm.astype(np.float32)

        self.vol_tex = gloo.Texture3D(
            vol_norm, interpolation='linear', wrapping='clamp_to_edge')

        # create a simple transfer function (grayscale -> alpha ramp)
        tf = self.make_default_transfer_function()
        self.tf_tex = gloo.Texture2D(
            tf, interpolation='linear', wrapping='clamp_to_edge')

        # program
        self.program = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)

        # fullscreen quad
        quad = np.array([[-1.0, -1.0], [1.0, -1.0],
                        [-1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self.program['a_position'] = quad

        self.program['u_volume'] = self.vol_tex
        self.program['u_transfer'] = self.tf_tex
        x, y, z = volume.shape[2], volume.shape[1], volume.shape[0]
        self.program['u_vol_dims'] = (float(x), float(y), float(z))
        self.program['u_steps'] = 256.0
        self.program['u_opacity_scale'] = 1.0

        # camera / model transforms (we render the unit cube centered)
        self._fov = 45.0
        self._near = 0.1
        self._far = 100.0
        self._translate = [0, 0, -2.5]
        self._rotation = [0, 0, 0]

        self._mouse_pos = None
        self._zoom = 1.0

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.timer = vispy.app.Timer('auto', connect=self.on_timer, start=True)

        self.show()

    def make_default_transfer_function(self):
        # A simple ramp: intensity to rgba; using 256 entries in width
        size = 512
        tf = np.zeros((size, 4), dtype=np.float32)
        for i in range(size):
            t = i / float(size-1)
            tf[i, 0:3] = t ** 1.0  # grayscale
            tf[i, 3] = smoothstep(0.0, 1.0, (t - 0.05) / 0.95) * 0.9
        # store in a 2D texture (height 1)
        return np.tile(tf[np.newaxis, :, :], (1, 1, 1)).reshape((1, size, 4))

    def on_timer(self, event):
        self.update()

    def on_key_press(self, event):
        key = event.key.name if event.key is not None else None
        if key == 'Up':
            self.program['u_steps'] = float(self.program['u_steps']) + 32.0
            print('steps ->', self.program['u_steps'])
        elif key == 'Down':
            self.program['u_steps'] = max(
                8.0, float(self.program['u_steps']) - 32.0)
            print('steps ->', self.program['u_steps'])
        elif key == 'Right':
            self.program['u_opacity_scale'] = min(
                3.0, float(self.program['u_opacity_scale']) + 0.1)
            print('opacity ->', self.program['u_opacity_scale'])
        elif key == 'Left':
            self.program['u_opacity_scale'] = max(
                0.01, float(self.program['u_opacity_scale']) - 0.1)
            print('opacity ->', self.program['u_opacity_scale'])

    def on_mouse_wheel(self, event):
        delta = event.delta[1]
        self._translate[2] += delta * 0.1

    def on_mouse_move(self, event):
        if event.is_dragging:
            if self._mouse_pos is None:
                self._mouse_pos = event.pos
                return
            dx = event.pos[0] - self._mouse_pos[0]
            dy = event.pos[1] - self._mouse_pos[1]
            self._mouse_pos = event.pos
            self._rotation[0] += dy * 0.5
            self._rotation[1] += dx * 0.5

    def on_mouse_press(self, event):
        self._mouse_pos = event.pos

    def on_mouse_release(self, event):
        self._mouse_pos = None

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_draw(self, event):
        gloo.clear()
        w, h = self.physical_size
        aspect = w / float(h)
        proj = perspective(self._fov, aspect, self._near, self._far)

        model = np.eye(4, dtype=np.float32)
        # scale to unit cube, apply rotations then translation
        model = np.dot(model, safe_rotate(
            self._rotation[0], np.array([1, 0, 0])))
        model = np.dot(model, safe_rotate(
            self._rotation[1], np.array([0, 1, 0])))
        model = safe_translate(model, [0.0, 0.0, self._translate[2]])

        mv = model
        inv_mv = np.linalg.inv(mv).astype(np.float32)
        self.program['u_inv_modelview'] = inv_mv
        # camera pos in model space (we place camera at origin in view space)
        self.program['u_cam_pos'] = (0.0, 0.0, 0.0)

        self.program.draw('triangle_strip')


# small helper function

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------- CLI and startup ----------------------

def main():
    parser = argparse.ArgumentParser(description='GPU Volume Renderer (Vispy)')
    parser.add_argument('path', help='Path to .mhd or .raw file')
    parser.add_argument('--dims', nargs=3, type=int,
                        help='Dimensions for raw (x y z)')
    parser.add_argument('--dtype', type=str, default='uint16',
                        help='dtype for raw (uint8,uint16,float32)')
    parser.add_argument('--endian', type=str,
                        default='little', choices=['little', 'big'])
    args = parser.parse_args()

    path = args.path
    if path.lower().endswith('.mhd') or path.lower().endswith('.mha'):
        volume, spacing, origin = load_mhd(path)
    else:
        if args.dims is None:
            print('For raw files you must provide --dims width height depth')
            sys.exit(1)
        endian = '<' if args.endian == 'little' else '>'
        volume, spacing, origin = load_raw(
            path, args.dims, dtype=args.dtype, endian=endian)

    # If volume is z,y,x -> keep that order
    print('Loaded volume shape (z,y,x):', volume.shape)

    # create canvas and run
    canvas = VolumeCanvas(volume, spacing)
    vispy.app.run()


if __name__ == '__main__':
    main()
