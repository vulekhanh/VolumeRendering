# GPU Volume Renderer (Vispy + OpenGL GLSL)

Single-file example implementing GPU raymarching (direct volume rendering)
for medical volumes (RAW, MHD/RAW pairs, or simple .raw binary files).

## Features:

- Load .mhd/.raw pairs (via SimpleITK) and raw binary .raw files
- Upload 3D texture to GPU
- Raymarching fragment shader (front-to-back compositing)
- Transfer function (1D colormap texture) and basic Phong lighting
- Interactive controls: mouse rotate, scroll zoom, keys to adjust steps/opacity

## Dependencies:

- Python 3.8+
- numpy
- vispy
- SimpleITK (for MHD/RAW reading)
- imageio (optional, for saving screenshots)

## Instruction:

Run:
python volume_renderer_vispy.py /path/to/volume.mhd
or
python volume_renderer_vispy.py /path/to/volume.raw --dims 512 512 200 --dtype uint16

Notes:

- For .mhd files SimpleITK will read spacing/origin and data cleanly.
- For raw you need to provide dimensions and dtype (use --dims and --dtype).
- The app creates a 3D texture; large volumes require enough GPU memory.
