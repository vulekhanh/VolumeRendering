import struct
import sys
import os

def convert_dat_to_mhd(dat_file):
    base = os.path.splitext(dat_file)[0]
    raw_file = base + ".raw"
    mhd_file = base + ".mhd"

    # Read header and raw data
    with open(dat_file, "rb") as f:
        # read 6-byte header
        sizeX = struct.unpack("<H", f.read(2))[0]
        sizeY = struct.unpack("<H", f.read(2))[0]
        sizeZ = struct.unpack("<H", f.read(2))[0]
        print(f"Volume size: {sizeX} x {sizeY} x {sizeZ}")

        # read remaining data (unsigned short per voxel)
        data = f.read()

    # Write raw volume data to .raw (skip header)
    with open(raw_file, "wb") as out:
        out.write(data)

    # Create .mhd header
    mhd_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 1 1 1
DimSize = {sizeX} {sizeY} {sizeZ}
ElementType = MET_USHORT
ElementDataFile = {os.path.basename(raw_file)}
"""

    with open(mhd_file, "w") as out:
        out.write(mhd_content)

    print(f"✅ Created {mhd_file} and {raw_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_data.py <file.dat>")
        sys.exit(1)
    convert_dat_to_mhd(sys.argv[1])
