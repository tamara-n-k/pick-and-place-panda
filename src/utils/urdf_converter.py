# pip install mjcf-urdf-simple-converter
from mjcf_urdf_simple_converter import convert
convert("model/franka_emika_panda/mjx_panda.xml", "model/franka_emika_panda/mjx_panda.urdf", asset_file_prefix="model/franka_emika_panda/assets")