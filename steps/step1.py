import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr.viewer import open_mesh, visualize_mesh
from cbsr.config import ORIGINAL_DB


database_root = ORIGINAL_DB
# Change this variable if you want to see a different object
filename = f"{database_root}/Bird/D00089.obj"

mesh = open_mesh(filename)
# This is important for lighting
mesh.compute_vertex_normals()

visualize_mesh(mesh)