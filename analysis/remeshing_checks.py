import os

from cbsr.viewer import open_mesh, visualize_mesh
from cbsr.config import ORIGINAL_DB, REMESHED_DB



def compare_objects(object_filename):
    old_mesh = open_mesh(os.path.join(ORIGINAL_DB, object_filename))
    resampled_mesh = open_mesh(os.path.join(REMESHED_DB, object_filename))


    old_mesh.compute_vertex_normals()
    resampled_mesh.compute_vertex_normals()
    
    print("Before resampling")
    print(len(old_mesh.vertices))
    print("After resampling")
    print(len(resampled_mesh.vertices))

    visualize_mesh(old_mesh, "wireframe_on_shaded")
    visualize_mesh(resampled_mesh, "wireframe_on_shaded")
