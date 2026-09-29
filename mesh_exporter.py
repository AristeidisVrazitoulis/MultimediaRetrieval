import os
import pandas as pd
from mesh_vizualizer import open_mesh, get_mesh_data


# Iterates a whole database and export per each shape its data(vert count, faces count, etc etc)
def export_media_to_csv(database_root, export_filname):
    all_shape_data = []

    for class_name in os.listdir(database_root):
        class_folder = os.path.join(database_root, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            if filename.endswith(".obj"):
                file_path = os.path.join(class_folder, filename)
                # now load file_path with open3d, compute vertices/faces/bbox, etc.
                current_mesh = open_mesh(file_path)
                if current_mesh is not None:
                    all_shape_data.append(get_mesh_data(current_mesh, file_path))

    df = pd.DataFrame(all_shape_data)
    df.to_csv(export_filname, index=False)