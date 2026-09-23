import open3d as o3d
import numpy as np
import pandas as pd
import os


def open_mesh(mesh_path : str):
    # Load the mesh with open3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        print("File: " + mesh_path + " could not be loaded (empty mesh)")
        return None
    # firstly calculate the unit vectors(faces) for each triangle and then average them for each vertex
    return mesh


def get_face_type(mesh_path):
    face_vertex_counts = set()
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                indices = line.strip().split()[1:]  # drop the leading 'f'
                face_vertex_counts.add(len(indices))

    if face_vertex_counts == {3}:
        return "triangles only"
    elif face_vertex_counts == {4}:
        return "quads only"
    return f"mixed {sorted(face_vertex_counts)}"


# The number of vertices 
# The number of faces
# The shape of the class
# the axis-aligned 3D bounding box of the shapes
def get_mesh_data(mesh, file_path):
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)
    class_name = file_path.split('/')[1]
    face_type = get_face_type(file_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    min_b, max_b = bbox.get_min_bound(), bbox.get_max_bound()

    return {
            "class": class_name,
            "file": file_path,
            "num_vertices": num_vertices,
            "num_faces": num_faces,
            "face_type": face_type,
            "bbox_min_x": min_b[0], "bbox_min_y": min_b[1], "bbox_min_z": min_b[2],
            "bbox_max_x": max_b[0], "bbox_max_y": max_b[1], "bbox_max_z": max_b[2],
        }



def visualize_mesh(mesh, vis_option="smoothshade"):
    if vis_option == "smoothshade":
        o3d.visualization.draw_geometries([mesh], width=1280, height=720)
    elif vis_option == "wireframe_on_shaded":
        o3d.visualization.draw_geometries([mesh], width=1280, height=720, mesh_show_wireframe=True)
    elif vis_option == "wireframe":
        # We first need to obtain a lineset of the wireframe if we don't want to render the mesh itself
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh) 
        o3d.visualization.draw_geometries([wireframe], width=1280, height=720)
    elif vis_option == "world_axes":
        # Display the mesh including a world axis system.

        # Create the endpoints of each line. Each line is unit-length.
        # For the world axes, the origin is shared by all lines. So we have 4 endpoints in total
        line_endpoints = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]

        # List of indices into the 'line_endpoints' list, which describes which endpoints form which line
        line_indices = [[0, 1], [0, 2], [0, 3]]

        # Create a line set from the endpoints and indices
        world_axes = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_endpoints),
            lines=o3d.utility.Vector2iVector(line_indices),
        )

        # Render the line set and the mesh
        o3d.visualization.draw_geometries([mesh, world_axes], width=1280, height=720)

    elif vis_option == "black_background":
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)

        # Set render options (e.g. background color)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

        # Run the visualizer
        vis.run()
        vis.destroy_window()


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

# mesh.compute_vertex_normals()
if __name__ == '__main__':
    # For step 1
    export_media_to_csv("ShapeDatabase", "shape_statistics.csv")
    


