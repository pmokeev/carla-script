import os
from tqdm import tqdm
import numpy as np
import copy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import argparse


def color_to_string(color_arr):
    return "{0},{1},{2}".format(color_arr[0], color_arr[1], color_arr[2])


def color_from_string(color_str):
    return np.fromiter((float(channel_str) for channel_str in color_str.split(',')), dtype=np.float64)


def normalize_color(color):
    return np.asarray([channel / 255 for channel in color])


def normalize_color_arr(color_arr):
    return color_arr / 255


def denormalize_color(color):
    return np.round(np.asarray([channel * 255 for channel in color]))


def denormalize_color_arr(color_arr):
    return color_arr * 255


UNSEGMENTED_PCD_COLOR = [0, 0, 0]
UNSEGMENTED_PCD_COLOR_NORMALISED = normalize_color(UNSEGMENTED_PCD_COLOR)
produced_colors_set = {color_to_string(UNSEGMENTED_PCD_COLOR)}


def get_random_color():
    random_color = np.asarray(UNSEGMENTED_PCD_COLOR)
    while color_to_string(random_color) in produced_colors_set:
        random_color = np.asarray([int(x) for x in np.random.choice(range(256), size=3)])

    produced_colors_set.add(color_to_string(random_color))

    return random_color


def get_random_normalized_color():
    return normalize_color(get_random_color())


def filter_small_triangles(mesh: o3d.geometry.TriangleMesh, min_area: float, min_ratio: float):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    mask = np.zeros(triangles.shape[:-1], dtype=bool)
    for triangle_index, triangle in enumerate(triangles):
        p1, p2, p3 = vertices[triangle]

        v1 = p3 - p1
        v2 = p2 - p1
        v3 = p2 - p3
        lens = np.sort(np.array([np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)]))
        cp = np.cross(v1, v2)
        
        if np.linalg.norm(cp) / 2 < min_area or lens[0] / lens[2] < min_ratio:
            mask[triangle_index] = True
    mesh.remove_triangles_by_mask(mask)

    return mesh


def normalize_norm_direction(normals):
    new_normals = []
    for i in range(normals.shape[0]):
        s0 = np.sign(normals[i][0])
        if s0 != 0:
            new_normals.append(normals[i] * s0)
        else:
            s1 = np.sign(normals[i][1])
            if s1 != 0:
                new_normals.append(normals[i] * s1)
            else:
                s2 = np.sign(normals[i][2])
                new_normals.append(normals[i] * s2)
    return np.array(new_normals)


def merge_vertices(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    unique_verts, ind, cnt = np.unique(vertices, axis=0, return_inverse=True, return_counts=True)
    for uniq_vert in unique_verts:
        indices = np.where((vertices == uniq_vert).all(axis=1))[0]
        for ind in indices[1:]:
            triangles[np.where(triangles == ind)] = indices[0]
    
    return mesh


def submesh(mesh, indices):
    triangles = np.asarray(mesh.triangles)[indices]
    sub_vertices = np.asarray(mesh.vertices)[triangles.flatten()]
    triangles_cnt = len(indices)
    sub_triangles = np.arange(triangles_cnt * 3).reshape((triangles_cnt, 3))
    
    submesh = o3d.geometry.TriangleMesh()
    submesh.triangles = o3d.utility.Vector3iVector(copy.deepcopy(sub_triangles))
    submesh.vertices = o3d.utility.Vector3dVector(copy.deepcopy(sub_vertices))
    
    return merge_vertices(submesh)


def similarity(x, y):
    return np.abs(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def segment_by_normals(mesh):
    mesh.compute_triangle_normals(normalized=True)
    normals = normalize_norm_direction(np.asarray(mesh.triangle_normals))

    cluster_normals = DBSCAN(eps=1e-1, min_samples=1, n_jobs=10, metric='cosine').fit(normals)
    unique_labels = np.unique(cluster_normals.labels_)
    
    segmented_meshes = []
    for plane_id in unique_labels:
        segmented_meshes.append(submesh(mesh, np.where(cluster_normals.labels_ == plane_id)[0]))
        
    return segmented_meshes


def segment_by_d(mesh):
    mesh.compute_triangle_normals(normalized=True)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)
    vertices = np.asarray(mesh.vertices)
    
    ds = []
    for triangle_index, triangle in enumerate(triangles):        
        ds.append(np.dot(normals[triangle_index], vertices[triangle[2]]))
    ds = np.array(ds).reshape(-1, 1)
    
    cluster_normals = DBSCAN(eps=15, min_samples=1, n_jobs=10).fit(ds)
    
    unique_labels = np.unique(cluster_normals.labels_)
    
    segmented_meshes = []
    for plane_id in unique_labels:
        segmented_meshes.append(submesh(mesh, np.where(cluster_normals.labels_ == plane_id)[0]))
        
    return segmented_meshes


def cut_high_buildings(mesh):
    return mesh.crop(o3d.geometry.AxisAlignedBoundingBox([-1e5,-1e5,-1e5], [1e5, 2000, 1e5]))
    
    
def filter_by_surface_area(mesh_list, area=1e4+800):
    filtered_list = [m for m in mesh_list if m.get_surface_area() > area]
    return filtered_list


def visualize_mesh_list(mesh_list):
    geometries = []
    for mesh in mesh_list:
        mesh.paint_uniform_color(get_random_normalized_color())
        geometries.append(mesh)

    o3d.visualization.draw_geometries(geometries)


def generate(input_path: str, output_path: str):
    objects = os.listdir(input_path)
    for obj in tqdm(objects):
        obj_path = os.path.join(input_path, obj)

        mesh = o3d.io.read_triangle_mesh(obj_path)

        mesh = cut_high_buildings(mesh)
        mesh = filter_small_triangles(mesh, 150, 0.14)
        if np.asarray(mesh.triangles).size == 0:
            continue

        meshes_by_normals = segment_by_normals(mesh)

        segmented_planes = o3d.geometry.PointCloud()
        f_mesh = filter_by_surface_area(meshes_by_normals)
        label_counter = 1
        labels = np.empty((0, 1), int)

        for mesh_by_normal in tqdm(f_mesh):
            meshes = filter_by_surface_area(segment_by_d(mesh_by_normal))    
            for m in meshes:
                pcd_m = m.sample_points_uniformly(number_of_points=int(m.get_surface_area() // 150))
                pcd_m = pcd_m.voxel_down_sample(3)
                cluster_normals = DBSCAN(eps=20, min_samples=1, n_jobs=10).fit(np.asarray(pcd_m.points))

                unique_labels = np.unique(cluster_normals.labels_)

                for plane_id in unique_labels:
                    plane_indices = np.where(cluster_normals.labels_ == plane_id)[0]
                    if len(plane_indices) > 400:
                        new_plane = o3d.geometry.PointCloud()
                        new_plane.points = o3d.utility.Vector3dVector(np.asarray(pcd_m.points)[plane_indices])
                        new_plane.paint_uniform_color(get_random_normalized_color())
                        segmented_planes += new_plane
                        labels = np.append(labels, [label_counter] * len(plane_indices))
                        label_counter += 1
                    
        o3d.io.write_point_cloud(os.path.join(output_path, obj)[:-3] + 'pcd', segmented_planes)
        labels_filename = obj[:-4]+".npy"
        np.save(os.path.join(output_path, labels_filename), labels)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--input_path',
        type=str,
        help="Input path with carla objs"
    )
    argparser.add_argument(
        '--output_path',
        type=str,
        help="Output path for result psd's"
    )
    args = argparser.parse_args()
    
    generate(args.input_path, args.output_path)