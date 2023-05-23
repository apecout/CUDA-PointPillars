import glob 
import open3d
import numpy as np

INT_2_CLASS = {
    0: 'class_0',
    1: 'class_1',
    2: 'class_4'
}

POINT_CLOUD_RANGE = [-40.0, -80.0, -2, 40.0, 0.0, 20]

box_colormap = [
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
]

def pointcloud_borders():

    dtype_f = open3d.core.float32
    dtype_i = open3d.core.int32

    # Create an empty line set
    # Use lineset.point to access the point attributes
    # Use lineset.line to access the line attributes
    lineset = open3d.geometry.LineSet()

    # Default attribute: point.positions, line.indices
    # These attributes is created by default and are required by all line
    # sets. The shape must be (N, 3) and (N, 2) respectively. The device of
    # "positions" determines the device of the line set.
        
    points = [
        [POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1], 0],
        [POINT_CLOUD_RANGE[3], POINT_CLOUD_RANGE[1], 0],
        [POINT_CLOUD_RANGE[3], POINT_CLOUD_RANGE[4], 0],
        [POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[4], 0]
        ]
    
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]]
    
    lineset.points = open3d.utility.Vector3dVector(points)

    lineset.lines = open3d.utility.Vector2iVector(lines)

    # Common attributes: line.colors
    # Common attributes are used in built-in line set operations. The
    # spellings must be correct. For example, if "color" is used instead of
    # "color", some internal operations that expects "colors" will not work.
    # "colors" must have shape (N, 3) and must be on the same device as the
    # line set.
    lineset.paint_uniform_color([0.3, 0.3, 0.3])
    
    return lineset


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    #borders 
    borders = pointcloud_borders()
    vis.add_geometry(borders)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def read_pp_res(file_pwd): 
    with open(file_pwd, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        boxes = []
        labels = []
        for line in lines:
            values = line.split(' ')
            label = int(values[7])
            box = [
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
                float(values[6])
            ]

            labels.append(label)
            boxes.append(box)

    return np.array(labels), np.array(boxes)

if __name__=='__main__':
    labels = glob.glob('./res/*')
    for label in labels:
        
        labels, boxes = read_pp_res(label)
        pointcloud_pwd = label.replace('txt', 'bin').replace('res', 'data')
        pointcloud = np.fromfile(pointcloud_pwd, dtype=np.float32).reshape(-1, 4)

        draw_scenes(pointcloud, ref_boxes=boxes, ref_labels=labels)

