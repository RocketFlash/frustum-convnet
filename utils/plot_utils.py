import pandas as pd
import plotly.graph_objects as go
import plotly
import numpy as np
import cv2


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    Args:
        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
        normalize: Whether to normalize the remaining coordinate (along the third axis).

    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.

    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def corners(pred_box):
    """Returns the bounding box corners.

    Args:
        wlh_factor: Multiply width, length, height by a factor to scale the box.

    Returns: First four corners are the ones facing forward.
            The last four are the ones facing backwards.

    """

    # outputs = [[x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry, score],]

    width, length, height = pred_box[8], pred_box[9], pred_box[7]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    ry = pred_box[10]
    # Rotate
    # corners = np.dot(self.orientation.rotation_matrix, corners)

    # Translate
    x, y, z = pred_box[4], pred_box[5], pred_box[6]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def render_sample_3d_interactive(
            pc,
            image,
            pred_boxes,
            save_name='filename',
            render_sample: bool = True,
            render_gt: bool = False,
    ) -> None:
        """Render 3D visualization of the sample using plotly
        Args:
            sample_id: Unique sample identifier.
            render_sample: call self.render_sample (Render all LIDAR and camera sample_data in sample along with annotations.)
        """

        # outputs = [[x1, y1, x2,  y2, tx, ty, tz, h, w, l, ry, score],]
        
        df_tmp = pd.DataFrame(pc[:,:3], columns=['x', 'y', 'z'])
        df_tmp['norm'] = np.sqrt(np.power(df_tmp[['x', 'y', 'z']].values, 2).sum(axis=1))
        scatter = go.Scatter3d(
            x=df_tmp['x'],
            y=df_tmp['y'],
            z=df_tmp['z'],
            mode='markers',
            marker=dict(
                size=1,
                color=df_tmp['norm'],
                opacity=0.8
            )
        )

        x_lines = []
        y_lines = []
        z_lines = []

        x_gt_lines = []
        y_gt_lines = []
        z_gt_lines = []

        def f_lines_add_nones():
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        def f_gt_lines_add_nones():
            x_gt_lines.append(None)
            y_gt_lines.append(None)
            z_gt_lines.append(None)

        ixs_box_0 = [0, 1, 2, 3, 0]
        ixs_box_1 = [4, 5, 6, 7, 4]


        for box in pred_boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
            points = view_points(corners(box), view=np.eye(3), normalize=False)
            x_lines.extend(points[0, ixs_box_0])
            y_lines.extend(points[1, ixs_box_0])
            z_lines.extend(points[2, ixs_box_0])
            f_lines_add_nones()
            x_lines.extend(points[0, ixs_box_1])
            y_lines.extend(points[1, ixs_box_1])
            z_lines.extend(points[2, ixs_box_1])
            f_lines_add_nones()
            for i in range(4):
                x_lines.extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
                y_lines.extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
                z_lines.extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])
                f_lines_add_nones()

        lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name='lines',
            line=go.scatter3d.Line(
                color="blue"
            )
        )

        gt_lines = go.Scatter3d(
            x=x_gt_lines,
            y=y_gt_lines,
            z=z_gt_lines,
            mode='lines',
            name='lines',
            line=go.scatter3d.Line(
                color="red"
            )
        )

        fig = go.Figure(data=[scatter, lines, gt_lines])
        fig.update_layout(scene_aspectmode='data')
        plotly.offline.plot(fig, filename = '/root/workdir/frustum-convnet/output_pc/{}.html'.format(save_name), auto_open=False)
        cv2.imwrite('/root/workdir/frustum-convnet/output_pc/{}.png'.format(save_name), image)
        # fig.show()
