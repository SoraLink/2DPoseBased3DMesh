def render_cut_mesh_overlay(image_path: str, mesh, pred_cam: dict, out_path: str,
                            color=(0.30, 0.75, 0.95), alpha=0.75):
    """
    High-quality custom renderer for cut / residual mesh.
    Suitable for paper visualization.

    mesh: trimesh.Trimesh
    pred_cam: {
        'focal': np.array([fx, fy]),
        'princpt': np.array([cx, cy]),
    }
    """
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import cv2
    import numpy as np
    import trimesh
    import pyrender

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    fx, fy = pred_cam["focal"]
    cx, cy = pred_cam["princpt"]

    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces)

    # OpenCV camera coords -> OpenGL coords for pyrender
    vertices[:, 1] *= -1.0
    vertices[:, 2] *= -1.0

    vertex_colors = np.ones((vertices.shape[0], 4), dtype=np.float32)
    vertex_colors[:, 0] = color[0]
    vertex_colors[:, 1] = color[1]
    vertex_colors[:, 2] = color[2]
    vertex_colors[:, 3] = alpha

    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(render_mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    scene.add(camera, pose=np.eye(4))

    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.8)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)

    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[:3, 3] = np.array([0.2, 0.2, 0.5])

    scene.add(light1, pose=pose1)
    scene.add(light2, pose=pose2)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=w,
        viewport_height=h,
    )

    color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    mesh_rgb = cv2.cvtColor(color_rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    mesh_alpha = color_rgba[:, :, 3:4].astype(np.float32) / 255.0

    out = mesh_rgb * mesh_alpha + img.astype(np.float32) * (1.0 - mesh_alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)

    cv2.imwrite(out_path, out)
    return out_path