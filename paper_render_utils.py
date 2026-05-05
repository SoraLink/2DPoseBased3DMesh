def render_cut_mesh_overlay(
    image_path: str,
    mesh,
    pred_cam: dict,
    out_path: str,
    color=(0.92, 0.92, 0.92),
    alpha=1.0,
    edge_color=(40, 95, 140),
    edge_alpha=0.0,
    edge_width=0,
):
    """
    High-quality opaque renderer for cut / residual mesh.

    This version keeps the old function signature for compatibility, but:
      - does not use transparency blending
      - does not draw wireframe / stripe lines
      - renders an opaque shaded mesh on top of the image
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

    vertices_cv = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces)

    # OpenCV camera coordinates -> OpenGL camera coordinates for pyrender
    vertices_gl = vertices_cv.copy()
    vertices_gl[:, 1] *= -1.0
    vertices_gl[:, 2] *= -1.0

    tri_mesh = trimesh.Trimesh(
        vertices=vertices_gl,
        faces=faces,
        process=False,
    )

    # Opaque material: alpha is kept in the signature but not used for blending.
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.05,
        roughnessFactor=0.72,
        alphaMode="OPAQUE",
        baseColorFactor=(color[0], color[1], color[2], 1.0),
    )

    render_mesh = pyrender.Mesh.from_trimesh(
        tri_mesh,
        material=material,
        smooth=True,
    )

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.28, 0.28, 0.28],
    )
    scene.add(render_mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    scene.add(camera, pose=np.eye(4))

    # Lighting only, no wireframe.
    light_main = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
    light_back = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)

    pose_main = np.eye(4)

    pose_fill = np.eye(4)
    pose_fill[:3, 3] = np.array([0.35, 0.20, 0.60])

    pose_back = np.eye(4)
    pose_back[:3, 3] = np.array([-0.30, -0.20, 0.50])

    scene.add(light_main, pose=pose_main)
    scene.add(light_fill, pose=pose_fill)
    scene.add(light_back, pose=pose_back)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=w,
        viewport_height=h,
    )

    color_rgba, depth = renderer.render(
        scene,
        flags=pyrender.RenderFlags.RGBA,
    )
    renderer.delete()

    rendered_bgr = cv2.cvtColor(color_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    # Use rendered alpha/depth only as a visibility mask.
    # No alpha blending with the original image.
    visible_mask = color_rgba[:, :, 3] > 0

    out = img.copy()
    out[visible_mask] = rendered_bgr[visible_mask]

    cv2.imwrite(out_path, out)
    return out_path