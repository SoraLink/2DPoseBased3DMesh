def render_cut_mesh_overlay(
    image_path: str,
    mesh,
    pred_cam: dict,
    out_path: str,
    color=(0.30, 0.75, 0.95),
    alpha=0.82,
    edge_color=(40, 95, 140),   # BGR for OpenCV edge drawing
    edge_alpha=0.45,
    edge_width=1,
):
    """
    High-quality renderer for cut / residual mesh.
    Compared with a plain translucent mask, this version adds:
      1. stronger lighting/shading
      2. a light wireframe overlay
    so the result looks like a real mesh instead of a flat mask.
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

    # =========================================================
    # 1. Render solid mesh with proper shading
    # =========================================================
    # OpenCV camera coordinates -> OpenGL coordinates
    vertices_gl = vertices_cv.copy()
    vertices_gl[:, 1] *= -1.0
    vertices_gl[:, 2] *= -1.0

    tri_mesh = trimesh.Trimesh(
        vertices=vertices_gl,
        faces=faces,
        process=False,
    )

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.05,
        roughnessFactor=0.70,
        alphaMode="BLEND",
        baseColorFactor=(color[0], color[1], color[2], alpha),
    )

    render_mesh = pyrender.Mesh.from_trimesh(
        tri_mesh,
        material=material,
        smooth=True,
    )

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.22, 0.22, 0.22],
    )
    scene.add(render_mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    scene.add(camera, pose=np.eye(4))

    # multi-light setup for better 3D feeling
    light_main = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.8)
    light_back = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)

    pose_main = np.eye(4)
    pose_fill = np.eye(4)
    pose_back = np.eye(4)

    pose_fill[:3, 3] = np.array([0.35, 0.20, 0.60])
    pose_back[:3, 3] = np.array([-0.30, -0.20, 0.50])

    scene.add(light_main, pose=pose_main)
    scene.add(light_fill, pose=pose_fill)
    scene.add(light_back, pose=pose_back)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=w,
        viewport_height=h,
    )

    color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    mesh_rgb = cv2.cvtColor(color_rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    mesh_alpha = color_rgba[:, :, 3:4].astype(np.float32) / 255.0

    base = mesh_rgb * mesh_alpha + img.astype(np.float32) * (1.0 - mesh_alpha)
    base = np.clip(base, 0, 255).astype(np.uint8)

    # =========================================================
    # 2. Add projected wireframe to emphasize "mesh feeling"
    # =========================================================
    verts = vertices_cv
    z = verts[:, 2].copy()

    # avoid division by zero
    valid_z = z > 1e-6

    proj = np.zeros((verts.shape[0], 2), dtype=np.float32)
    proj[valid_z, 0] = fx * (verts[valid_z, 0] / z[valid_z]) + cx
    proj[valid_z, 1] = fy * (verts[valid_z, 1] / z[valid_z]) + cy

    # collect unique edges from faces
    edges = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))

    wire = base.copy()

    for i, j in edges:
        if not (valid_z[i] and valid_z[j]):
            continue

        x1, y1 = proj[i]
        x2, y2 = proj[j]

        # skip obviously off-image edges
        if ((x1 < -50 and x2 < -50) or (x1 > w + 50 and x2 > w + 50) or
            (y1 < -50 and y2 < -50) or (y1 > h + 50 and y2 > h + 50)):
            continue

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        cv2.line(wire, p1, p2, edge_color, edge_width, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(wire, edge_alpha, base, 1.0 - edge_alpha, 0)

    cv2.imwrite(out_path, out)
    return out_path