import numpy as np
import trimesh
from pathlib import Path

def cylinder_between_points(
    start,
    end,
    radius,
    sections=64,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length = np.linalg.norm(direction)

    if length == 0:
        raise ValueError("Start and end cannot be the same.")

    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=sections,
    )

    transform = trimesh.geometry.align_vectors(
        [0, 0, 1],
        direction / length,
    )

    mesh.apply_transform(transform)
    mesh.apply_translation((start + end) / 2)

    return mesh

def create_trophy_body(
    height=3.0,
    bottom_radius=0.58,
    top_radius=0.92,
    sections=96,
    levels=30,
):
    """
    Same trophy shape as before, but:
    - no hollow interior
    - straight/closed corners
    - keeps the existing curved/slanted outer shape
    """

    vertices = []
    faces = []

    start_angle = np.radians(38)
    end_angle = np.radians(322)

    angles = np.linspace(
        start_angle,
        end_angle,
        sections
    )
    
    for level in range(levels):

        t = level / (levels - 1)

        z = t * height

        radius = (
            bottom_radius * (1 - t)
            + top_radius * t
        )

        radius += 0.06 * np.sin(t * np.pi)

        center_x = -0.28 * t

        opening_shift = 0.30 * t

        for angle in angles:

            x = center_x + radius * np.cos(angle)
            y = radius * np.sin(angle)

            z_actual = z

            if np.cos(angle) > 0.45:
                z_actual -= opening_shift

            vertices.append([
                x,
                y,
                z_actual
            ])

    for level in range(levels - 1):

        current = level * sections
        nxt = (level + 1) * sections

        for i in range(sections - 1):

            a = current + i
            b = current + i + 1

            c = nxt + i
            d = nxt + i + 1

            faces.append([a, b, d])
            faces.append([a, d, c])

    for level in range(levels - 1):

        current = level * sections
        nxt = (level + 1) * sections

        a = current
        b = nxt

        c = current + sections - 1
        d = nxt + sections - 1

        faces.append([a, b, d])
        faces.append([a, d, c])

    bottom_center_index = len(vertices)

    vertices.append([
        0,
        0,
        0
    ])

    for i in range(sections - 1):

        faces.append([
            bottom_center_index,
            i + 1,
            i
        ])

    faces.append([
        bottom_center_index,
        0,
        sections - 1
    ])
    
    top_center_x = -0.28

    top_center_index = len(vertices)

    vertices.append([
        top_center_x,
        0,
        height
    ])

    top_start = (levels - 1) * sections

    for i in range(sections - 1):

        faces.append([
            top_center_index,
            top_start + i,
            top_start + i + 1
        ])

    faces.append([
        top_center_index,
        top_start + sections - 1,
        top_start
    ])

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        process=True,
    )

    return mesh

def generate_trophy(
    ball_radius=0.88,
    body_height=2.85,
    body_bottom_radius=0.55,
    body_top_radius=0.86,
    lower_base_radius=1.25,
    lower_base_height=0.18,
    upper_base_radius=1.02,
    upper_base_height=0.16,
):

    lower_base = trimesh.creation.cylinder(
        radius=lower_base_radius,
        height=lower_base_height,
        sections=96,
    )

    lower_base.apply_translation([
        0,
        0,
        lower_base_height / 2
    ])

    upper_base = trimesh.creation.cylinder(
        radius=upper_base_radius,
        height=upper_base_height,
        sections=96,
    )

    upper_base.apply_translation([
        0,
        0,
        lower_base_height + upper_base_height / 2
    ])

    base_top = (
        lower_base_height
        + upper_base_height
    )

    body = create_trophy_body(
        height=body_height,
        bottom_radius=body_bottom_radius,
        top_radius=body_top_radius,
        sections=96,
        levels=35,
    )

    body.apply_translation([
        0,
        0,
        base_top
    ])

    rim_radius = body_top_radius
    rim_height = 0.07

    rim = trimesh.creation.cylinder(
        radius=rim_radius,
        height=rim_height,
        sections=96,
    )

    rim_x = -0.28

    rim_z = (
        base_top
        + body_height
        + rim_height / 2
    )

    rim.apply_translation([
        rim_x,
        0,
        rim_z
    ])

    neck_height = 0.20
    neck_radius = 0.17

    neck = trimesh.creation.cylinder(
        radius=neck_radius,
        height=neck_height,
        sections=64,
    )

    neck_z = (
        base_top
        + body_height
        + rim_height
        + neck_height / 2
    )

    neck.apply_translation([
        0.90,
        0,
        neck_z
    ])

    ball_z = (
        base_top
        + body_height
        + rim_height
        + neck_height
        + ball_radius * 0.92
    )

    ball = trimesh.creation.icosphere(
        subdivisions=4,
        radius=ball_radius,
    )

    ball.apply_translation([
        0.90,
        0,
        ball_z
    ])

    trophy = trimesh.util.concatenate([
        lower_base,
        upper_base,
        body,
        rim,
        neck,
        ball,
    ])

    return trophy

if __name__ == "__main__":
    ball_radius_values = [
        0.70,
        0.80,
        0.88,  # original
        1.00,
        1.10,
        1.20,
    ]

    body_height_values = [
        2.30,
        2.55,
        2.70,
        2.85,  # original
        3.00,
        3.20,
        3.40,
    ]

    body_bottom_radius_values = [
        0.40,
        0.45,
        0.50,
        0.55,  # original
        0.60,
        0.65,
        0.70,
    ]

    body_top_radius_values = [
        0.65,
        0.75,
        0.80,
        0.86,  # original
        0.95,
        1.05,
        1.15,
    ]

    lower_base_radius_values = [
        0.95,
        1.10,
        1.20,
        1.25,  # original
        1.35,
        1.50,
        1.65,
    ]

    lower_base_height_values = [
        0.10,
        0.14,
        0.18,  # original
        0.22,
        0.26,
        0.30,
    ]

    upper_base_radius_values = [
        0.80,
        0.90,
        1.00,
        1.02,  # original
        1.10,
        1.20,
        1.30,
    ]

    upper_base_height_values = [
        0.08,
        0.12,
        0.16,  # original
        0.20,
        0.24,
        0.28,
    ]

    for i, radius in enumerate(ball_radius_values):        
        output_dir = Path("outputs/parameter_tests/ball_radius")
        output_dir.mkdir(parents=True, exist_ok=True)

        trophy = generate_trophy(
            ball_radius=radius
        )

        output_path = (output_dir / f"ball_radius_{i}_{radius:.2f}.obj")

        trophy.export(output_path)

        print(f"Saved: {output_path}")
    
    for i, bheight in enumerate(body_height_values):        
        output_dir = Path("outputs/parameter_tests/body_height")
        output_dir.mkdir(parents=True, exist_ok=True)

        trophy = generate_trophy(
            body_height=bheight
        )

        output_path = (output_dir / f"body_height_{i}_{bheight:.2f}.obj")

        trophy.export(output_path)

        print(f"Saved: {output_path}")
    
    for i, bbottom_radius in enumerate(body_bottom_radius_values):        
        output_dir = Path("outputs/parameter_tests/body_bottom_radius")
        output_dir.mkdir(parents=True, exist_ok=True)

        trophy = generate_trophy(body_bottom_radius=bbottom_radius)

        output_path = (output_dir / f"body_bottom_radius_{i}_{bbottom_radius:.2f}.obj")

        trophy.export(output_path)

        print(f"Saved: {output_path}")
    
    for i, btop_radius in enumerate(body_top_radius_values):        
            output_dir = Path("outputs/parameter_tests/body_top_radius")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            trophy = generate_trophy(body_top_radius=btop_radius)
    
            output_path = (output_dir / f"body_top_radius_{i}_{btop_radius:.2f}.obj")
    
            trophy.export(output_path)
    
            print(f"Saved: {output_path}")
    
    for i, lbottom_radius in enumerate(lower_base_radius_values):        
            output_dir = Path("outputs/parameter_tests/lower_base_radius")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            trophy = generate_trophy(lower_base_radius=lbottom_radius)
    
            output_path = (output_dir / f"lower_base_radius_{i}_{lbottom_radius:.2f}.obj")
    
            trophy.export(output_path)
    
            print(f"Saved: {output_path}")
    
    for i, lbottom_height in enumerate(lower_base_height_values):        
            output_dir = Path("outputs/parameter_tests/lower_base_height")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            trophy = generate_trophy(lower_base_height=lbottom_height)
    
            output_path = (output_dir / f"lower_base_height_{i}_{lbottom_height:.2f}.obj")
    
            trophy.export(output_path)
    
            print(f"Saved: {output_path}")
    
    for i, ubase_radius in enumerate(upper_base_radius_values):        
            output_dir = Path("outputs/parameter_tests/upper_base_radius")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            trophy = generate_trophy(upper_base_radius=ubase_radius)
    
            output_path = (output_dir / f"upper_base_radius_{i}_{ubase_radius:.2f}.obj")
    
            trophy.export(output_path)
    
            print(f"Saved: {output_path}")
    
    for i, ubase_height in enumerate(upper_base_height_values):        
            output_dir = Path("outputs/parameter_tests/upper_base_height")
            output_dir.mkdir(parents=True, exist_ok=True)
    
            trophy = generate_trophy(upper_base_height=ubase_height)
    
            output_path = (output_dir / f"upper_base_height_{i}_{ubase_height:.2f}.obj")
    
            trophy.export(output_path)
    
            print(f"Saved: {output_path}")