import numpy as np
import trimesh
from pathlib import Path

def cylinder_between_points(start, end, radius, sections=64):
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
    support_sweep=0.28,
    body_bulge=0.06,
    body_twist=0.0,
    lobe_amplitude=0.0,
    lobe_count=3.0,
    opening_width=76.0,
    sections=96,
    levels=30,  
):
    vertices = []
    faces = []

    half_opening = np.radians(opening_width / 2)
    angles = np.linspace(
        half_opening,
        2 * np.pi - half_opening,
        sections,
    )
    
    for level in range(levels):
        t = level / (levels - 1)

        z = t * height

        base_radius = (
    bottom_radius * (1 - t)
    + top_radius * t
)

        base_radius += body_bulge * np.sin(t * np.pi)

        center_x = -support_sweep * (t ** 1.35)
        opening_shift = 0.30 * t
        twist = body_twist * t

        for angle in angles:
            warped_angle = angle + twist

            local_radius = base_radius * (
                1.0
                + lobe_amplitude
                * np.sin(lobe_count * warped_angle + 2 * np.pi * t)
            )

            x = center_x + local_radius * np.cos(warped_angle)
            y = local_radius * np.sin(warped_angle)

            z_actual = z

            if np.cos(warped_angle) > 0.45:
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
    
    top_center_x = -support_sweep

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
    ball_offset=-0.90, 
    support_sweep=0.28, 
    body_height=2.85, 
    body_bottom_radius=0.55, 
    body_top_radius=0.86, 
    lower_base_radius=1.25, 
    lower_base_height=0.18, 
    upper_base_radius=1.02, 
    upper_base_height=0.16,
    body_bulge=0.06,
    body_twist=0.0,
    lobe_amplitude=0.0,
    lobe_count=3.0,
    opening_width=76.0,
):
    lower_base = trimesh.creation.cylinder(radius=lower_base_radius, height=lower_base_height, sections=96)
    lower_base.apply_translation([0, 0, lower_base_height / 2])

    upper_base = trimesh.creation.cylinder(radius=upper_base_radius, height=upper_base_height, sections=96)
    upper_base.apply_translation([0, 0, lower_base_height + upper_base_height / 2])

    base_top = lower_base_height + upper_base_height

    body = create_trophy_body(
        height=body_height,
        bottom_radius=body_bottom_radius,
        top_radius=body_top_radius,
        support_sweep=support_sweep,
        body_bulge=body_bulge,
        body_twist=body_twist,
        lobe_amplitude=lobe_amplitude,
        lobe_count=lobe_count,
        opening_width=opening_width,
        sections=96,
        levels=35,
    )
    body.apply_translation([0, 0, base_top])

    rim_radius = body_top_radius
    rim_height = 0.07
    rim = trimesh.creation.cylinder(radius=rim_radius, height=rim_height, sections=96)
    rim_x = -support_sweep
    rim_z = base_top + body_height + rim_height / 2
    rim.apply_translation([rim_x, 0, rim_z])

    neck_height = 0.20
    neck_radius = 0.17
    neck = trimesh.creation.cylinder(radius=neck_radius, height=neck_height, sections=64)
    neck_z = base_top + body_height + rim_height + neck_height / 2
    neck.apply_translation([ball_offset, 0, neck_z])

    ball_z = base_top + body_height + rim_height + neck_height + ball_radius * 0.92
    ball = trimesh.creation.icosphere(subdivisions=4, radius=ball_radius)
    ball.apply_translation([ball_offset, 0, ball_z])

    trophy = trimesh.util.concatenate([lower_base, upper_base, body, rim, neck, ball])
    return trophy

DEFAULT_PARAMS = {
    "ball_radius": 0.88,
    "ball_offset": -0.90,
    "support_sweep": 0.28,
    "body_height": 2.85,
    "body_bottom_radius": 0.55,
    "body_top_radius": 0.86,
    "lower_base_radius": 1.25,
    "lower_base_height": 0.18,
    "upper_base_radius": 1.02,
    "upper_base_height": 0.16,
}


PARAMETER_SWEEPS = {
    "ball_radius": [0.78, 0.82, 0.86, 0.90, 0.94, 0.98],
    "ball_offset": [-0.92, -0.912, -0.904, -0.896, -0.888, -0.88],
    "support_sweep": [0.22, 0.244, 0.268, 0.292, 0.316, 0.34],
    "body_height": [2.65, 2.73, 2.81, 2.89, 2.97, 3.05],
    "body_bottom_radius": [0.48, 0.508, 0.536, 0.564, 0.592, 0.62],
    "body_top_radius": [0.78, 0.816, 0.852, 0.888, 0.924, 0.96],
    "lower_base_radius": [1.10, 1.16, 1.22, 1.28, 1.34, 1.40],
    "lower_base_height": [0.14, 0.156, 0.172, 0.188, 0.204, 0.22],
    "upper_base_radius": [0.90, 0.944, 0.988, 1.032, 1.076, 1.12],
    "upper_base_height": [0.12, 0.136, 0.152, 0.168, 0.184, 0.20],
}


def export_parameter_sweeps():
    root = Path("outputs/parameter_tests")

    for parameter_name, values in PARAMETER_SWEEPS.items():
        output_dir = root / parameter_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, value in enumerate(values):
            params = DEFAULT_PARAMS.copy()
            params[parameter_name] = value

            trophy = generate_trophy(**params)

            output_path = (
                output_dir
                / f"{parameter_name}_{i:02d}_{value:.4f}.obj"
            )

            trophy.export(output_path)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    export_parameter_sweeps()