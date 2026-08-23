import numpy as np
import trimesh
from pathlib import Path


# ------------------------------------------------------------
# CYLINDER BETWEEN TWO POINTS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# CREATE THE MAIN OPEN TROPHY BODY
# ------------------------------------------------------------

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

    # Keep the same partial circumference / overall shape
    start_angle = np.radians(38)
    end_angle = np.radians(322)

    angles = np.linspace(
        start_angle,
        end_angle,
        sections
    )

    # --------------------------------------------------------
    # OUTER SURFACE
    # --------------------------------------------------------

    for level in range(levels):

        t = level / (levels - 1)

        z = t * height

        # KEEP your original widening shape
        radius = (
            bottom_radius * (1 - t)
            + top_radius * t
        )

        # KEEP your original slight curve
        radius += 0.06 * np.sin(t * np.pi)

        # KEEP your original sideways lean
        center_x = -0.28 * t

        # KEEP your original top shape
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

    # --------------------------------------------------------
    # CONNECT OUTER SURFACE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FILL THE OPEN SIDE WITH A STRAIGHT WALL
    # --------------------------------------------------------

    for level in range(levels - 1):

        current = level * sections
        nxt = (level + 1) * sections

        # One edge of opening
        a = current
        b = nxt

        # Other edge of opening
        c = current + sections - 1
        d = nxt + sections - 1

        # Two triangles make a flat wall
        faces.append([a, b, d])
        faces.append([a, d, c])

    # --------------------------------------------------------
    # FILL BOTTOM
    # --------------------------------------------------------

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

    # Close across missing section
    faces.append([
        bottom_center_index,
        0,
        sections - 1
    ])

    # --------------------------------------------------------
    # FILL TOP
    # --------------------------------------------------------

    t = 1.0

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

    # Close across missing section
    faces.append([
        top_center_index,
        top_start + sections - 1,
        top_start
    ])

    # --------------------------------------------------------
    # CREATE MESH
    # --------------------------------------------------------

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        process=True,
    )

    return mesh


# ------------------------------------------------------------
# COMPLETE TROPHY
# ------------------------------------------------------------

def generate_trophy():

    # ========================================================
    # LOWER BASE
    # ========================================================

    lower_base_radius = 1.25
    lower_base_height = 0.18

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

    # ========================================================
    # UPPER BASE
    # ========================================================

    upper_base_radius = 1.02
    upper_base_height = 0.16

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

    base_top = lower_base_height + upper_base_height

    # ========================================================
    # MAIN BODY
    # ========================================================

    body_height = 2.85

    body = create_trophy_body(
        height=body_height,
        bottom_radius=0.55,
        top_radius=0.86,
        sections=96,
        levels=35,
    )

    body.apply_translation([
        0,
        0,
        base_top
    ])

    # ========================================================
    # TOP RIM
    # ========================================================

    #
    # The real trophy has a thin horizontal lip underneath
    # the basketball.
    #
    
    rim_radius = 0.86
    rim_height = 0.07

    rim = trimesh.creation.cylinder(
        radius=rim_radius,
        height=rim_height,
        sections=96,
    )

    rim_x = -0.28
    
    ball_x = -0.90

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

    # ========================================================
    # SMALL NECK UNDER BALL
    # ========================================================

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
        ball_x,
        0,
        neck_z
    ])

    # ========================================================
    # BASKETBALL
    # ========================================================

    ball_radius = 0.88

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
        ball_x,
        0,
        ball_z
    ])

    # ========================================================
    # COMBINE EVERYTHING
    # ========================================================

    trophy = trimesh.util.concatenate([
        lower_base,
        upper_base,
        body,
        rim,
        neck,
        ball,
    ])

    return trophy


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

if __name__ == "__main__":

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    trophy = generate_trophy()

    output_path = (
        output_dir
        / "larry_obrien_trophy.obj"
    )

    trophy.export(output_path)

    print(
        f"Saved trophy to: {output_path}"
    )

    trophy.show()