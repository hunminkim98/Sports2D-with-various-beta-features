from pathlib import Path


def build_ball_blender_helper_script(output_prefix, marker_name='ball', sphere_radius_m=0.12):
    '''
    Build a Blender helper script that turns an imported ball marker into a sphere mesh.
    '''
    marker_name = str(marker_name or 'ball')
    output_prefix = str(output_prefix or 'Sports2D_output')
    sphere_radius_m = float(sphere_radius_m)
    example_trc_name = f'{output_prefix}_m_person00.trc'
    return f'''# Sports2D Blender helper
"""
Create a mesh sphere that follows the imported "{marker_name}" marker.

Usage:
1. Import a Sports2D TRC that contains the "{marker_name}" marker
   (for example "{example_trc_name}" when meter export is enabled).
2. In Blender, select the imported "{marker_name}" marker object.
3. Run this script from Blender's Text Editor.
4. Adjust BALL_RADIUS_M below if the mesh size should change.
"""

import bpy

BALL_MARKER_NAME = {marker_name!r}
BALL_RADIUS_M = {sphere_radius_m:.4f}
BALL_OBJECT_NAME = "Sports2D_Ball"
BALL_MATERIAL_NAME = "Sports2D_Ball_Material"


def _find_ball_target():
    active = bpy.context.active_object
    if active is not None and BALL_MARKER_NAME.lower() in active.name.lower():
        return active

    for obj in bpy.context.selected_objects:
        if BALL_MARKER_NAME.lower() in obj.name.lower():
            return obj

    matches = [obj for obj in bpy.data.objects if BALL_MARKER_NAME.lower() in obj.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple objects matched the ball marker. "
            "Select the imported ball marker and rerun the script."
        )
    raise RuntimeError(
        "Could not find an imported ball marker. Import the Sports2D TRC, "
        "select the ball marker object, and rerun the script."
    )


def _ensure_ball_material():
    material = bpy.data.materials.get(BALL_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(name=BALL_MATERIAL_NAME)
    material.diffuse_color = (1.0, 0.45, 0.05, 1.0)
    return material


def _ensure_ball_object(target):
    existing = bpy.data.objects.get(BALL_OBJECT_NAME)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=BALL_RADIUS_M,
        location=target.matrix_world.translation,
        segments=32,
        ring_count=16,
    )
    ball_object = bpy.context.active_object
    ball_object.name = BALL_OBJECT_NAME
    return ball_object


def _set_ball_look(ball_object):
    if ball_object.type != "MESH":
        return
    for polygon in ball_object.data.polygons:
        polygon.use_smooth = True
    material = _ensure_ball_material()
    ball_object.data.materials.clear()
    ball_object.data.materials.append(material)


def _ensure_copy_location(ball_object, target):
    constraint = ball_object.constraints.get("Sports2D Ball Follow")
    if constraint is None or constraint.type != "COPY_LOCATION":
        if constraint is not None:
            ball_object.constraints.remove(constraint)
        constraint = ball_object.constraints.new(type="COPY_LOCATION")
        constraint.name = "Sports2D Ball Follow"
    constraint.target = target
    return constraint


def main():
    target = _find_ball_target()
    ball_object = _ensure_ball_object(target)
    _set_ball_look(ball_object)
    _ensure_copy_location(ball_object, target)
    ball_object.hide_viewport = False
    ball_object.hide_render = False
    ball_object.location = target.matrix_world.translation
    try:
        target.hide_set(True)
    except AttributeError:
        target.hide_viewport = True
    target.hide_render = True
    print(
        f"Created {{ball_object.name}} following {{target.name}}. "
        f"Edit BALL_RADIUS_M in this script if the mesh size should change."
    )


main()
'''


def write_ball_blender_helper(output_dir, output_prefix, marker_name='ball', sphere_radius_m=0.12):
    '''
    Write a Blender helper script beside the TRC outputs for quick ball mesh creation.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f'{output_prefix}_ball_mesh_blender.py'
    script_path.write_text(
        build_ball_blender_helper_script(
            output_prefix,
            marker_name=marker_name,
            sphere_radius_m=sphere_radius_m,
        ),
        encoding='utf-8',
    )
    return script_path
