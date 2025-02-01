import mcubes
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from PIL import Image, ImageDraw
import numpy as np

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_voxels(
        voxels,
        image_size = 256,
):
    device = voxels.device
    voxels_np = voxels[0].detach().cpu().numpy()
    vertices, faces = mcubes.marching_cubes(voxels_np, 0)
    # vertices = vertices / (voxel_size - 1) * 4 - 2

    # Convert to tensors
    vertices = torch.tensor(vertices).float().to(device)
    faces = torch.tensor(faces.astype(int)).to(device)

    # Create colors based on normalized vertex positions
    points = vertices.unsqueeze(0)  # Add batch dimension
    color = (points - points.min()) / (points.max() - points.min())

    # Create mesh with colored vertices
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=pytorch3d.renderer.TexturesVertex(color),
    ).to(device)

    # Set up renderer
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=100, elev=30, azim=0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    rendered_image = renderer(mesh, cameras=cameras)

    # Convert to numpy and scale to 0-255
    image_np = rendered_image[0, ..., :3].cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL Image and save
    image = Image.fromarray(image_np)
    # image.save(output_file)
    
    return image

def render_points(
        points,
        image_size=256,
        background_color=(1, 1, 1),
):
    
    device = points.device
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    points = points.to(device)
    if len(points.shape) == 3:
        points = points.squeeze(0)  
    color = (points - points.min()) / (points.max() - points.min())

    rotation_transform = pytorch3d.transforms.RotateAxisAngle(angle=0, axis="X", degrees=True).to(device)
    points = rotation_transform.transform_points(points)
    point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[color])

    R, T = pytorch3d.renderer.look_at_view_transform(dist=1, elev=30, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)


    rendered_image = renderer(point_cloud, cameras=cameras)

    # Convert to numpy and scale to 0-255
    image_np = rendered_image[0, ..., :3].detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL Image and save
    image = Image.fromarray(image_np)
    # image.save(output_file)

    return image


def render_mesh(
        mesh,
        image_size=256,
        background_colour=(1,1,1)

):
    device = mesh.device
    if not mesh.textures:
        verts_rgb = torch.ones_like(mesh.verts_padded())[..., :3]  # (1, V, 3)
        mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rendered_image = renderer(mesh, cameras=cameras, lights=lights)
    # Convert to numpy and scale to 0-255
    image_np = rendered_image[0, ..., :3].detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Convert to PIL Image and save
    image = Image.fromarray(image_np)
    # image.save(output_file)

    return image

