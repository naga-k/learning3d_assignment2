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