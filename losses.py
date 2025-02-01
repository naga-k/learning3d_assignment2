import pytorch3d.loss
import torch
import pytorch3d


# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	loss = torch.nn.functional.binary_cross_entropy_with_logits(
        voxel_src,
        voxel_tgt,
        reduction='mean'
    )
	return loss

def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_src: b x n_points x 3
    
    # Get distances
    s1_dist_s2 = pytorch3d.ops.knn_points(point_cloud_src, point_cloud_tgt).dists
    s2_dist_s1 = pytorch3d.ops.knn_points(point_cloud_tgt, point_cloud_src).dists
    
    # Add small epsilon and clip values
    eps = 1e-8
    s1_dist_s2 = torch.clamp(s1_dist_s2, min=eps, max=1e4)
    s2_dist_s1 = torch.clamp(s2_dist_s1, min=eps, max=1e4)
    
    # Calculate loss with numerical stability
    loss_chamfer = torch.mean(torch.sqrt(s1_dist_s2 + eps)) + torch.mean(torch.sqrt(s2_dist_s1 + eps))
    
    # Debug check for NaN
    if torch.isnan(loss_chamfer):
        print("Warning: NaN detected!")
        print("s1_dist_s2 stats:", torch.min(s1_dist_s2), torch.max(s1_dist_s2))
        print("s2_dist_s1 stats:", torch.min(s2_dist_s1), torch.max(s2_dist_s1))
    
    return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
    return pytorch3d.loss.mesh_laplacian_smoothing(mesh_src)