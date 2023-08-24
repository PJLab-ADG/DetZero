import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def get_overlapping_voxel_indices(point_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        point_coords: (N, 3)
        downsample_times: (int)
        voxel_size: [x_size, y_size, z_size]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    This assumes that the (pc_range[3:6] - pc_range[0:3]) / voxel_size is an integer. If the pc_range
    is not entirely divisible by the voxel_size, points on the far extremes may be excluded.
    E.g.
    1)
        pc_range = [0, 0, 0, 1, 1, 1], voxel_size = [0.5, 0.5, 0.5]
        The point [0,0,0.7] will be considered inside and will return a value of (0, 0, 1)
    2)
        pc_range = [0, 0, 0, 1, 1, 1], voxel_size = [0.6, 0.6, 0.6]
        The point [0,0,0.7] will be considered outside and will return a value of (-1, -1, -1)
    Returns: voxel_indices (xyz). If the point cloud is outside the range of the voxels,
             it returns a value of (-1, -1, -1)
    """
    assert point_coords.shape[1] == 3
    voxel_size = torch.tensor(voxel_size, device=point_coords.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range, device=point_coords.device).float()

    voxel_indices = ((point_coords - pc_range[0:3]) / voxel_size)

    # Calculate number of voxels in each dimension
    grid_size = ((pc_range[3:6] - pc_range[0:3]) / voxel_size).long()

    # Check which points are in and which points are outside the point cloud range and set to -1
    points_out_of_range = ((voxel_indices < 0) | (voxel_indices >= grid_size)).sum(dim=-1) > 0
    voxel_indices[points_out_of_range] = -1

    return voxel_indices.long() # (xyz)


def get_voxel_indices_to_voxel_list_index(x_conv):
    """
    Args:
        x_conv: (SparseConvTensor)
    Returns:
        x_conv_hash_table: (B, X, Y, Z) Dense representation of sparse voxel indices
    """
    x_conv_indices = x_conv.indices
    # Note that we need to offset the values by 1 since the dense representation has "0" to indicate an empty location
    x_conv_values = torch.arange(1, x_conv_indices.shape[0]+1, device=x_conv_indices.device)
    x_conv_shape = [x_conv.batch_size] + list(x_conv.spatial_shape)

    # TODO: Need to convert to_dense representation. Can we use rule table instead? Can try scatter_nd in spconv too
    x_conv_hash_table = torch.sparse_coo_tensor(x_conv_indices.T, x_conv_values, x_conv_shape, device=x_conv_indices.device).to_dense()
    return x_conv_hash_table


def get_nonempty_voxel_feature_indices(voxel_indices, x_conv):
    """
    Args:
        voxel_indices: (N, 4) [bxyz]
        x_conv: (SparseConvTensor)
    Returns:
        overlapping_voxel_feature_indices_nonempty: (N', 4)
        overlapping_voxel_feature_nonempty_mask: (N)
    """
    x_conv_hash_table = get_voxel_indices_to_voxel_list_index(x_conv)

    # Get corresponding voxel feature indices
    overlapping_voxel_feature_indices = torch.zeros(voxel_indices.shape[0], device=voxel_indices.device, dtype=torch.int64)
    overlapping_voxel_feature_indices = x_conv_hash_table[voxel_indices[:,0], voxel_indices[:,1],
                                                          voxel_indices[:,2], voxel_indices[:,3]]
    # Remove empty voxels features
    overlapping_voxel_feature_nonempty_mask = overlapping_voxel_feature_indices != 0
    overlapping_voxel_feature_indices_nonempty = overlapping_voxel_feature_indices[overlapping_voxel_feature_nonempty_mask] - 1
    return overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask


def get_centroid_per_voxel(points, voxel_idxs, num_points_in_voxel=None):
    """
    Args:
        points: (N, 4 + (f)) [bxyz + (f)]
        voxel_idxs: (N, 4) [bxyz]
        num_points_in_voxel: (N)
    Returns:
        centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel
        centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids
        labels_count: (N') Number of points in each voxel
    """
    assert points.shape[0] == voxel_idxs.shape[0]

    centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(dim=0, return_inverse=True, return_counts=True)
    unique_idxs = unique_idxs.view(unique_idxs.size(0), 1).expand(-1, points.size(-1))

    # Scatter add points based on unique voxel idxs
    if num_points_in_voxel is not None:
        centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device, dtype=torch.float).scatter_add_(0, unique_idxs, points * num_points_in_voxel.unsqueeze(-1))
        num_points_in_centroids = torch.zeros((centroid_voxel_idxs.shape[0]), device=points.device, dtype=torch.int64).scatter_add_(0, unique_idxs[:,0], num_points_in_voxel)
        centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
    else:
        centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device, dtype=torch.float).scatter_add_(0, unique_idxs, points)
        centroids = centroids / labels_count.float().unsqueeze(-1)

    return centroids, centroid_voxel_idxs, labels_count


def get_centroids_per_voxel_layer(points, feature_locations, multi_scale_3d_strides, voxel_size, point_cloud_range):
    """
    Group points that lie within the same voxel together and average their xyz location.
    Details can be found here: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
    Args:
        points: (N, 4 + (f)) [bxyz + (f)]
        feature_locations: [str] (Order matters! Needs to be xconv1 -> xconv4)
        multi_scale_3d_strides: (dict) Map feature_locations to stride
        voxel_size: [x_size, y_size, z_size]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        centroids_all: (dict) Centroids for each feature_locations
        centroid_voxel_idxs_all: (dict) Centroid voxel ids for each feature_locations
    """
    assert len(points.shape) == 2
    assert len(feature_locations) > 0

    centroids_all = {}
    centroid_voxel_idxs_all = {}

    # Take first layer feature locations
    feature_location_first = feature_locations[0]
    downsample_factor_first = multi_scale_3d_strides[feature_location_first]
    # Calculate centroids
    voxel_idxs = get_overlapping_voxel_indices(points[:, 1:4],
                                               downsample_times=downsample_factor_first,
                                               voxel_size=voxel_size,
                                               point_cloud_range=point_cloud_range)
    # Add batch_idx
    voxel_idxs = torch.cat((points[:,0:1].long(), voxel_idxs), dim=-1)

    # Filter out points that are outside the valid point cloud range (invalid indices have -1)
    voxel_idxs_valid_mask = (voxel_idxs != -1).all(-1)
    voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]
    # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer
    voxel_idxs_valid = voxel_idxs_valid[:, [0,3,2,1]]
    points_valid = points[voxel_idxs_valid_mask]

    centroids_first, centroid_voxel_idxs_first, num_points_in_centroids_first = get_centroid_per_voxel(points_valid, voxel_idxs_valid)
    centroids_all[feature_location_first] = centroids_first
    centroid_voxel_idxs_all[feature_location_first] = centroid_voxel_idxs_first

    for feature_location in feature_locations[1:]:
        grid_scaling = int(multi_scale_3d_strides[feature_location] / downsample_factor_first)
        voxel_idxs = centroid_voxel_idxs_first.clone()
        # voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
        voxel_idxs[:, 1:] = torch.div(centroid_voxel_idxs_first[:, 1:], grid_scaling, rounding_mode='trunc')
        centroids, centroid_voxel_idxs, _ = get_centroid_per_voxel(centroids_first, voxel_idxs, num_points_in_centroids_first)
        centroids_all[feature_location] = centroids
        centroid_voxel_idxs_all[feature_location] = centroid_voxel_idxs

    return centroids_all, centroid_voxel_idxs_all


def get_centroids_point_per_voxel_layer(points, feature_locations, multi_scale_3d_strides, voxel_size, point_cloud_range):
    """
    Group points that lie within the same voxel together and average their xyz location.
    Details can be found here: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
    Args:
        points: (N, 4 + (f)) [bxyz + (f)]
        feature_locations: [str] (Order matters! Needs to be xconv1 -> xconv4)
        multi_scale_3d_strides: (dict) Map feature_locations to stride
        voxel_size: [x_size, y_size, z_size]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        centroids_all: (dict) Centroids for each feature_locations
        centroid_voxel_idxs_all: (dict) Centroid voxel ids for each feature_locations
    """
    assert len(points.shape) == 2
    assert len(feature_locations) > 0

    centroids_all = {}
    centroid_voxel_idxs_all = {}

    # Take first layer feature locations
    feature_location_first = feature_locations[0]
    downsample_factor_first = multi_scale_3d_strides[feature_location_first]
    # Calculate centroids
    voxel_idxs = get_overlapping_voxel_indices(points[:, 1:4],
                                               downsample_times=downsample_factor_first,
                                               voxel_size=voxel_size,
                                               point_cloud_range=point_cloud_range)
    # Add batch_idx
    voxel_idxs = torch.cat((points[:,0:1].long(), voxel_idxs), dim=-1)

    # Filter out points that are outside the valid point cloud range (invalid indices have -1)
    voxel_idxs_valid_mask = (voxel_idxs != -1).all(-1)
    voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]
    # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer
    voxel_idxs_valid = voxel_idxs_valid[:, [0,3,2,1]]
    points_valid = points[voxel_idxs_valid_mask]

    centroids_first, centroid_voxel_idxs_first, num_points_in_centroids_first = get_centroid_per_voxel(points_valid, voxel_idxs_valid)
    centroids_all[feature_location_first] = centroids_first
    centroid_voxel_idxs_all[feature_location_first] = centroid_voxel_idxs_first

    for feature_location in feature_locations[1:]:
        grid_scaling = int(multi_scale_3d_strides[feature_location] / downsample_factor_first)
        voxel_idxs = centroid_voxel_idxs_first.clone()
        # voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
        voxel_idxs[:, 1:] = torch.div(centroid_voxel_idxs_first[:, 1:], grid_scaling, rounding_mode='trunc')
        centroids, centroid_voxel_idxs, _ = get_centroid_per_voxel(centroids_first, voxel_idxs, num_points_in_centroids_first)
        centroids_all[feature_location] = centroids
        centroid_voxel_idxs_all[feature_location] = centroid_voxel_idxs

    return centroids_all, centroid_voxel_idxs_all
