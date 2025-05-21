import torch
import roma

def rotation_unitquat_to_6dvec(
        tensor_quat: torch.Tensor,
    ) -> torch.Tensor:
    """Convert unit quaternion rotation to 6d vector representation
    Args:
        tensor_quat: Unit quaternion with XYZW convention (..., 4).
            Quaternion are renormalized.
    Return:
        Tensor of 6d representation (flatten two first 
        columns of rotation matrix) (..., 6).
    """
    #Re-normalize the quaternion (...,4)
    tensor_quat = roma.quat_normalize(tensor_quat)
    #Convert to rotation matrix (...,3,3)
    tensor_rotmat = roma.unitquat_to_rotmat(tensor_quat)
    #Select the first two columns (...,3,2)
    tensor_6d = tensor_rotmat[...,0:2]
    #Flatten to 6d vector (...,6)
    return torch.flatten(tensor_6d, -2, -1)

def rotation_6dvec_to_unitquat(
        tensor_6d: torch.Tensor,
    ) -> torch.Tensor:
    """Convert rotation from 6d vector representation to unit quaternion
    with reprojection to quaternion manifold.
    Args:
        tensor_6d: Tensor of 6d representation (flatten two first 
            columns of rotation matrix) (..., 6).
    Return:
        Reprojected tensor unit quaternion with XYZW convention (..., 4).
    """
    #Unflatten the rotation vector (...,3,2)
    tensor_3x2 = torch.unflatten(tensor_6d, -1, (3,2))
    #Reconstruct rotation matrix with reprojection (...,3,3)
    tensor_rotmat = roma.special_gramschmidt(tensor_3x2)
    #Convert to quaternion (...,4)
    tensor_quat = roma.rotmat_to_unitquat(tensor_rotmat)
    #Standardize quaternion to prevent the double cover 
    #by making sure that the last X component is positive
    tensor_quat[tensor_quat[...,3] < 0.0] *= -1.0
    return tensor_quat

def rotation_unitquat_diff_rotvec(
        tensor_quat_src: torch.Tensor,
        tensor_quat_dst: torch.Tensor,
    ) -> torch.Tensor:
    """Compute unit quaternion rotation difference as axis angle vector
    Args:
        tensor_quat_src: Source unit quaternion (renormalized) (...,4), 
            using XYZW convention.
        tensor_quat_dst: Destination unit quaternion (renormalized) (...,4),
            using XYZW convention.
    Returns:
        Delta rotation (left applied) from source to destination 
        as axis angle (...,3) such as q_dst = delta * q_src.
    """
    #Re-normalize quaternions (...,4)
    tensor_quat_src = roma.quat_normalize(tensor_quat_src)
    tensor_quat_dst = roma.quat_normalize(tensor_quat_dst)
    #Diff rotation q_delta = q_dst * q_src.inv() (...,4)
    tensor_quat_diff = roma.quat_product(
        tensor_quat_dst, roma.quat_inverse(tensor_quat_src))
    #Convert to axis angle (...,3)
    return roma.unitquat_to_rotvec(tensor_quat_diff)

def rotation_unitquat_add_rotvec(
        tensor_quat_src: torch.Tensor,
        tensor_rotvec: torch.Tensor,
    ) -> torch.Tensor:
    """Add to given  unit quaternion a rotation as axis angle
    Args:
        tensor_quat_src: Source unit quaternion (renormalized) (...,4),
            using XYZW convention.
        tensor_rotvec: Delta axis angle (...,3) to be left applied to 
            the source quaternion.
    Returns:
        Unit quaternion (...,4) using XYZW convention such as 
            q_dst = delta * q_src.
    """
    #Re-normalize quaternions (...,4)
    tensor_quat_src = roma.quat_normalize(tensor_quat_src)
    #Convert axis angle to quaternion
    tensor_quat_delta = roma.rotvec_to_unitquat(tensor_rotvec)
    #Apply left transformation delta
    return roma.quat_product(tensor_quat_delta, tensor_quat_src)

