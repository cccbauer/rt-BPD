import nibabel as nib
import numpy as np
import sys
import os

def check_isometric(image_path, tolerance=1e-6):
    """
    Check if an fMRI image has isometric voxels.
    
    Parameters
    ----------
    image_path : str
        Path to the NIfTI image file
    tolerance : float, optional
        Tolerance for floating point comparison of voxel dimensions
        Default is 1e-6
        
    Returns
    -------
    bool
        True if voxels are isometric (equal dimensions), False otherwise
    dict
        Dictionary containing voxel dimensions and additional information
    """
    
    # Load the image
    try:
        img = nib.load(image_path)
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
    
    # Get voxel dimensions (pixdim)
    voxel_dims = img.header.get_zooms()[:3]  # Get first 3 dimensions (spatial)
    
    # Check if all dimensions are equal within tolerance
    is_isometric = np.allclose(voxel_dims, [voxel_dims[0]] * 3, rtol=tolerance)
    
    # Prepare detailed information
    info = {
        'voxel_dims': voxel_dims,
        'resolution': voxel_dims[0] if is_isometric else None,
        'max_difference': max(voxel_dims) - min(voxel_dims),
        'dimensions': img.shape[:3],
        'units': img.header.get_xyzt_units()[0]  # Spatial units
    }
    
    return is_isometric, info

def main():
    # Check if image path is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_nifti_image>")
        sys.exit(1)
    
    # Get image path from command line argument
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist")
        sys.exit(1)
        
    try:
        # Check if image is isometric
        is_isometric, info = check_isometric(image_path)
        print(is_isometric)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
