import os
from PIL import Image

def combine_images_horizontal(image_left, image_right, output_filename):
    """
    Combines two images horizontally and saves the result
    
    Args:
        image_left: First PIL Image
        image_right: Second PIL Image 
        output_filename: Name of output file
    """
    total_width = image_left.width + image_right.width
    max_height = max(image_left.height, image_right.height)
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    new_image.paste(image_left, (0, 0))
    new_image.paste(image_right, (image_left.width, 0))

    # Create output directory if needed
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save combined image
    output_path = os.path.join(output_dir, output_filename)
    new_image.save(output_path)
    return output_path