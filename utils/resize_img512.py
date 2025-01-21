import os
import sys
from PIL import Image

def resize_image(input_dir, output_dir, size=(640, 640)):
    # Check if output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(input_dir, filename)
            with Image.open(image_path) as img:
                # Ensure image is 512x512 before resizing
                if img.size == (512, 512):
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    save_path = os.path.join(output_dir, filename)
                    resized_img.save(save_path)
                    print(f"Resized and saved {filename} to {output_dir}")
                else:
                    print(f"Skipping {filename}: Not 512x512")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize_images.py <input_dir> <output_dir>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    resize_image(input_directory, output_directory)
