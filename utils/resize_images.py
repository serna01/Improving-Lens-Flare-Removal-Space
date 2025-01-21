#python resize_images.py /path/to/input_dir /path/to/output_dir 512 512 1008 752
import os
import sys
import cv2

def resize_images(input_dir, output_dir, original_size, target_size):
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            if img is not None:
                # Check if the image matches the original size
                if img.shape[1] == original_size[0] and img.shape[0] == original_size[1]:
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, resized_img)
                    print(f"Resized and saved {filename} to {output_dir}")
                else:
                    print(f"Skipping {filename}: Image size does not match the original size {original_size}")
            else:
                print(f"Skipping {filename}: Failed to load")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python resize_images.py <input_dir> <output_dir> <original_width> <original_height> <target_width> <target_height>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    original_width = int(sys.argv[3])
    original_height = int(sys.argv[4])
    target_width = int(sys.argv[5])
    target_height = int(sys.argv[6])

    original_size = (original_width, original_height)
    target_size = (target_width, target_height)

    resize_images(input_directory, output_directory, original_size, target_size)
