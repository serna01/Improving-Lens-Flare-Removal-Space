import os
import sys
import cv2

def resize_image(input_dir, output_dir, target_size=(1008, 752)):
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            if img is not None:
                if img.shape[0] == 756 and img.shape[1] == 1008:
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, resized_img)
                    print(f"Resized and saved {filename} to {output_dir}")
                else:
                    print(f"Skipping {filename}: Image is not 1008x756 pixels")
            else:
                print(f"Skipping {filename}: Failed to load")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resize_images.py <input_dir> <output_dir>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    resize_image(input_directory, output_directory)
