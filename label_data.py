from pathlib import Path
from PIL import Image
import os


def label_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"The provided path {image_dir} is not a valid directory.")
    
    print("there are ", len(list(image_dir.glob('*_label_.jpg'))), "images to label.")
    # Only pick images that are still unlabeled: *_label_.jpg
    for image_path in image_dir.glob('*_label_.jpg'):
        with Image.open(image_path) as img:
            img.show()
            label = input(f"Enter label for {image_path.name} (type 'del' to delete): ")
        
        if label.strip().lower() == "del":
            image_path.unlink()
            print(f"Deleted {image_path.name}")
        else:
            # Rename to include label after label__ (double underscore to match your pattern)
            stem_without_label = image_path.stem.rsplit('_label_', 1)[0]
            new_name = f"{stem_without_label}_label__{label}{image_path.suffix}"
            image_path.rename(image_path.with_name(new_name))
            print(f"Labeled and renamed to {new_name}")
            

def filter_data_names(image_dir):
    image_dir = Path(image_dir)
    for i, image_path in enumerate(image_dir.glob('*.jpg')):
        new_name = f"image_{i}_label_.jpg"
        image_path.rename(image_path.with_name(new_name))
        

if __name__ == "__main__":
    image_directory = 'subset'  # Path to the directory containing images
    # filter_data_names(image_directory)
    label_images(image_directory)