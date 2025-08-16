
# -------------------------------------------------------------
# train.py
#
# @description
#   This script trains a pet image classifier using fastai and PyTorch.
#   It downloads the Oxford-IIIT PETS dataset, prepares the data,
#   trains a ResNet50 model, logs progress, and saves model checkpoints.
#
# Sections:
#   1. Imports and setup
#   2. Custom callback for checkpointing
#   3. Main function: data prep, training, logging, export
#
# Usage:
#   python basic-training-model.py
# 
# Dependency installation command
# pip install --no-cache-dir fastai torch torchvision pillow
# 
# 
# Best run to date: 
#   - Epoch 10: Accuracy 0.949256
#  
# -------------------------------------------------------------

# 1. Imports and setup
from fastai.vision.all import *
import logging
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path



## 3. Main function: data prep, training, logging, export
def main():
    # download dataset
    path = untar_data(URLs.PETS)
    images_path = path/'images'
    fnames = get_image_files(images_path)
    print(f"Number of images in dataset: {len(fnames)}")
    logging.basicConfig(
        filename=Path(__file__).parent / 'training.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.info(f"Starting training with {len(fnames)} images.")

    # regex to extract label from filename like "Abyssinian_1.jpg"
    pat = r'(.+)_\d+\.jpg$'

    dls = ImageDataLoaders.from_name_re(
        images_path, fnames, pat, valid_pct=0.2, seed=42,
        item_tfms=Resize(460), batch_tfms=aug_transforms(size=224),
        num_workers=0   # 👈 add this for Windows safety
    )


    learn = vision_learner(dls, resnet50, metrics=accuracy)
    logging.info("Beginning fine-tune...")
    learn.fine_tune(10)   # short demo: increase epochs for better results
    logging.info("Training complete.")

    # export a single-file model for inference
    export_path = Path(__file__).parent / 'export.pkl'
    learn.export(export_path)
    print(f"Model saved to {export_path.resolve()}")
    logging.info(f"Final model saved to {export_path.resolve()}")
    # Print the location of the pathlib module for debugging
    import pathlib
    print(f"pathlib module location: {pathlib.__file__}")

if __name__ == "__main__":
    main()

