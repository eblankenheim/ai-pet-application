from fastai.vision.all import *
from pathlib import Path
import re

def main():
    path = untar_data(URLs.PETS)
    images_path = path/'images'
    fnames = get_image_files(images_path)
    pat = r'(.+)_\d+\.jpg$'
    breeds = set([re.match(pat, f.name).groups()[0] for f in fnames])
    print("Available breeds:", breeds)
    print("Number of breeds:", len(breeds))

    dls = ImageDataLoaders.from_name_re(
        images_path, fnames, pat, valid_pct=0.2, seed=42,
        item_tfms=Resize(256), batch_tfms=None,
        bs=8, num_workers=0
    )

    learn = vision_learner(dls, resnet34, metrics=accuracy)
    learn.fine_tune(15)  # Only 6 epochs
    export_path = Path(__file__).parent / 'export.pkl'
    learn.export(export_path)
    print(f"Model saved to {export_path.resolve()}")

if __name__ == "__main__":
    main()