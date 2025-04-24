import fiftyone.zoo as foz

# Download 100 images from COCO 2017 val split (without launching the app)
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections", "segmentations"],
    max_samples=100,
)

