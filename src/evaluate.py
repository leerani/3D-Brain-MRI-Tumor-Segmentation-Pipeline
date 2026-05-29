from pathlib import Path

import numpy as np
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CropForegroundd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric


PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_root = PROJECT_ROOT / "data" / "brats" / "BraTS2024" / "training_data"

cases = sorted(list(data_root.glob("BraTS-*")))[:100]
print(f"Total selected cases: {len(cases)}")

data_dicts = []
for case in cases:
    t1n = list(case.glob("*t1n.nii.gz"))[0]
    t1c = list(case.glob("*t1c.nii.gz"))[0]
    t2w = list(case.glob("*t2w.nii.gz"))[0]
    t2f = list(case.glob("*t2f.nii.gz"))[0]
    label = list(case.glob("*seg.nii.gz"))[0]

    data_dicts.append({
        "image": [str(t1n), str(t1c), str(t2w), str(t2f)],
        "label": str(label),
    })

train_files = data_dicts[:70]
val_files = data_dicts[70:85]
test_files = data_dicts[85:100]

print(f"Train cases: {len(train_files)}")
print(f"Validation cases: {len(val_files)}")
print(f"Test cases: {len(test_files)}")

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

test_ds = Dataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model_path = PROJECT_ROOT / "outputs" / "best_model_n.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dice_metric = DiceMetric(include_background=True, reduction="mean")

post_pred = AsDiscrete(threshold=0.95)
post_label = AsDiscrete(threshold=0.5)

ious = []

with torch.no_grad():
    for idx, test_data in enumerate(test_loader, start=1):
        val_inputs = test_data["image"].to(device)
        val_labels = (test_data["label"] > 0).float().to(device)

        # BraTS labels are converted to binary whole-tumor masks.
        # label > 0: tumor, label == 0: background
        val_labels = (test_data["label"] > 0).float().to(device)

        val_outputs = sliding_window_inference(
            val_inputs,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
        )

        val_outputs = torch.sigmoid(val_outputs)

        val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]

        dice_metric(y_pred=val_outputs_list, y=val_labels_list)

        pred = val_outputs_list[0].cpu().numpy().astype(bool)
        gt = val_labels_list[0].cpu().numpy().astype(bool)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        iou = 1.0 if union == 0 else intersection / union
        ious.append(iou)

        print(
            f"[{idx}/{len(test_loader)}]",
            "pred voxels:", pred.sum(),
            "gt voxels:", gt.sum(),
            "intersection:", intersection,
            "union:", union,
            f"IoU: {iou:.4f}",
        )

mean_dice = dice_metric.aggregate().item()
dice_metric.reset()

mean_iou = np.mean(ious)

print("\n=== BraTS Whole Tumor Binary Segmentation ===")
print(f"Dice: {mean_dice:.4f}")
print(f"IoU : {mean_iou:.4f}")