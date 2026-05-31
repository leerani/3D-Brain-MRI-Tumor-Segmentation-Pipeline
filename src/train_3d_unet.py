from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd, 
    RandCropByPosNegLabeld,
    EnsureTyped,     
    RandFlipd,        
    RandRotate90d,    
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric 
from monai.inferers import sliding_window_inference 
from monai.transforms import AsDiscrete 
from monai.data import decollate_batch  

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_root = PROJECT_ROOT / "data" / "brats" / "BraTS2024" / "training_data"

cases = sorted(list(data_root.glob("BraTS-*")))[:100]
print(f"총 선택된 케이스 수: {len(cases)}")

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
test_files = data_dicts[85:]


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=4,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5),
    
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(
    sigmoid=True,
    squared_pred=True,
    lambda_dice=1.0,
    lambda_ce=0.5,
)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)
dice_metric = DiceMetric(include_background=True, reduction="mean")

post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)



output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

max_epochs = 50
val_interval = 1
best_metric = -1

for epoch in range(max_epochs):
    print("-" * 30)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0

    for step, batch_data in enumerate(train_loader, start=1):
        inputs = batch_data["image"].to(device)

        labels = (batch_data["label"] > 0).float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")

    epoch_loss /= len(train_loader)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        dice_metric.reset() 

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = (val_data["label"] > 0).float().to(device)

                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=model,
                )

                val_outputs = torch.sigmoid(val_outputs)

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            print(f"validation dice: {metric:.4f}")

            scheduler.step(metric)

            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), output_dir / "best_model_n.pth")
                print(f"saved new best model: {best_metric:.4f}")
