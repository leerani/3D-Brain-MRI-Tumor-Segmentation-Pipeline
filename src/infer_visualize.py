import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, CropForegroundd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
import nibabel as nib
import scipy.ndimage as ndi

# ------------------------
# 경로
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_root = PROJECT_ROOT / "data" / "brats" / "BraTS2024" / "training_data"
model_path = PROJECT_ROOT / "outputs" / "best_model_n.pth"

# ------------------------
# 케이스 하나 선택
# ------------------------
case = sorted(list(data_root.glob("BraTS-*")))[28]  # 아무거나 하나

t1n = list(case.glob("*t1n.nii.gz"))[0]
t1c = list(case.glob("*t1c.nii.gz"))[0]
t2w = list(case.glob("*t2w.nii.gz"))[0]
t2f = list(case.glob("*t2f.nii.gz"))[0]
label = list(case.glob("*seg.nii.gz"))[0]

data = {
    "image": [str(t1n), str(t1c), str(t2w), str(t2f)],
    "label": str(label)
}

# ------------------------
# transform
# ------------------------
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

data = transforms(data)

image = data["image"].unsqueeze(0)  # [1, 4, D, H, W]
label = data["label"]

# ------------------------
# 모델 로드
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

# ------------------------
# inference
# ------------------------
with torch.no_grad():
    output = sliding_window_inference(
        image.to(device),
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
    )

img = image.numpy()[0, 3]  # t2f 채널
gt = (label.numpy()[0] > 0).astype(np.float32)

# ------------------------
# slice 선택
# ------------------------
z = np.argmax([np.sum(gt[:, :, i]) for i in range(gt.shape[2])])

# ------------------------
# 2D slice 기준 post-processing
# ------------------------
pred = (output > 0.5).float().cpu().numpy()[0, 0]

slice_pred = pred[:, :, z]

labeled, num = ndi.label(slice_pred)

if num > 0:
    sizes = ndi.sum(slice_pred, labeled, range(1, num + 1))
    largest = sizes.argmax() + 1
    slice_pred = (labeled == largest).astype(np.float32)
else:
    slice_pred = slice_pred.astype(np.float32)



# ------------------------
# 시각화
# ------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 원본 MRI slice 시각화용
axes[0].imshow(img[:, :, z], cmap="gray")
axes[0].set_title("MRI (T2-FLAIR)")

# 정답 마스크 시각화용
axes[1].imshow(gt[:, :, z], cmap="gray")
axes[1].set_title("Ground Truth")

# 예측 마스크 시각화용
axes[2].imshow(slice_pred, cmap="gray")
axes[2].set_title("Prediction")

# 보고 싶은 slice index
axes[3].imshow(img[:, :, z], cmap="gray")
axes[3].imshow(slice_pred, cmap="Greens", alpha=0.35)
axes[3].set_title("Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs" / "result_tuning_28.png", dpi=150)
plt.show()