
from pathlib import Path
import nibabel as nib # 의료영상 파일을 읽기 위해
import numpy as np
import matplotlib.pyplot as plt


case_dir = Path("data/brats/BraTS2024/training_data/BraTS-GLI-00005-100")


files = sorted(case_dir.glob("*.nii.gz")) 
print("Found files:")
for f in files:
    print(" -", f.name)


img_path = None
for pattern in ["*t2f.nii.gz", "*flair.nii.gz", "*t2f.nii", "*flair.nii"]:
    matches = list(case_dir.glob(pattern))
    if matches:
        img_path = matches[0]
        break

seg_matches = list(case_dir.glob("*seg.nii.gz"))
if not seg_matches:
    raise FileNotFoundError(f"seg 파일을 못 찾았어: {case_dir}")
seg_path = seg_matches[0]

if img_path is None:
    raise FileNotFoundError(f"t2f/flair 파일을 못 찾았어: {case_dir}")

print("Using image:", img_path.name)
print("Using mask :", seg_path.name)

img = nib.load(str(img_path)).get_fdata()
seg = nib.load(str(seg_path)).get_fdata()

print("image shape:", img.shape) 
print("mask shape :", seg.shape)
print("mask unique values:", np.unique(seg))

tumor_per_slice = [np.sum(seg[:, :, i] > 0) for i in range(seg.shape[2])]
z = int(np.argmax(tumor_per_slice))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img[:, :, z], cmap="gray")
axes[0].set_title(f"Image ({z})")
axes[0].axis("off")

axes[1].imshow(seg[:, :, z], cmap="gray")
axes[1].set_title("Mask")
axes[1].axis("off")

axes[2].imshow(img[:, :, z], cmap="gray")
axes[2].imshow(seg[:, :, z], cmap="jet", alpha=0.35)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout() 
plt.savefig("outputs/brats_best_slice.png", dpi=150)
plt.show()

