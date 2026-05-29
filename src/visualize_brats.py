# BraTS 3D 의료영상 데이터에서 특정 케이스의 MRI 이미지와 종양 마스크를 불러와서, 
# 종양이 가장 많이 보이는 z-slice를 자동으로 찾아 시각화

# Nifti: MRI 뇌 이미지 데이터를 저장하기 위해 사용되는 오픈 파일 포맷

from pathlib import Path
import nibabel as nib # 의료영상 파일을 읽기 위해
import numpy as np
import matplotlib.pyplot as plt

# 케이스 폴더 경로 수정
case_dir = Path("data/brats/BraTS2024/training_data/BraTS-GLI-00005-100")

# 폴더 안 파일 전부 출력
files = sorted(case_dir.glob("*.nii.gz")) # 특정 패턴을 가진 파일 탐색
print("Found files:")
for f in files:
    print(" -", f.name)

# FLAIR/t2f 둘 다 대응
img_path = None
for pattern in ["*t2f.nii.gz", "*flair.nii.gz", "*t2f.nii", "*flair.nii"]:
    matches = list(case_dir.glob(pattern))
    if matches:
        img_path = matches[0]
        break

# seg 찾기 (종양 마스크 파일을 찾는)
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

print("image shape:", img.shape) # 가로 x 세로 x z-slice 개수
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

plt.tight_layout() # 이미지 간격 자동 정리
plt.savefig("outputs/brats_best_slice.png", dpi=150)
plt.show()

# BraTS 케이스 폴더 지정
# → MRI 파일과 seg 마스크 파일 찾기
# → nibabel로 3D 볼륨 로드
# → 마스크 기준으로 종양 픽셀이 가장 많은 z-slice 선택
# → 원본 / 마스크 / 오버레이 3장 시각화
# → outputs/brats_best_slice.png로 저장