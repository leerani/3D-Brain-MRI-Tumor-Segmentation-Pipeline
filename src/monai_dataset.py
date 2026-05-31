# BraTS 3D 의료영상 데이터를 MONAI 학습용 Dataset/DataLoader 형태로 준비하고, 
# 실제로 한 batch가 잘 나오는지 확인

from pathlib import Path
from monai.transforms import ( 
    Compose,             
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd, 
    CropForegroundd,    
)
from monai.data import Dataset, DataLoader

# ------------------------
# 1. 데이터 경로 설정
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_root = PROJECT_ROOT / "data" / "brats" / "BraTS2024" / "training_data"

# ------------------------
# 2. 케이스 10개만 선택
# ------------------------
cases = sorted(list(data_root.glob("BraTS-*")))[:10]
print(f"총 선택된 케이스 수: {len(cases)}")

# ------------------------
# 3. 데이터 dict 구성
# ------------------------
data_dicts = []

for case in cases:
    # FLAIR (t2f) 찾기
    img = list(case.glob("*t2f.nii.gz"))[0]

    # segmentation
    label = list(case.glob("*seg.nii.gz"))[0]

    data_dicts.append({
        "image": str(img),
        "label": str(label)
    })

print("샘플 데이터:")
print(data_dicts[0])

# ------------------------
# 4. transform 정의
# ------------------------
transforms = Compose([
    LoadImaged(keys=["image", "label"]), # 파일 경로 문자열이 실제 데이터로 바뀜
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
])

# ------------------------
# 5. Dataset / DataLoader
# ------------------------
dataset = Dataset(data=data_dicts, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------
# 6. 데이터 확인
# ------------------------
for batch in loader:
    print("image shape:", batch["image"].shape)
    print("label shape:", batch["label"].shape)
    break
