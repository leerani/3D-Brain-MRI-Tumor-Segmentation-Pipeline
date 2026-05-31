from pathlib import Path
from monai.transforms import ( 
    Compose,             
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd, 
    CropForegroundd,    
)
from monai.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_root = PROJECT_ROOT / "data" / "brats" / "BraTS2024" / "training_data"


cases = sorted(list(data_root.glob("BraTS-*")))[:10]
print(f"총 선택된 케이스 수: {len(cases)}")

data_dicts = []

for case in cases:

    img = list(case.glob("*t2f.nii.gz"))[0]


    label = list(case.glob("*seg.nii.gz"))[0]

    data_dicts.append({
        "image": str(img),
        "label": str(label)
    })

print("샘플 데이터:")
print(data_dicts[0])


transforms = Compose([
    LoadImaged(keys=["image", "label"]), 
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
])


dataset = Dataset(data=data_dicts, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


for batch in loader:
    print("image shape:", batch["image"].shape)
    print("label shape:", batch["label"].shape)
    break
