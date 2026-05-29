# BraTS 3D 의료영상 데이터를 MONAI 학습용 Dataset/DataLoader 형태로 준비하고, 
# 실제로 한 batch가 잘 나오는지 확인

from pathlib import Path
from monai.transforms import ( # 의료영상 전처리 도구
    Compose,             # transform들을 순서대로 묶어줌
    LoadImaged,
    EnsureChannelFirstd, # 이미지 shape를 모델 입력에 맞게 바꿈
    NormalizeIntensityd, # 이미지 밝기값을 정규화
    CropForegroundd,     # 배경을 줄이고 실제 뇌 영역이 있는 부분 위주로 자름
)
from monai.data import Dataset, DataLoader
# Monai: Medical Open Network for AI
# 의료영상 AI를 만들기 위한 Pytorch 기반 라이브러리
# FLAIR는 뇌 병변이 잘 보이도록 만든 MRI 촬영 방식 중 하나

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

# 프로젝트 기준 경로 잡기
# → BraTS training_data 폴더 찾기
# → 케이스 10개 선택
# → image/label 경로를 딕셔너리로 정리
# → MONAI transform 정의
# → Dataset 생성
# → DataLoader 생성
# → batch 하나 꺼내서 shape 확인