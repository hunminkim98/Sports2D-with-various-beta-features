# RT-DETRv4 Checkpoint Folder

이 폴더에 RT-DETRv4 체크포인트 파일(.pth)을 넣어주세요.

## 다운로드 링크
- **RT-DETRv4-X (xlarge)**: https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3

## 설치 방법

RT-DETRv4는 pip 패키지가 아니므로 수동 설치가 필요합니다:

```bash
# 1. 레포지토리 클론
git clone https://github.com/RT-DETRs/RT-DETRv4.git

# 2. PYTHONPATH에 추가 (Windows PowerShell)
$env:PYTHONPATH = "C:\path\to\RT-DETRv4;$env:PYTHONPATH"

# 3. 또는 configs 폴더만 복사
# RT-DETRv4/configs/rtv4/ → Sports2D/models/rtdetrv4/configs/
```

## 사용법
1. 위 링크에서 모델을 다운로드
2. 이 폴더에 저장 (예: `rtdetrv4_x.pth`)
3. Sports2D Config에서 `synthpose_detector = 'rtdetrv4'` 설정

## 지원 모델
| 파일명 | 모델 |
|--------|------|
| `rtdetrv4_s.pth` | Small |
| `rtdetrv4_m.pth` | Medium |
| `rtdetrv4_l.pth` | Large |
| `rtdetrv4_x.pth` | X-Large (기본값) |
