# 🛡️ Skin Analysis Test Engine v1.0
OpenCV 기반의 안면 피부 부위별 분석 및 수치 테스트 엔진

## 📂 프로젝트 구조
```text
MIRROR Me
ai_engine/
├── models/                # YOLOv8 학습 완료 가중치 (.pt, .onnx)
├── core/                  # 핵심 분석 알고리즘 모듈 (유소희 담당)
│   ├── processor.py       # preprocess, extract_roi 로직
│   ├── detectors.py       # detect_acne, pigmentation, pore, sebum 로직
│   └── normalizer.py      # MAX_VALUES 기반 점수 정규화 및 수치화 알고리즘
├── utils/                 # 보조 기능 모듈
│   ├── json_parser.py     # parse_label_files (장비 측정치 파싱)
│   └── visualizer.py      # visualize (분석 결과 시각화 및 차트 생성)
├── engine.py              # 전체 파이프라인 실행 엔트리포인트
└── config.yaml            # MAX_VALUES 및 알고리즘 하이퍼파라미터 설정
```

## 📊 Performance Validation

### 단일 샘플 검증 (AI Hub 한국인 안면 피부 데이터)
AI Hub의 '한국인 안면 피부 데이터' 라벨링(JSON) 및 전문 장비 측정치와 1:1 대조 검증

| 분석 항목 | 장비 측정치 | 알고리즘 결과 | 오차율 | 상태 |
| :--- | :---: | :---: | :---: | :---: |
| **색소 침착** | 147개 | 106개 | **27.9%** | ✅ PASS |
| **모공 지수** | 61점 | 68점 | **11.5%** | ✅ PASS |
| **피지(유분)** | 33.0% | 33.7% | **2.1%** | ✅ PASS |
| **여드름** | 60개 | 54개 | **10.0%** | ✅ PASS |

### 다중 샘플 교차 검증 (피부 타입별 분기 처리 적용)
피부 타입(지성/건성)이 다른 2개 샘플로 교차 검증하여 알고리즘 일반화 성능 확인

| 분석 항목 | 샘플 1 오차율 | 샘플 2 오차율 | **평균 오차율** | 상태 |
| :--- | :---: | :---: | :---: | :---: |
| **색소 침착** | 27.9% | 12.3% | **20.1%** | ✅ PASS |
| **모공 지수** | 16.4% | 20.0% | **18.2%** | ✅ PASS |
| **피지(유분)** | 2.1% | 20.1% | **11.1%** | ✅ PASS |
| **여드름** (ACNE04) | 13.1% | 10.0% | **11.6%** | ✅ PASS |

> 캡스톤 목표 기준: 오차율 30% 이내 → **4개 항목 전부 달성**

<img width="1950" height="600" alt="result" src="https://github.com/user-attachments/assets/1ff80b10-184a-4600-951d-c8af526c84dd" />
<img width="448" height="146" alt="terminal" src="https://github.com/user-attachments/assets/a24bbaba-91ef-4674-9d34-05e9d1996e0d" />

## 🛠️ 핵심 기술 (Core Tech)

### 전처리
- **Lighting Normalization**: `cv2.cvtColor(LAB)` + `cv2.createCLAHE`를 통한 조명 환경 정규화
- **ROI Extraction**: JSON Metadata의 `bbox` 좌표 기반 부위별(Facepart) 정밀 크롭 / OpenCV Haar Cascade 얼굴 인식 폴백

### 탐지 알고리즘
- **여드름**: HSV 색공간 기반 붉은기 영역 반점 추출 (`inRange` + `morphologyEx`)
- **색소 침착**: `Adaptive Threshold` + `Morphology Open`을 이용한 어두운 반점 카운팅
- **모공**: `Laplacian Variance`를 활용한 피부 표면 거칠기(텍스처) 수치화
- **피지**: 밝기 임계값 기반 하이라이트 픽셀 비율 추출

### 피부 타입 분기
- ROI 평균 밝기(≥130: 지성, <130: 건성)로 자동 판별
- 피부 타입별 탐지 파라미터 분기 적용 → 다중 샘플 일반화 성능 확보

## 📈 Optimization Log
| 항목 | 내용 |
| :--- | :--- |
| **Calibration** | AI Hub 장비 측정치 및 ACNE04 마스크와 비교하여 `MAX_VALUES` 및 탐지 파라미터 최적화 |
| **Skin Type Branch** | 피부 타입(지성/건성) 자동 분기로 sebum·pigmentation 다중 샘플 오차율 개선 |
| **Error Handling** | 이미지 경로 인식 오류(`raw string`) 및 JSON 파싱 예외 처리 완료 |
| **Dataset** | AI Hub 한국인 피부 데이터(pigmentation·pore·sebum) + ACNE04-v2 Kaggle(acne) 병행 활용 |

## 🔜 Next Steps
- [ ] YOLOv8 여드름 탐지 모델 학습 (Roboflow 데이터셋 준비 후 HSV 임시 알고리즘 교체)
- [ ] Flask API → Android 앱 연결 테스트
- [ ] MediaPipe 랜드마크 기반 정밀 ROI 교체 (현재 Haar Cascade 사용 중)