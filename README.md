# 🛡️ Skin Analysis Engine v1.0
OpenCV 기반의 안면 피부 부위별 분석 및 수치 테스트 엔진

## 📊 Performance Validation
AI-Hub의 '한국인 안면 피부 데이터' 라벨링(JSON)과 측정치를 대조하여 검증 완료

| 분석 항목 | 장비 측정치 | 알고리즘 결과 | 오차율 | 상태 |
| :--- | :---: | :---: | :---: | :---: |
| **색소 침착** | 147개 | 106개 | **27.9%** | ✅ PASS |
| **모공 지수** | 61점 | 69점 | **13.1%** | ✅ PASS |
| **피지(유분)** | 33.0% | 33.7% | **2.1%** | ✅ PASS |

<img width="1950" height="600" alt="result" src="https://github.com/user-attachments/assets/1ff80b10-184a-4600-951d-c8af526c84dd" />


## 🛠️ 핵심 기술 (Core Tech)
- **Image Preprocessing**: `cv2.cvtColor(LAB)`, `cv2.createCLAHE`를 통한 조명 정규화
- **ROI Extraction**: JSON Metadata의 `bbox` 좌표를 활용한 부위별(Facepart) 정밀 크롭
- **Feature Detection**: 
  - `Laplacian Variance`를 활용한 모공 텍스처 수치화
  - `Adaptive Threshold` & `Morphology`를 이용한 색소 침착 카운팅

## 📈 Optimization Log
- **Calibration**: 장비 측정치와의 편차를 줄이기 위해 `MAX_VALUES` 및 탐지 파라미터 최적화
- **Error Handling**: 이미지 경로 인식 오류(`raw string`) 및 JSON 파싱 예외 처리 완료
