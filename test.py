import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.pyplot as plt
from matplotlib import rc

# 맑은 고딕 설정 (윈도우 기준)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ── 전처리 ──────────────────────────────────────
def preprocess(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    return img_bgr, img_gray

# ── ROI: bbox 좌표 직접 사용 ──────────────────────
def extract_roi_by_bbox(img, bbox):
    """JSON의 bbox [x, y, w, h] 로 정확하게 자르기"""
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def extract_roi_center(img_bgr, img_gray):
    """bbox 없을 때 임시 중앙 크롭"""
    h, w = img_gray.shape
    return img_bgr[h//4:h*3//4, w//4:w*3//4], img_gray[h//4:h*3//4, w//4:w*3//4]

def detect_pigmentation(roi_gray):
    binary = cv2.adaptiveThreshold(
        roi_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,   # 21 → 51  (더 넓은 영역 기준으로 판단, 노이즈 감소)
        C=20            # 10 → 20  (더 엄격하게, 진한 반점만 잡음)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spots = [c for c in contours if 80 < cv2.contourArea(c) < 1500]
    #                                 ↑ 30→80  (작은 노이즈 제거)
    return len(spots)

# ── 모공 ─────────────────────────────────────────
def detect_pore(roi_gray):
    """Laplacian variance → l/r_cheek_pore 평균과 비교"""
    return float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

# ── 피지 (수분의 역수로 추정) ─────────────────────
def detect_sebum(roi_gray):
    """밝기 반사 → moisture 역수와 비교"""
    _, thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(thresh)) / thresh.size

# ── 정규화 ────────────────────────────────────────
MAX_VALUES = {
    "pigmentation": 200,    # AI Hub 장비 측정치 기준 캘리브레이션 (오차 27.9%)
    "pore":         20.0,   # l/r_cheek_pore 점수 비교 기준 (오차 11.5%)
    "sebum":        0.6,    # moisture 역수 기준 (오차 2.1%)
}

def normalize(key, value):
    return min(100, int(value / MAX_VALUES[key] * 100))

# ── JSON 8개 파싱 → 비교 기준값 추출 ─────────────
def parse_label_files(json_paths):
    """
    facepart별로 JSON을 분류해서
    비교에 필요한 장비 측정치만 추출
    """
    ref = {
        "pigmentation_count": None,  # facepart 0
        "pore_avg":           None,  # facepart 5,6 평균
        "moisture_avg":       None,  # facepart 1,5,6 평균
        "bbox_by_part":       {},    # facepart → bbox
    }
    moisture_vals = []
    pore_vals     = []

    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)

        part = d["images"]["facepart"]
        bbox = d["images"]["bbox"]
        ref["bbox_by_part"][part] = bbox
        eq   = d.get("equipment") or {}

        if part == 0:
            ref["pigmentation_count"] = eq.get("pigmentation_count")

        if part == 1 and "forehead_moisture" in eq:
            moisture_vals.append(eq["forehead_moisture"])

        if part == 5:
            if "l_cheek_pore" in eq:
                pore_vals.append(eq["l_cheek_pore"])
            if "l_cheek_moisture" in eq:
                moisture_vals.append(eq["l_cheek_moisture"])

        if part == 6:
            if "r_cheek_pore" in eq:
                pore_vals.append(eq["r_cheek_pore"])
            if "r_cheek_moisture" in eq:
                moisture_vals.append(eq["r_cheek_moisture"])

    if pore_vals:
        ref["pore_avg"] = round(sum(pore_vals) / len(pore_vals), 1)
    if moisture_vals:
        ref["moisture_avg"] = round(sum(moisture_vals) / len(moisture_vals), 1)

    return ref

# ── 편차 계산 + 튜닝 힌트 ────────────────────────
def calc_error(label, algo, label_max, key):
    """오차율 계산 및 MAX_VALUES 조정 힌트 출력"""
    if label is None or algo is None:
        return None
    error = abs(algo - label)
    rate  = round(error / max(label, 1) * 100, 1)
    ratio = algo / max(label, 1)

    hint = ""
    if ratio > 1.3:
        hint = f"  ▲ 과검출 → MAX_VALUES['{key}'] 를 {round(MAX_VALUES[key]*ratio,1)} 으로 올려봐"
    elif ratio < 0.7:
        hint = f"  ▼ 미검출 → MAX_VALUES['{key}'] 를 {round(MAX_VALUES[key]*ratio,1)} 으로 내려봐"
    else:
        hint = f"  ✓ 오차 {rate}% — 허용 범위 (30% 이내)"

    return {"error": error, "rate": rate, "hint": hint}

# ── 비교 출력 ─────────────────────────────────────
def compare(ref, my_raw, my_scores):
    """
    장비 측정치 vs 내 알고리즘 raw 값 비교.
    pore는 단위가 달라서 정규화 점수로 비교.
    """
    # moisture → sebum 역수 변환 (수분 100% = 피지 0)
    moisture_as_sebum = round(100 - ref["moisture_avg"], 1) if ref["moisture_avg"] else None

    rows = [
        # (항목명,       장비값,                        내 raw값,                단위)
        ("pigmentation", ref["pigmentation_count"],     my_raw["pigmentation"],  "개"),
        ("pore(점수)",   None,                          None,                    ""),   # 단위 달라 점수 비교
        ("sebum(수분역)", moisture_as_sebum,            round(my_raw["sebum"]*100,1), "%"),
    ]

    print("\n" + "="*52)
    print("      알고리즘 vs 전문 장비 비교 리포트")
    print("="*52)
    print(f"  {'항목':<16} {'장비 측정치':>12}  {'내 알고리즘':>12}  {'오차율':>6}")
    print("-"*52)

    errors = {}

    # pigmentation
    pig_err = calc_error(ref["pigmentation_count"], my_raw["pigmentation"],
                         MAX_VALUES["pigmentation"], "pigmentation")
    if pig_err:
        print(f"  {'pigmentation':<16} {ref['pigmentation_count']:>12}개  "
              f"{int(my_raw['pigmentation']):>11}개  {pig_err['rate']:>5}%")
        print(pig_err["hint"])
        errors["pigmentation"] = pig_err["rate"]

    # pore — 단위 달라서 점수로 비교
    pore_score_label = None
    if ref["pore_avg"]:
        # 장비 pore는 픽셀 면적(629), 우리는 variance → 점수로만 비교
        pore_score_label = min(100, int(ref["pore_avg"] / 1000 * 100))
    pore_score_algo = my_scores["pore"]
    pore_err = calc_error(pore_score_label, pore_score_algo, 100, "pore")
    if pore_err:
        print(f"  {'pore(점수)':<16} {pore_score_label:>11}점  "
              f"{pore_score_algo:>11}점  {pore_err['rate']:>5}%")
        print(pore_err["hint"])
        errors["pore"] = pore_err["rate"]

    # sebum (수분 역수)
    if moisture_as_sebum is not None:
        sebum_algo = round(my_raw["sebum"] * 100, 1)
        sebum_err  = calc_error(moisture_as_sebum, sebum_algo, 100, "sebum")
        if sebum_err:
            print(f"  {'sebum(수분역)':<16} {moisture_as_sebum:>11}%  "
                  f"{sebum_algo:>11}%  {sebum_err['rate']:>5}%")
            print(sebum_err["hint"])
            errors["sebum"] = sebum_err["rate"]

    print("="*52)
    return errors

# ── 시각화 ───────────────────────────────────────
def visualize(my_scores, ref, errors, save_path="result.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 왼쪽: 내 알고리즘 점수
    ax1 = axes[0]
    keys   = list(my_scores.keys())
    values = list(my_scores.values())
    bars = ax1.bar(keys, values, color=["#faf1f2", "#f1d1d2", "#c1a3a3"]) # "#7d5959"
    ax1.set_ylim(0, 100)
    ax1.set_title("내 알고리즘 점수 (낮을수록 좋음)")
    ax1.set_ylabel("점수")
    for bar, v in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 str(v), ha='center', fontsize=10)

    # 오른쪽: 장비 vs 알고리즘 오차율
    ax2 = axes[1]
    if errors:
        err_keys = list(errors.keys())
        err_vals = list(errors.values())
        bar_colors = ["#86dadb" if v <= 30 else "#007095" for v in err_vals]
        bars2 = ax2.bar(err_keys, err_vals, color=bar_colors)
        ax2.axhline(y=30, color='gray', linestyle='--', linewidth=1, label='허용 기준 30%')
        ax2.set_ylim(0, max(max(err_vals) * 1.3, 50))
        ax2.set_title("장비 측정치 대비 오차율 (%)")
        ax2.set_ylabel("오차율 (%)")
        ax2.legend()
        for bar, v in zip(bars2, err_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f"{v}%", ha='center', fontsize=10)
    else:
        ax2.text(0.5, 0.5, "비교 데이터 없음", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=13, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  결과 저장: {save_path}")

# ── 메인 ─────────────────────────────────────────
def analyze(image_path, json_paths=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"이미지를 못 읽었어: {image_path}")

    img_bgr, img_gray = preprocess(img_bgr)

    # JSON bbox가 있으면 정확한 ROI, 없으면 중앙 크롭
    ref = None
    if json_paths:
        ref = parse_label_files(json_paths)
        bbox_full = ref["bbox_by_part"].get(0)  # facepart 0 = 전체 얼굴
        if bbox_full:
            roi_bgr  = extract_roi_by_bbox(img_bgr,  bbox_full)
            roi_gray = extract_roi_by_bbox(img_gray, bbox_full)
        else:
            roi_bgr, roi_gray = extract_roi_center(img_bgr, img_gray)
    else:
        roi_bgr, roi_gray = extract_roi_center(img_bgr, img_gray)

    raw = {
        "pigmentation": detect_pigmentation(roi_gray),
        "pore":         detect_pore(roi_gray),
        "sebum":        detect_sebum(roi_gray),
    }

    scores = {k: normalize(k, v) for k, v in raw.items()}
    total  = 100 - round(sum(scores.values()) / len(scores))

    print("=== 피부 분석 결과 ===")
    for k, v in scores.items():
        unit = "개" if k == "pigmentation" else f"{raw[k]:.4f}"
        print(f"  {k:>14}: {v:>3}점  (raw: {unit})")
    print(f"  {'종합':>14}: {total}점")

    errors = {}
    if ref:
        errors = compare(ref, raw, scores)

    visualize(scores, ref or {}, errors)
    return scores, total

# ── 실행 ─────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "jpg_0001/sample.jpg"

    # 같은 폴더의 JSON 8개를 리스트로 넘기면 돼
    JSON_PATHS = [
        "labeling_0001/0001_sample_00.json",
        "labeling_0001/0001_sample_01.json",
        "labeling_0001/0001_sample_02.json",
        "labeling_0001/0001_sample_03.json",
        "labeling_0001/0001_sample_04.json",
        "labeling_0001/0001_sample_05.json",
        "labeling_0001/0001_sample_06.json",
        "labeling_0001/0001_sample_07.json",
        "labeling_0001/0001_sample_08.json",
    ]
    # 없는 파일은 자동으로 건너뜀
    existing = [p for p in JSON_PATHS if os.path.exists(p)]
    analyze(IMAGE_PATH, existing)