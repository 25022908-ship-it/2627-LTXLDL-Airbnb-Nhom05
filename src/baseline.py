import re
import yaml
import pandas as pd
from pathlib import Path

#Từ điển ngôn ngữ
ASPECT_PATTERNS: dict[str, re.Pattern] = {
    "vị trí": re.compile(
        r"\b(?:location|located|central|center|centre|metro|subway|tram|ubahn|u-bahn|"
        r"station|walk|walking|neighborhood|neighbourhood|area|close to|near|"
        r"lage|zentral|u-bahn|bahnhof|fußläufig|emplacement|ubicaci[oó]n)\b",
        re.IGNORECASE,
    ),
    "sạch sẽ": re.compile(
        r"\b(?:clean|cleanliness|spotless|tidy|hygiene|dirty|dust|dusty|stain|smell|"
        r"mold|hair|cockroach|bug|sauber|sauberkeit|schmutzig|dreckig|propre|limpio|sucio)\b",
        re.IGNORECASE,
    ),
    "chủ nhà": re.compile(
        r"\b(?:host|hosts|staff|owner|manager|communication|responsive|helpful|friendly|"
        r"welcoming|check-in|checkin|rude|unhelpful|gastgeber|freundlich|kommunikation|"
        r"h[oô]te|accueil|anfitri[oó]n)\b",
        re.IGNORECASE,
    ),
    "giá trị": re.compile(
        r"\b(?:value|price|priced|money|worth|affordable|cheap|expensive|overpriced|"
        r"budget|cost|preis|preis-leistungs|günstig|teuer|qualit[eé]-prix|precio|calidad)\b",
        re.IGNORECASE,
    ),
    "tiện nghi": re.compile(
        r"\b(?:wifi|wi-fi|internet|kitchen|bed|beds|mattress|shower|bathroom|towel|towels|"
        r"heating|ac|air conditioning|elevator|lift|washer|coffee|tv|equipment|equipped|"
        r"ausstattung|küche|bett|dusche|aufzug|heizung|wlan|[eé]quipement|literie|cocina|cama)\b",
        re.IGNORECASE,
    ),
    "độ ồn": re.compile(
        r"\b(?:quiet|peaceful|silent|calm|noise|noisy|loud|sound|street noise|construction|"
        r"party|neighbors|neighbours|walls|earplugs|ruhig|leise|lärm|laut|bruit|bruyant|"
        r"calme|ruido|ruidoso|tranquilo)\b",
        re.IGNORECASE,
    ),
    # Bỏ "sur" (tiếng Pháp = "trên", rất phổ biến) chỉ giữ "sûr" có dấu mũ
    "an toàn": re.compile(
        r"\b(?:safe|safety|secure|security|lock|locks|key|keys|neighborhood safe|sketchy|"
        r"dangerous|unsafe|sicher|sicherheit|schloss|sécurit[eé]|sûr|seguro|seguridad)\b",
        re.IGNORECASE,
    ),
}

POS_PATTERN = re.compile(
    r"\b(great|good|nice|excellent|perfect|amazing|wonderful|lovely|clean|comfortable|"
    r"cosy|cozy|recommend|recommended|best|super|fantastic|helpful|friendly|quiet|"
    r"easy|spacious|beautiful|awesome|tolle|super|sehr gut|sauber|schön|perfekt|"
    r"empfehlen|ruhig|freundlich|parfait|tr[eè]s bien|magnifique|excelente|genial|limpio)\b",
    re.IGNORECASE,
)

# "hot" đứng riêng mơ hồ (hot shower là khen) nên chỉ giữ "too hot"
NEG_PATTERN = re.compile(
    r"\b(bad|dirty|noisy|loud|terrible|horrible|worst|poor|broken|smell|smelly|cold|"
    r"too hot|uncomfortable|rude|expensive|overpriced|issue|issues|problem|problems|"
    r"disappointed|disappointing|not clean|no hot water|clogged|neglected|dusty|"
    r"schlecht|schmutzig|dreckig|laut|lärm|kaputt|kalt|teuer|probl[eè]me|sale|bruyant|"
    r"malo|sucio|ruidoso|roto)\b",
    re.IGNORECASE,
)

# Từ phủ định (EN/DE/FR/ES) và số từ nhìn ngược lại để xét phủ định
NEGATION_PATTERN = re.compile(
    r"(?:not|no|never|isn't|wasn't|aren't|weren't|don't|didn't|nicht|kein|keine|nie|pas|jamais|sin|nunca)",
    re.IGNORECASE,
)
NEGATION_WINDOW = 3

SENTENCE_SPLIT_RE = re.compile(r"[.!?\n;]+|\b(?:but|however|although|though|aber|mais|pero)\b", re.IGNORECASE)

# Bỏ "la" khỏi fr/es vì trùng nhau (và trùng cả tiếng Ý/Bồ Đào Nha)
LANG_PATTERNS: dict[str, re.Pattern] = {
    "de": re.compile(r"\b(und|die|der|das|ist|war|sehr|wir|mit|für|nicht|ein|eine|wohnung|sauber|lage)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(le|les|des|est|très|nous|avec|pour|dans|appartement|bien|propre|séjour)\b", re.IGNORECASE),
    "es": re.compile(r"\b(el|los|las|muy|con|para|una|todo|apartamento|ubicaci[oó]n|limpio|bien)\b", re.IGNORECASE),
    "en": re.compile(r"\b(the|and|was|very|with|for|stay|apartment|place|great|good|clean|location)\b", re.IGNORECASE),
}

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_comment(text: object) -> str:
    """Làm sạch thẻ HTML (<br/>) và khoảng trắng thừa; dữ liệu bẩn/NaN trả về chuỗi rỗng."""
    if pd.isna(text):
        return ""
    cleaned = re.sub(r"<br\s*/?>", " ", str(text), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()

def is_negated(text: str, start: int) -> bool:
    """Có từ phủ định trong NEGATION_WINDOW từ liền trước vị trí start (trong cùng câu) không."""
    same_sentence = re.split(r"[.!?;\n]", text[:start])[-1]
    prev_words = same_sentence.split()[-NEGATION_WINDOW:]
    return any(NEGATION_PATTERN.fullmatch(w.strip(",:\"'()")) for w in prev_words)

def count_polarity(text: str) -> tuple[int, int]:
    """Đếm (số từ khen, số từ chê) có xử lý phủ định: 'not great' -> chê, 'no problems' -> khen."""
    pos_count = neg_count = 0
    for m in POS_PATTERN.finditer(text):
        if is_negated(text, m.start()):
            neg_count += 1
        else:
            pos_count += 1
    for m in NEG_PATTERN.finditer(text):
        if is_negated(text, m.start()):
            pos_count += 1
        else:
            neg_count += 1
    return pos_count, neg_count

def detect_language_rule(text: object) -> str:
    """Nhận diện ngôn ngữ ('en', 'de', 'fr', 'es', 'other') bằng đếm tần suất từ khóa; hòa điểm -> 'other'."""
    clean_txt = normalize_comment(text)
    if not clean_txt:
        return "other"
    scores = {lang: len(pat.findall(clean_txt)) for lang, pat in LANG_PATTERNS.items()}
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return "other"
    return max(scores.items(), key=lambda x: x[1])[0]

def classify_sentiment_rule(text: object) -> str:
    """Phân loại cảm xúc ('positive', 'negative', 'mixed') dựa trên số từ khen và chê."""
    clean_txt = normalize_comment(text)
    if not clean_txt:
        return "positive"
    pos_count, neg_count = count_polarity(clean_txt)

    if pos_count > 0 and neg_count == 0:
        return "positive"
    if neg_count > 0 and pos_count == 0:
        return "negative"
    if pos_count > 0 and neg_count > 0:
        # Nếu số từ khen áp đảo gấp >= 3 lần số từ chê thì vẫn coi là positive, ngược lại là mixed
        if pos_count >= neg_count * 3:
            return "positive"
        if neg_count >= pos_count * 3:
            return "negative"
        return "mixed"
    return "positive"

def extract_aspects_rule(text: object, valid_aspects: list[str] | None = None) -> tuple[str, str]:
    """Tách câu theo mệnh đề để phân loại từng khía cạnh vào nhóm khen (pos) hoặc chê (neg)."""
    clean_txt = normalize_comment(text)
    if not clean_txt:
        return "", ""

    allowed = set(valid_aspects) if valid_aspects else set(ASPECT_PATTERNS.keys())
    clauses = [c.strip() for c in SENTENCE_SPLIT_RE.split(clean_txt) if c.strip()]

    pos_aspects: list[str] = []
    neg_aspects: list[str] = []

    for aspect, pat in ASPECT_PATTERNS.items():
        if aspect not in allowed:
            continue
        matched_clauses = [c for c in clauses if pat.search(c)]
        if not matched_clauses:
            continue

        counts = [count_polarity(c) for c in matched_clauses]
        clause_pos = sum(p for p, _ in counts)
        clause_neg = sum(n for _, n in counts)

        # Quy tắc đặc thù cho 'độ ồn': nếu xuất hiện từ 'noisy/loud/lärm/bruit/ruido' -> ưu tiên chê
        has_noise_complaint = any(
            re.search(r"\b(noisy|noise|loud|lärm|laut|bruit|bruyant|ruido|ruidoso)\b", c, re.I)
            for c in matched_clauses
        )
        has_noise_negation = any(
            re.search(r"\b(no noise|not noisy|very quiet|super quiet|kein lärm)\b", c, re.I)
            for c in matched_clauses
        )

        if aspect == "độ ồn" and has_noise_complaint and not has_noise_negation:
            neg_aspects.append(aspect)
        elif clause_neg > clause_pos:
            neg_aspects.append(aspect)
        elif clause_neg == clause_pos and clause_neg > 0:
            continue  # khen và chê ngang nhau: không đủ căn cứ, bỏ qua thay vì mặc định là khen
        else:
            pos_aspects.append(aspect)

    return ", ".join(pos_aspects), ", ".join(neg_aspects)

def predict_single_review_baseline(text: object, valid_aspects: list[str] | None = None) -> dict[str, str]:
    """Trích xuất đủ 4 trường (sentiment, aspects_pos, aspects_neg, lang) cho 1 bình luận."""
    sentiment = classify_sentiment_rule(text)
    asp_pos, asp_neg = extract_aspects_rule(text, valid_aspects)
    lang = detect_language_rule(text)

    # Đồng bộ logic: nếu không có từ khen/chê rõ ràng nhưng có khía cạnh bị chê -> chuyển sang mixed/negative
    if asp_neg and asp_pos and sentiment == "positive":
        sentiment = "mixed"
    elif asp_neg and not asp_pos and sentiment == "positive":
        sentiment = "negative"

    return {
        "base_sentiment": sentiment,
        "base_aspects_pos": asp_pos,
        "base_aspects_neg": asp_neg,
        "base_lang": lang,
    }

def load_llm_sample_reviews(snapshot_str: str, drop_flagged: bool = True) -> pd.DataFrame:
    """Đọc bảng reviews ở mốc cuối và lọc đúng các dòng thuộc Mẫu LLM ('x').
    drop_flagged=False giữ nguyên cả dòng bị cờ QA (dùng để tạo đủ mẫu nhãn tay)."""
    rev_path = Path("data/processed") / snapshot_str / "reviews.parquet"
    if not rev_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp {rev_path}. Hãy chạy filter.py và qa.py trước.")

    df = pd.read_parquet(rev_path)
    if "Mẫu LLM" in df.columns:
        df = df[df["Mẫu LLM"].astype(str).str.strip().str.lower() == "x"].copy()

    # Loại bỏ dòng bị thiếu nội dung và review do Airbnb tự sinh nếu có cờ QA
    if drop_flagged:
        for flag_col in ["flag_thieu_noi_dung", "flag_review_tu_dong"]:
            if flag_col in df.columns:
                df = df[~df[flag_col]].copy()

    return df.reset_index(drop=True)

def add_vectorized_aspect_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các cột cờ khía cạnh bằng .str.contains(na=False) """
    df_out = df.copy()
    comments_s = df_out["comments"].fillna("").astype(str) if "comments" in df_out.columns else pd.Series("", index=df_out.index)
    for aspect, pat in ASPECT_PATTERNS.items():
        col_name = f"has_{aspect.replace(' ', '_')}"
        df_out[col_name] = comments_s.str.contains(pat.pattern, case=False, regex=True, na=False)
    return df_out

def apply_baseline_to_dataframe(df: pd.DataFrame, valid_aspects: list[str] | None = None) -> pd.DataFrame:
    """Áp dụng hàm trích xuất Baseline lên toàn bộ bảng đánh giá Mẫu LLM."""
    df_out = add_vectorized_aspect_flags(df)
    if "comments" not in df_out.columns:
        for col in ["base_sentiment", "base_aspects_pos", "base_aspects_neg", "base_lang"]:
            df_out[col] = ""
        return df_out

    extracted_records = [
        predict_single_review_baseline(txt, valid_aspects)
        for txt in df_out["comments"]
    ]
    extracted_df = pd.DataFrame(extracted_records, index=df_out.index)
    for col in extracted_df.columns:
        df_out[col] = extracted_df[col]
    return df_out

def warn_unknown_aspects(valid_aspects: list[str]) -> None:
    """Cảnh báo nếu config có khía cạnh mà baseline không có từ điển (sẽ bị bỏ qua âm thầm)."""
    unknown = set(valid_aspects) - set(ASPECT_PATTERNS)
    if unknown:
        print(f"    [!] Khía cạnh trong config chưa có từ điển baseline: {sorted(unknown)}")

def init_hand_label_file_if_missing(df_baseline: pd.DataFrame, out_path: str | Path = "data/samples/nhan_tay_100.csv") -> None:
    """Tạo tệp mẫu 100 câu cần Nhãn tay (cột human_* để TRỐNG). Không điền sẵn dự đoán của baseline
    để tránh thiên lệch neo; không ghi đè nếu tệp đã tồn tại."""
    target = Path(out_path)
    if target.exists():
        print(f"    [i] Đã có sẵn tệp nhãn tay tại {target} (giữ nguyên, không ghi đè).")
        return

    if "Nhãn tay" not in df_baseline.columns:
        return

    hand_df = df_baseline[df_baseline["Nhãn tay"].astype(str).str.strip().str.lower() == "x"].copy()
    for col in ["human_sentiment", "human_aspects_pos", "human_aspects_neg", "human_lang"]:
        hand_df[col] = ""

    keep_cols = [
        c for c in [
            "id", "listing_id", "date", "comments",
            "human_sentiment", "human_aspects_pos", "human_aspects_neg", "human_lang"
        ] if c in hand_df.columns
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    hand_df[keep_cols].to_csv(target, index=False, encoding="utf-8-sig")
    print(f"    [+] Đã khởi tạo tệp {target} ({len(hand_df)} dòng) với cột human_* TRỐNG. Hãy gán nhãn mù, rồi commit tệp.")

def run_baseline(config_path: str | Path = "configs/vienna.yaml") -> pd.DataFrame:
    """Chạy phương pháp đối chứng (Baseline) trên các mẫu đánh giá và lưu kết quả."""
    print("--- BẮT ĐẦU CHẠY PHƯƠNG PHÁP ĐỐI CHỨNG (BASELINE - REGEX/KEYWORDS) ---")
    config = load_config(config_path)
    last_snapshot = str(config["snapshots"][-1])
    valid_aspects = config.get("llm", {}).get("aspects", list(ASPECT_PATTERNS.keys()))
    warn_unknown_aspects(valid_aspects)

    # Đọc dòng Mẫu LLM từ mốc chụp cuối cùng
    df_sample = load_llm_sample_reviews(last_snapshot)
    print(f"[*] Đã nạp {len(df_sample)} bình luận thuộc Mẫu LLM từ mốc {last_snapshot}.")

    # Chạy trích xuất bằng bộ quy tắc từ khóa / biểu thức chính quy
    df_res = apply_baseline_to_dataframe(df_sample, valid_aspects)

    # Lưu kết quả Baseline ra data/processed/
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_parquet(out_dir / "baseline_results.parquet", index=False)
    df_res.to_csv(out_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")

    # Khởi tạo tệp nhan_tay_100.csv (trống nhãn) nếu chưa có; dùng bảng CHƯA lọc cờ để đủ mọi dòng giảng viên chọn
    init_hand_label_file_if_missing(load_llm_sample_reviews(last_snapshot, drop_flagged=False), "data/samples/nhan_tay_100.csv")

    # In thống kê nhanh để kiểm chứng
    print(f"\n--- THỐNG KÊ NHANH KẾT QUẢ BASELINE ({len(df_res)} MẪU) ---")
    print("1. Phân bổ cảm xúc (base_sentiment):")
    print(df_res["base_sentiment"].value_counts().to_dict())
    print("2. Phân bổ ngôn ngữ (base_lang):")
    print(df_res["base_lang"].value_counts().to_dict())

    aspect_cols = [c for c in df_res.columns if c.startswith("has_")]
    if aspect_cols:
        rates = (df_res[aspect_cols].mean() * 100).round(1).to_dict()
        print("3. Tỷ lệ bình luận nhắc tới từng khía cạnh (%):")
        print(rates)

    print("\n HOÀN TẤT BASELINE! Đã lưu tại data/processed/baseline_results.parquet")
    return df_res

if __name__ == "__main__":
    run_baseline()
