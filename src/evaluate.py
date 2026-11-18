import json
import yaml
import pandas as pd
from pathlib import Path

VALID_SENTIMENTS = ["positive", "mixed", "negative"]
VALID_LANGS = ["en", "de", "fr", "es", "other"]

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    """Đọc tệp cấu hình YAML an toàn với bảng mã utf-8."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_label_str(val: object, default: str = "") -> str:
    """Chuẩn hóa nhãn đơn (sentiment, lang) về chữ thường; dữ liệu bẩn/NaN trả về default."""
    if pd.isna(val):
        return default
    cleaned = str(val).strip().lower()
    return cleaned if cleaned else default

def parse_aspect_set(val: object) -> set[str]:
    """Chuyển chuỗi khía cạnh ('vị trí, sạch sẽ') hoặc list thành tập hợp set chuẩn hóa."""
    if val is None or (not isinstance(val, (list, set, tuple)) and pd.isna(val)):
        return set()
    if isinstance(val, (list, set, tuple)):
        return {str(x).strip().lower() for x in val if str(x).strip()}
    text = str(val).strip()
    if not text or text.lower() == "nan":
        return set()
    return {x.strip().lower() for x in text.split(",") if x.strip()}

def build_combined_aspect_set(pos_val: object, neg_val: object, polar: bool = False) -> set[str]:
    """Gộp khía cạnh khen và chê thành 1 tập hợp (có thể gắn tiền tố pos:/neg: nếu polar=True)."""
    pos_set = parse_aspect_set(pos_val)
    neg_set = parse_aspect_set(neg_val)
    if polar:
        return {f"pos:{a}" for a in pos_set} | {f"neg:{a}" for a in neg_set}
    return pos_set | neg_set

def compute_accuracy(pred_s: pd.Series, true_s: pd.Series) -> float:
    """Tính độ chính xác (Accuracy) giữa hai cột nhãn đơn."""
    if len(true_s) == 0:
        return 0.0
    p_clean = pred_s.map(lambda x: normalize_label_str(x, ""))
    t_clean = true_s.map(lambda x: normalize_label_str(x, ""))
    return round(float((p_clean == t_clean).mean()), 3)

def compute_macro_f1(pred_s: pd.Series, true_s: pd.Series, labels: list[str]) -> float:
    """Macro-F1 trên các lớp nhãn đơn (bỏ qua lớp không xuất hiện ở cả dự đoán lẫn nhãn thật)."""
    p_clean = pred_s.map(lambda x: normalize_label_str(x, ""))
    t_clean = true_s.map(lambda x: normalize_label_str(x, ""))
    f1_list = []
    for lab in labels:
        tp = int(((p_clean == lab) & (t_clean == lab)).sum())
        fp = int(((p_clean == lab) & (t_clean != lab)).sum())
        fn = int(((p_clean != lab) & (t_clean == lab)).sum())
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_list.append(2 * tp / denom)
    return round(float(sum(f1_list) / len(f1_list)), 3) if f1_list else 0.0

def majority_baseline_accuracy(true_s: pd.Series) -> float:
    """Accuracy nếu luôn đoán lớp phổ biến nhất: mốc tối thiểu để so với accuracy của các phương pháp."""
    t_clean = true_s.map(lambda x: normalize_label_str(x, ""))
    if len(t_clean) == 0:
        return 0.0
    return round(float(t_clean.value_counts(normalize=True).iloc[0]), 3)

def build_confusion_table(pred_s: pd.Series, true_s: pd.Series, labels: list[str]) -> pd.DataFrame:
    """Ma trận nhầm lẫn: hàng = nhãn tay, cột = dự đoán."""
    p_clean = pred_s.map(lambda x: normalize_label_str(x, ""))
    t_clean = true_s.map(lambda x: normalize_label_str(x, ""))
    table = pd.crosstab(t_clean, p_clean, rownames=["nhan_tay"], colnames=["du_doan"])
    return table.reindex(index=labels, columns=labels, fill_value=0)

def compute_micro_f1_from_sets(pred_sets: list[set[str]], true_sets: list[set[str]]) -> tuple[float, float, float]:
    """Tính (Precision, Recall, Micro-F1) từ danh sách tập hợp dự đoán và nhãn thật."""
    tp = fp = fn = 0
    for p_set, t_set in zip(pred_sets, true_sets):
        tp += len(p_set & t_set)
        fp += len(p_set - t_set)
        fn += len(t_set - p_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 1.0
    return round(precision, 3), round(recall, 3), round(f1, 3)

def compute_aspect_f1(
    pred_pos_s: pd.Series,
    pred_neg_s: pd.Series,
    true_pos_s: pd.Series,
    true_neg_s: pd.Series,
    polar: bool = False,
) -> float:
    """Tính điểm Micro-F1 cho bài toán trích xuất khía cạnh trên toàn bộ bảng."""
    pred_sets = [build_combined_aspect_set(p, n, polar=polar) for p, n in zip(pred_pos_s, pred_neg_s)]
    true_sets = [build_combined_aspect_set(p, n, polar=polar) for p, n in zip(true_pos_s, true_neg_s)]
    _, _, f1 = compute_micro_f1_from_sets(pred_sets, true_sets)
    return f1

def load_usage_averages(usage_path: Path) -> dict[str, float] | None:
    """Trung bình token thực tế mỗi bình luận từ llm_usage.jsonl (None nếu chưa có log)."""
    if not usage_path.exists():
        return None
    n_rev = tok_in = tok_out = 0
    with open(usage_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                n_rev += int(rec["n_reviews"])
                tok_in += int(rec["prompt_tokens"])
                tok_out += int(rec["output_tokens"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    if n_rev == 0:
        return None
    return {"in": tok_in / n_rev, "out": tok_out / n_rev}

def count_city_reviews(raw_path: Path, chunksize: int = 500000) -> int | None:
    """Đếm tổng số review của thành phố ở mốc cuối từ bảng thô (None nếu chưa tải)."""
    if not raw_path.exists():
        return None
    total = 0
    for chunk in pd.read_csv(raw_path, compression="gzip", usecols=["id"], chunksize=chunksize):
        total += len(chunk)
    return total

def estimate_api_token_cost(
    df_sample: pd.DataFrame,
    usage_avg: dict[str, float] | None = None,
    total_reviews: int | None = None,
    price_in_per_1m: float = 0.25,
    price_out_per_1m: float = 1.50,
) -> dict[str, object]:
    """Ước tính token và chi phí API cho mẫu và cho TOÀN BỘ review của thành phố.
    Ưu tiên token thực tế từ usage log; nếu chưa có thì ước tính thô từ độ dài bình luận."""
    n_rows = max(len(df_sample), 1)
    if usage_avg:
        avg_in_tokens = round(usage_avg["in"], 1)
        avg_out_tokens = round(usage_avg["out"], 1)
        source = "thuc_te"
    else:
        avg_chars = float(df_sample["comments"].fillna("").astype(str).str.len().mean()) if "comments" in df_sample.columns else 400.0
        # 1 token ~ 4 ký tự + phần hướng dẫn prompt chia đều mỗi câu trong lô (ước tính thô)
        avg_in_tokens = round((avg_chars / 4.0) + 95.0, 1)
        avg_out_tokens = 36.0
        source = "uoc_tinh"

    cost_per_rev = (avg_in_tokens / 1_000_000) * price_in_per_1m + (avg_out_tokens / 1_000_000) * price_out_per_1m
    n_city = total_reviews if total_reviews else n_rows

    return {
        "so_cau_danh_gia": n_rows,
        "tong_review_thanh_pho": n_city,
        "nguon_token": source,
        "token_vao_tb_moi_cau": avg_in_tokens,
        "token_ra_tb_moi_cau": avg_out_tokens,
        "chi_phi_mau_usd": round(cost_per_rev * n_rows, 4),
        "chi_phi_toan_thanh_pho_usd": round(cost_per_rev * n_city, 2),
    }

def load_hand_labels(hand_path: str | Path = "data/samples/nhan_tay_100.csv") -> pd.DataFrame:
    """Đọc tệp 100 câu gán nhãn tay (Ground Truth) và chuẩn hóa tên cột khóa về 'id'."""
    path = Path(hand_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp nhãn tay tại {path}. Hãy chạy baseline.py để tạo tệp mẫu rồi gán nhãn.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "id" not in df.columns and "review_id" in df.columns:
        df = df.rename(columns={"review_id": "id"})
    return df.drop_duplicates(subset=["id"]).reset_index(drop=True)


def validate_hand_labels(df_human: pd.DataFrame, valid_aspects: list[str]) -> None:
    """Báo lỗi nếu nhãn tay còn trống hoặc có giá trị ngoài tập cho phép (gõ sai chính tả, sai khía cạnh...)."""
    problems = []
    for col in ["human_sentiment", "human_aspects_pos", "human_aspects_neg", "human_lang"]:
        if col not in df_human.columns:
            problems.append(f"Thiếu cột {col}")
    if problems:
        raise ValueError("; ".join(problems))

    sent = df_human["human_sentiment"].map(lambda x: normalize_label_str(x, ""))
    lang = df_human["human_lang"].map(lambda x: normalize_label_str(x, ""))
    if (sent == "").any():
        problems.append(f"{int((sent == '').sum())} dòng chưa gán human_sentiment (cần gán tay đủ trước khi đánh giá)")
    if (~sent.isin(VALID_SENTIMENTS + [""])).any():
        problems.append(f"human_sentiment ngoài {VALID_SENTIMENTS}: {sorted(set(sent) - set(VALID_SENTIMENTS) - {''})}")
    if (lang == "").any():
        problems.append(f"{int((lang == '').sum())} dòng chưa gán human_lang")
    if (~lang.isin(VALID_LANGS + [""])).any():
        problems.append(f"human_lang ngoài {VALID_LANGS}: {sorted(set(lang) - set(VALID_LANGS) - {''})}")

    allowed = {a.strip().lower() for a in valid_aspects}
    used = set()
    for pos, neg in zip(df_human["human_aspects_pos"], df_human["human_aspects_neg"]):
        used |= parse_aspect_set(pos) | parse_aspect_set(neg)
    if used - allowed:
        problems.append(f"Khía cạnh lạ trong nhãn tay: {sorted(used - allowed)}")

    if problems:
        raise ValueError("Nhãn tay không hợp lệ:\n - " + "\n - ".join(problems))

def load_predictions_dataframe(proc_dir: Path = Path("data/processed")) -> pd.DataFrame:
    """Tự động nạp và ghép kết quả dự đoán từ baseline_results và llm_results (hoặc file gộp)."""
    base_path = proc_dir / "baseline_results.parquet"
    llm_path = proc_dir / "llm_results.parquet"
    combined_path = proc_dir / "reviews_llm_extracted.parquet"

    if base_path.exists() and llm_path.exists():
        df_b = pd.read_parquet(base_path)
        df_l = pd.read_parquet(llm_path)
        llm_cols = [c for c in ["id", "llm_sentiment", "llm_aspects_pos", "llm_aspects_neg", "llm_lang"] if c in df_l.columns]
        return df_b.merge(df_l[llm_cols].drop_duplicates(subset=["id"]), on="id", how="left", validate="one_to_one")

    if combined_path.exists():
        return pd.read_parquet(combined_path)

    if base_path.exists():
        df_b = pd.read_parquet(base_path)
        for col in ["llm_sentiment", "llm_aspects_pos", "llm_aspects_neg", "llm_lang"]:
            df_b[col] = None
        return df_b

    raise FileNotFoundError("Chưa tìm thấy tệp kết quả dự đoán trong data/processed/. Hãy chạy baseline.py và llm.py trước.")

def merge_eval_dataset(df_human: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """Ghép bảng nhãn tay với bảng dự đoán của Baseline và LLM trên khóa 'id'."""
    human_cols = [c for c in ["id", "listing_id", "comments", "human_sentiment", "human_aspects_pos", "human_aspects_neg", "human_lang"] if c in df_human.columns]
    pred_cols = [
        c for c in [
            "id",
            "base_sentiment", "base_aspects_pos", "base_aspects_neg", "base_lang",
            "llm_sentiment", "llm_aspects_pos", "llm_aspects_neg", "llm_lang",
        ] if c == "id" or (c in df_pred.columns and c not in human_cols)
    ]

    merged = df_human[human_cols].merge(
        df_pred[pred_cols].drop_duplicates(subset=["id"]),
        on="id",
        how="inner",
        validate="one_to_one",
    )
    return merged

def build_eval_summary_table(df_eval: pd.DataFrame, llm_label: str = "LLM (Gemini)") -> pd.DataFrame:
    """Tính bảng điểm đối chiếu: mốc 'luôn đoán lớp phổ biến', Baseline và LLM."""
    valid_llm = df_eval[df_eval["llm_sentiment"].notna()].copy() if "llm_sentiment" in df_eval.columns else pd.DataFrame()
    eval_target = valid_llm if len(valid_llm) > 0 else df_eval

    rows = [{
        "Phương pháp": "Luôn đoán lớp cảm xúc phổ biến nhất (mốc tối thiểu)",
        "Số câu chấm": len(eval_target),
        "Cảm xúc (accuracy)": majority_baseline_accuracy(eval_target["human_sentiment"]),
        "Cảm xúc (macro-F1)": None,
        "Khía cạnh (F1)": None,
        "Khía cạnh kèm khen/chê (F1)": None,
        "Ngôn ngữ (accuracy)": None,
    }]

    # 1. Điểm của Baseline
    rows.append({
        "Phương pháp": "Đối chứng bằng từ khoá (Baseline)",
        "Số câu chấm": len(eval_target),
        "Cảm xúc (accuracy)": compute_accuracy(eval_target["base_sentiment"], eval_target["human_sentiment"]),
        "Cảm xúc (macro-F1)": compute_macro_f1(eval_target["base_sentiment"], eval_target["human_sentiment"], VALID_SENTIMENTS),
        "Khía cạnh (F1)": compute_aspect_f1(
            eval_target["base_aspects_pos"], eval_target["base_aspects_neg"],
            eval_target["human_aspects_pos"], eval_target["human_aspects_neg"],
            polar=False,
        ),
        "Khía cạnh kèm khen/chê (F1)": compute_aspect_f1(
            eval_target["base_aspects_pos"], eval_target["base_aspects_neg"],
            eval_target["human_aspects_pos"], eval_target["human_aspects_neg"],
            polar=True,
        ),
        "Ngôn ngữ (accuracy)": compute_accuracy(eval_target["base_lang"], eval_target["human_lang"]),
    })

    # 2. Điểm của LLM (nếu đã có kết quả LLM)
    if len(valid_llm) > 0:
        rows.append({
            "Phương pháp": llm_label,
            "Số câu chấm": len(valid_llm),
            "Cảm xúc (accuracy)": compute_accuracy(valid_llm["llm_sentiment"], valid_llm["human_sentiment"]),
            "Cảm xúc (macro-F1)": compute_macro_f1(valid_llm["llm_sentiment"], valid_llm["human_sentiment"], VALID_SENTIMENTS),
            "Khía cạnh (F1)": compute_aspect_f1(
                valid_llm["llm_aspects_pos"], valid_llm["llm_aspects_neg"],
                valid_llm["human_aspects_pos"], valid_llm["human_aspects_neg"],
                polar=False,
            ),
            "Khía cạnh kèm khen/chê (F1)": compute_aspect_f1(
                valid_llm["llm_aspects_pos"], valid_llm["llm_aspects_neg"],
                valid_llm["human_aspects_pos"], valid_llm["human_aspects_neg"],
                polar=True,
            ),
            "Ngôn ngữ (accuracy)": compute_accuracy(valid_llm["llm_lang"], valid_llm["human_lang"]),
        })

    return pd.DataFrame(rows)

def polar_aspect_sets(df: pd.DataFrame, pos_col: str, neg_col: str) -> list[set[str]]:
    """Danh sách tập khía cạnh có gắn khen/chê cho từng dòng (dùng so khớp với nhãn tay)."""
    return [build_combined_aspect_set(p, n, polar=True) for p, n in zip(df[pos_col], df[neg_col])]

def extract_error_cases(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất các dòng mà Baseline hoặc LLM dự đoán cảm xúc/khía cạnh lệch với Nhãn tay.
    Cột error_type để TRỐNG cho nhóm tự phân loại nguyên nhân (phủ định, mỉa mai, đa ngôn ngữ, nhập nhằng...)."""
    df_out = df_eval.copy()
    h_sent = df_out["human_sentiment"].map(normalize_label_str)
    b_sent = df_out["base_sentiment"].map(normalize_label_str)
    l_sent = df_out["llm_sentiment"].map(normalize_label_str) if "llm_sentiment" in df_out.columns else b_sent

    df_out["err_base_sentiment"] = b_sent != h_sent
    df_out["err_llm_sentiment"] = l_sent != h_sent
    df_out["disagree_base_vs_llm"] = b_sent != l_sent

    human_sets = polar_aspect_sets(df_out, "human_aspects_pos", "human_aspects_neg")
    base_sets = polar_aspect_sets(df_out, "base_aspects_pos", "base_aspects_neg")
    df_out["err_base_aspects"] = [b != h for b, h in zip(base_sets, human_sets)]
    if "llm_aspects_pos" in df_out.columns:
        llm_sets = polar_aspect_sets(df_out, "llm_aspects_pos", "llm_aspects_neg")
        df_out["err_llm_aspects"] = [l != h for l, h in zip(llm_sets, human_sets)]
    else:
        df_out["err_llm_aspects"] = False

    df_out["error_type"] = ""

    error_mask = (
        df_out["err_base_sentiment"] | df_out["err_llm_sentiment"] | df_out["disagree_base_vs_llm"]
        | df_out["err_base_aspects"] | df_out["err_llm_aspects"]
    )
    return df_out[error_mask].reset_index(drop=True)


def build_error_summary(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Bảng tóm tắt số câu sai theo từng phương pháp và từng loại nhãn (cảm xúc, khía cạnh)."""
    n = len(df_eval)
    h_sent = df_eval["human_sentiment"].map(normalize_label_str)
    human_sets = polar_aspect_sets(df_eval, "human_aspects_pos", "human_aspects_neg")
    rows = []
    for name, prefix in [("Baseline", "base"), ("LLM", "llm")]:
        if f"{prefix}_sentiment" not in df_eval.columns:
            continue
        sent_err = int((df_eval[f"{prefix}_sentiment"].map(normalize_label_str) != h_sent).sum())
        pred_sets = polar_aspect_sets(df_eval, f"{prefix}_aspects_pos", f"{prefix}_aspects_neg")
        asp_err = sum(1 for p, h in zip(pred_sets, human_sets) if p != h)
        lang_err = int((df_eval[f"{prefix}_lang"].map(normalize_label_str) != df_eval["human_lang"].map(normalize_label_str)).sum())
        rows.append({"Phương pháp": name, "Số câu": n, "Sai cảm xúc": sent_err, "Sai khía cạnh (có khen/chê)": asp_err, "Sai ngôn ngữ": lang_err})
    return pd.DataFrame(rows)

def run_evaluate(config_path: str | Path = "configs/vienna.yaml") -> pd.DataFrame:
    """Chạy toàn bộ quy trình chấm điểm, phân tích lỗi và ước tính chi phí API."""
    print("--- BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG LLM VS BASELINE TRÊN BỘ NHÃN TAY ---")
    config = load_config(config_path)
    llm_cfg = config.get("llm", {})
    valid_aspects = llm_cfg.get("aspects", [])
    model_label = f"LLM ({llm_cfg.get('model', 'gemini')})"
    last_snap = str(config["snapshots"][-1])

    # 1. Đọc nhãn tay (kiểm tra hợp lệ) và bảng kết quả dự đoán
    df_human = load_hand_labels("data/samples/nhan_tay_100.csv")
    validate_hand_labels(df_human, valid_aspects)
    expected_hand = llm_cfg.get("hand_label_size")
    if expected_hand is not None and len(df_human) != expected_hand:
        print(f"    [!] Số nhãn tay = {len(df_human)}, config.hand_label_size = {expected_hand}")
    df_pred = load_predictions_dataframe(Path("data/processed"))
    df_eval = merge_eval_dataset(df_human, df_pred)
    print(f"    -> Đã đối chiếu thành công {len(df_eval)} câu trong bộ Nhãn tay.")
    if len(df_eval) < len(df_human):
        print(f"    [!] {len(df_human) - len(df_eval)} câu có nhãn tay nhưng không có dự đoán (bị loại khỏi mẫu do cờ QA, vd review tự động); số câu chấm thực tế là {len(df_eval)}.")

    # Chỉ chấm các câu có kết quả LLM hợp lệ để mọi bảng (điểm, lỗi, ma trận) dùng chung một tập câu
    if "llm_sentiment" in df_eval.columns and df_eval["llm_sentiment"].notna().any():
        n_missing_llm = int(df_eval["llm_sentiment"].isna().sum())
        if n_missing_llm:
            print(f"    [!] Bỏ {n_missing_llm} câu chưa có kết quả LLM khỏi phần chấm điểm.")
        df_eval = df_eval[df_eval["llm_sentiment"].notna()].reset_index(drop=True)

    # 2. Tính bảng điểm Accuracy, Macro-F1 & Micro-F1
    eval_df = build_eval_summary_table(df_eval, model_label)

    print("BẢNG ĐÁNH GIÁ CHẤT LƯỢNG TRÊN BỘ NHÃN TAY:")

    print(eval_df.to_string(index=False))

    # 3. Lưu bảng điểm, ma trận nhầm lẫn, tóm tắt lỗi và bảng phân tích lỗi vào reports/
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(reports_dir / "llm_eval_report.csv", index=False, encoding="utf-8-sig")

    for prefix in ["base", "llm"]:
        col = f"{prefix}_sentiment"
        if col in df_eval.columns and df_eval[col].notna().any():
            confusion = build_confusion_table(df_eval[col], df_eval["human_sentiment"], VALID_SENTIMENTS)
            confusion.to_csv(reports_dir / f"confusion_{prefix}_sentiment.csv", encoding="utf-8-sig")
            print(f"\nMa trận nhầm lẫn cảm xúc ({prefix}) - hàng: nhãn tay, cột: dự đoán")
            print(confusion.to_string())

    summary_df = build_error_summary(df_eval)
    summary_df.to_csv(reports_dir / "llm_error_summary.csv", index=False, encoding="utf-8-sig")
    print("\nTÓM TẮT LỖI THEO PHƯƠNG PHÁP:")
    print(summary_df.to_string(index=False))

    error_df = extract_error_cases(df_eval)
    error_df.to_csv(reports_dir / "llm_error_analysis.csv", index=False, encoding="utf-8-sig")

    print(f"\n[*] PHÂN TÍCH LỖI (ERROR ANALYSIS): Tìm thấy {len(error_df)}/{len(df_eval)} câu có sự khác biệt.")
    print("    Hãy điền cột error_type trong reports/llm_error_analysis.csv rồi thống kê theo loại lỗi.")
    for _, row in error_df.head(5).iterrows():
        snippet = str(row.get("comments", "")).replace("\n", " ")[:100]
        print(f"    - ID {row['id']}: \"{snippet}...\"")
        print(
            f"      + Nhãn tay: {row.get('human_sentiment')} ({row.get('human_aspects_pos')} | -{row.get('human_aspects_neg')}) "
            f"| Baseline: {row.get('base_sentiment')} | LLM: {row.get('llm_sentiment')}"
        )

    # 4. Ước tính chi phí API cho toàn bộ review của thành phố
    cost_info = estimate_api_token_cost(
        df_pred,
        usage_avg=load_usage_averages(Path(llm_cfg.get("usage_log", "data/processed/llm_usage.jsonl"))),
        total_reviews=count_city_reviews(Path("data/raw") / last_snap / "reviews.csv.gz"),
        price_in_per_1m=float(llm_cfg.get("price_in_per_1m", 0.25)),
        price_out_per_1m=float(llm_cfg.get("price_out_per_1m", 1.50)),
    )
    print(f"\n[*] ƯỚC TÍNH CHI PHÍ TOKEN API ({llm_cfg.get('model', 'gemini')}; nguồn token: {cost_info['nguon_token']}):")
    print(f"    - Trung bình ~{cost_info['token_vao_tb_moi_cau']} token vào, ~{cost_info['token_ra_tb_moi_cau']} token ra / 1 đánh giá.")
    print(f"    - Chi phí cho {cost_info['so_cau_danh_gia']} mẫu của nhóm : ~${cost_info['chi_phi_mau_usd']:.4f} USD")
    print(f"    - Chi phí cho toàn bộ {cost_info['tong_review_thanh_pho']} review của thành phố: ~${cost_info['chi_phi_toan_thanh_pho_usd']:.2f} USD")
    pd.DataFrame([cost_info]).to_csv(reports_dir / "llm_cost_estimate.csv", index=False, encoding="utf-8-sig")

    print("\n[+] HOÀN TẤT ĐÁNH GIÁ! Đã xuất reports/llm_eval_report.csv, llm_error_analysis.csv, llm_error_summary.csv, confusion_*.csv")
    return eval_df

if __name__ == "__main__":
    run_evaluate()
