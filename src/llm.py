import os
import re
import json
import time
import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

VALID_SENTIMENTS = {"positive", "negative", "mixed"}
VALID_LANGS = {"en", "de", "fr", "es", "other"}

# Gợi ý ngắn cho từng khía cạnh chuẩn (chỉ đưa vào prompt các khía cạnh có trong config)
ASPECT_HINTS = {
    "vị trí": "khu vực, giao thông, khoảng cách đi bộ",
    "sạch sẽ": "vệ sinh, mùi, bụi bẩn",
    "chủ nhà": "giao tiếp, đón tiếp, hỗ trợ, check-in",
    "giá trị": "giá so với chất lượng",
    "tiện nghi": "thiết bị, giường, bếp, wifi, nước nóng",
    "độ ồn": "tiếng ồn hoặc sự yên tĩnh",
    "an toàn": "an ninh, khóa cửa, cảm giác an toàn",
}

class ReviewExtractionItem(BaseModel):
    """Schema kết quả trích xuất cho 1 bình luận đơn lẻ."""
    idx: int = Field(description="Số thứ tự (idx) của bình luận trong lô, copy đúng từ đầu vào")
    sentiment: Literal["positive", "negative", "mixed"] = Field(
        description="Cảm xúc tổng thể của bình luận"
    )
    aspects_pos: list[str] = Field(
        default_factory=list,
        description="Danh sách khía cạnh được khen (chỉ chọn trong danh mục cho phép)",
    )
    aspects_neg: list[str] = Field(
        default_factory=list,
        description="Danh sách khía cạnh bị chê (chỉ chọn trong danh mục cho phép)",
    )
    lang: Literal["en", "de", "fr", "es", "other"] = Field(
        description="Ngôn ngữ chính của bình luận"
    )


class BatchReviewExtraction(BaseModel):
    """Schema kết quả trả về cho cả một lô (batch) bình luận."""
    items: list[ReviewExtractionItem]

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    """Đọc tệp cấu hình YAML an toàn với bảng mã utf-8."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_comment_for_prompt(val: object, max_chars: int = 1200) -> str:
    """Làm sạch thẻ <br/> và cắt ngắn bình luận quá dài để tiết kiệm token."""
    if pd.isna(val):
        return ""
    try:
        text = re.sub(r"<br\s*/?>", " ", str(val), flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except (ValueError, AttributeError, TypeError):
        return ""

def sanitize_single_llm_record(raw_rec: dict, valid_aspects: list[str]) -> dict[str, object]:
    """Chuẩn hóa và hậu kiểm 1 bản ghi trả về từ LLM/cache. Sentiment không hợp lệ -> None (không ép thành positive)."""
    allowed_aspects = set(valid_aspects)

    raw_sent = str(raw_rec.get("sentiment", "")).strip().lower()
    sentiment = raw_sent if raw_sent in VALID_SENTIMENTS else None

    raw_lang = str(raw_rec.get("lang", "other")).strip().lower()
    lang = raw_lang if raw_lang in VALID_LANGS else "other"

    pos_raw = raw_rec.get("aspects_pos", [])
    if isinstance(pos_raw, str):
        pos_raw = [x.strip() for x in pos_raw.split(",") if x.strip()]
    pos_clean = [str(a).strip() for a in (pos_raw or []) if str(a).strip() in allowed_aspects]

    neg_raw = raw_rec.get("aspects_neg", [])
    if isinstance(neg_raw, str):
        neg_raw = [x.strip() for x in neg_raw.split(",") if x.strip()]
    neg_clean = [str(a).strip() for a in (neg_raw or []) if str(a).strip() in allowed_aspects]

    return {
        "review_id": int(raw_rec["review_id"]),
        "sentiment": sentiment,
        "aspects_pos": pos_clean,
        "aspects_neg": neg_clean,
        "lang": lang,
    }

def is_valid_llm_record(rec: dict) -> bool:
    """Bản ghi chỉ hợp lệ khi có sentiment thuộc tập cho phép."""
    return rec.get("sentiment") in VALID_SENTIMENTS

def resolve_gemini_model_name(cfg_model: str) -> str:
    """Dùng đúng tên model trong config (không đổi ngầm); nếu sai tên, API sẽ báo lỗi rõ ràng."""
    return str(cfg_model).strip()

def map_local_idx_to_review_id(raw_items: list[dict], batch_items: list[dict]) -> list[dict]:
    """Đổi idx cục bộ (0..n-1) LLM trả về thành review_id thật; bỏ idx không thuộc lô."""
    id_by_idx = {int(b["idx"]): int(b["review_id"]) for b in batch_items}
    mapped = []
    for item in raw_items:
        try:
            idx = int(item.get("idx"))
        except (TypeError, ValueError):
            continue
        if idx in id_by_idx:
            mapped.append({**item, "review_id": id_by_idx[idx]})
    return mapped

def log_llm_usage(usage_path: Path, model_name: str, n_reviews: int, response: object) -> None:
    """Ghi số token thực tế của một lần gọi API (dùng để ước tính chi phí chính xác)."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return
    rec = {
        "model": model_name,
        "n_reviews": n_reviews,
        "prompt_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
    }
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(usage_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def is_fatal_api_error(err: Exception) -> bool:
    """Lỗi cấu hình/xác thực (key sai, model sai) thì thử lại vô ích."""
    return getattr(err, "code", None) in (400, 401, 403, 404)

def load_llm_cache(cache_path: Path, valid_aspects: list[str]) -> dict[int, dict]:
    """Đọc bộ nhớ đệm llm_cache.jsonl (nếu có) và trả về từ điển {review_id: record}."""
    cache_map: dict[int, dict] = {}
    if not cache_path.exists():
        return cache_map

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
                clean_obj = sanitize_single_llm_record(obj, valid_aspects)
                if is_valid_llm_record(clean_obj):
                    cache_map[clean_obj["review_id"]] = clean_obj
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    return cache_map


def append_records_to_cache(cache_path: Path, records: list[dict]) -> None:
    """Ghi nối tiếp (append) danh sách bản ghi mới vào tệp llm_cache.jsonl."""
    if not records:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_aspect_guide(valid_aspects: list[str]) -> str:
    """Mô tả ngắn từng khía cạnh để LLM hiểu cùng một nghĩa với nhóm."""
    lines = [f"- {a}: {ASPECT_HINTS[a]}" for a in valid_aspects if a in ASPECT_HINTS]
    return "\n".join(lines)


def build_batch_prompt(batch_items: list[dict], valid_aspects: list[str]) -> str:
    """Tạo câu lệnh prompt rõ ràng cho một lô bình luận Airbnb. batch_items chỉ gồm idx + comment."""
    aspects_str = ", ".join(valid_aspects)
    reviews_json = json.dumps(batch_items, ensure_ascii=False)
    return (
        "Bạn là hệ thống phân tích đánh giá khách hàng Airbnb tại thành phố Vienna.\n"
        f"Chỉ được phép chọn khía cạnh nằm trong danh sách sau: [{aspects_str}].\n"
        f"Ý nghĩa các khía cạnh:\n{build_aspect_guide(valid_aspects)}\n"
        "Quy tắc phân loại:\n"
        "- sentiment: 'positive' (khen hoàn toàn/chủ đạo), 'negative' (chê chủ đạo), 'mixed' (vừa khen vừa chê rõ rệt).\n"
        "- aspects_pos: các khía cạnh được khách khen trong bình luận.\n"
        "- aspects_neg: các khía cạnh bị khách phàn nàn hoặc chê trong bình luận.\n"
        "- Chỉ ghi khía cạnh khi bình luận nhắc rõ ràng; không suy đoán. Khía cạnh vừa khen vừa chê thì ghi ở cả hai danh sách.\n"
        "- lang: mã ngôn ngữ chính của bình luận ('en', 'de', 'fr', 'es', 'other').\n"
        "- idx: copy đúng số idx của bình luận đầu vào, mỗi bình luận đúng một kết quả.\n"
        "Ví dụ: {\"idx\": 0, \"comment\": \"Great location but very noisy at night.\"} -> "
        "{\"idx\": 0, \"sentiment\": \"mixed\", \"aspects_pos\": [\"vị trí\"], \"aspects_neg\": [\"độ ồn\"], \"lang\": \"en\"}\n"
        "Hãy trả về kết quả JSON khớp với schema cho danh sách bình luận dưới đây:\n"
        f"{reviews_json}"
    )


def call_gemini_batch_with_retry(
    client: object,
    model_name: str,
    batch_items: list[dict],
    valid_aspects: list[str],
    max_retries: int = 3,
    usage_path: Path | None = None,
) -> list[dict]:
    """Gọi Gemini API cho 1 lô (batch_items: idx, review_id, comment) với retry; lỗi xác thực/model thì dừng ngay."""
    from google.genai import types

    prompt_items = [{"idx": b["idx"], "comment": b["comment"]} for b in batch_items]
    prompt = build_batch_prompt(prompt_items, valid_aspects)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchReviewExtraction,
                    temperature=0.0,
                ),
            )
            parsed = json.loads(response.text)
            raw_items = map_local_idx_to_review_id(parsed.get("items", []), batch_items)
            records = [sanitize_single_llm_record(item, valid_aspects) for item in raw_items]
            if usage_path is not None:
                log_llm_usage(usage_path, model_name, len(batch_items), response)
            return [r for r in records if is_valid_llm_record(r)]
        except Exception as err:
            if is_fatal_api_error(err):
                raise RuntimeError(
                    f"Lỗi API không thể thử lại (mã {getattr(err, 'code', '?')}): {err}. "
                    "Kiểm tra GEMINI_API_KEY và tên model trong config."
                ) from err
            wait_sec = 2 ** attempt
            print(f"      [!] Lỗi gọi API (lần {attempt}/{max_retries}): {err}. Đợi {wait_sec}s...")
            time.sleep(wait_sec)

    return []


def fetch_missing_reviews_via_api(
    df_sample: pd.DataFrame,
    cache_map: dict[int, dict],
    cache_path: Path,
    model_name: str,
    valid_aspects: list[str],
    batch_size: int = 20,
    usage_path: Path | None = None,
    max_rounds: int = 2,
) -> dict[int, dict]:
    """Tìm các bình luận chưa có trong cache, chia lô gọi Gemini API và lưu vào cache (chạy lại phần còn thiếu)."""
    missing_df = df_sample[~df_sample["id"].isin(cache_map.keys())].copy()
    if missing_df.empty:
        print(f"    [i] Toàn bộ {len(df_sample)} bình luận đã có sẵn trong cache. Không cần gọi lại API!")
        return cache_map

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu biến môi trường GEMINI_API_KEY trong tệp .env! "
            "Hãy tạo tệp .env hoặc chạy với tham số --skip-llm nếu đã có cache."
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    resolved_model = resolve_gemini_model_name(model_name)

    for round_no in range(1, max_rounds + 1):
        missing_df = df_sample[~df_sample["id"].isin(cache_map.keys())]
        if missing_df.empty:
            break
        print(f"    [*] Lượt {round_no}: cần gọi API cho {len(missing_df)} câu (Model: {resolved_model}, Batch size: {batch_size})...")

        rows = [
            {"review_id": int(r["id"]), "comment": clean_comment_for_prompt(r.get("comments", ""))}
            for _, r in missing_df.iterrows()
        ]

        total_batches = (len(rows) + batch_size - 1) // batch_size
        for b_idx in range(total_batches):
            chunk = rows[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch = [{"idx": i, **row} for i, row in enumerate(chunk)]
            print(f"      -> Đang xử lý lô {b_idx + 1}/{total_batches} ({len(batch)} câu)...")

            batch_results = call_gemini_batch_with_retry(
                client, resolved_model, batch, valid_aspects, usage_path=usage_path
            )
            if batch_results:
                append_records_to_cache(cache_path, batch_results)
                for rec in batch_results:
                    cache_map[rec["review_id"]] = rec

            # Nghỉ nhẹ giữa các lô để tránh vượt giới hạn Rate Limit (RPM)
            if b_idx < total_batches - 1:
                time.sleep(1.0)

    return cache_map

def load_llm_sample_reviews(snapshot_str: str) -> pd.DataFrame:
    """Đọc bảng reviews ở mốc cuối và lọc đúng các dòng thuộc Mẫu LLM ('x')."""
    rev_path = Path("data/processed") / snapshot_str / "reviews.parquet"
    if not rev_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {rev_path}. Hãy chạy filter.py và qa.py trước.")

    df = pd.read_parquet(rev_path)
    if "Mẫu LLM" in df.columns:
        df = df[df["Mẫu LLM"].astype(str).str.strip().str.lower() == "x"].copy()
    # Loại dòng thiếu nội dung và review do Airbnb tự sinh nếu có cờ QA
    for flag_col in ["flag_thieu_noi_dung", "flag_review_tu_dong"]:
        if flag_col in df.columns:
            df = df[~df[flag_col]].copy()
    return df.reset_index(drop=True)

def attach_llm_predictions(df: pd.DataFrame, cache_map: dict[int, dict], valid_aspects: list[str]) -> pd.DataFrame:
    """Ghép kết quả từ cache_map vào DataFrame. Dòng chưa có kết quả để trống và llm_ok=False (không điền giả)."""
    df_out = df.copy()
    sent_list, pos_list, neg_list, lang_list, ok_list = [], [], [], [], []

    for rev_id in df_out["id"]:
        rec = cache_map.get(int(rev_id))
        if rec and is_valid_llm_record(rec):
            sent_list.append(rec["sentiment"])
            pos_list.append(", ".join(rec["aspects_pos"]))
            neg_list.append(", ".join(rec["aspects_neg"]))
            lang_list.append(rec["lang"])
            ok_list.append(True)
        else:
            sent_list.append(None)
            pos_list.append("")
            neg_list.append("")
            lang_list.append(None)
            ok_list.append(False)

    df_out["llm_sentiment"] = sent_list
    df_out["llm_aspects_pos"] = pos_list
    df_out["llm_aspects_neg"] = neg_list
    df_out["llm_lang"] = lang_list
    df_out["llm_ok"] = ok_list

    # Tạo thêm các cột boolean llm_pos_<aspect> và llm_neg_<aspect> phục vụ tính KPI 5 & 6
    for aspect in valid_aspects:
        slug = aspect.replace(" ", "_")
        df_out[f"llm_pos_{slug}"] = [
            aspect in (cache_map.get(int(rid), {}).get("aspects_pos", []))
            for rid in df_out["id"]
        ]
        df_out[f"llm_neg_{slug}"] = [
            aspect in (cache_map.get(int(rid), {}).get("aspects_neg", []))
            for rid in df_out["id"]
        ]

    return df_out

def check_llm_coverage(df_llm: pd.DataFrame, min_coverage: float, strict: bool = True) -> float:
    """Tính tỷ lệ bình luận có kết quả LLM hợp lệ; thấp hơn ngưỡng thì báo lỗi (strict) hoặc cảnh báo."""
    coverage = float(df_llm["llm_ok"].mean()) if len(df_llm) > 0 else 0.0
    print(f"    -> Độ phủ LLM: {int(df_llm['llm_ok'].sum())}/{len(df_llm)} ({coverage * 100:.1f}%), ngưỡng tối thiểu {min_coverage * 100:.0f}%")
    if coverage < min_coverage:
        msg = f"Độ phủ LLM {coverage * 100:.1f}% thấp hơn ngưỡng {min_coverage * 100:.0f}%: KPI 5 & 6 sẽ thiếu dữ liệu."
        if strict:
            raise RuntimeError(msg)
        print(f"    [!] {msg}")
    return coverage

def warn_if_sample_mismatch(n_loaded: int, expected: int | None) -> None:
    """Cảnh báo nếu số bình luận mẫu khác sample_size trong config."""
    if expected is not None and n_loaded != expected:
        print(f"    [!] Số bình luận Mẫu LLM = {n_loaded}, config.sample_size = {expected}")

def run_llm(config_path: str | Path = "configs/vienna.yaml", skip_llm: bool = False, batch_size: int = 20) -> pd.DataFrame:
    """Điều phối toàn bộ bước trích xuất văn bản bằng LLM (hỗ trợ cache & --skip-llm)."""
    print(f"--- BẮT ĐẦU CHẠY LLM PIPELINE (skip_llm={skip_llm}) ---")
    config = load_config(config_path)
    last_snap = str(config["snapshots"][-1])

    llm_cfg = config.get("llm", {})
    model_name = llm_cfg.get("model", "gemini-2.5-flash-lite")
    valid_aspects = llm_cfg.get("aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])
    cache_path = Path(llm_cfg.get("cache", "data/processed/llm_cache.jsonl"))
    usage_path = Path(llm_cfg.get("usage_log", "data/processed/llm_usage.jsonl"))
    min_coverage = float(llm_cfg.get("min_coverage", 0.95))

    # 1. Đọc các bình luận thuộc Mẫu LLM
    df_sample = load_llm_sample_reviews(last_snap)
    print(f"    -> Đã nạp {len(df_sample)} bình luận Mẫu LLM từ mốc {last_snap}.")
    warn_if_sample_mismatch(len(df_sample), llm_cfg.get("sample_size"))

    # 2. Đọc bộ nhớ đệm hiện có
    cache_map = load_llm_cache(cache_path, valid_aspects)
    print(f"    -> Đã tải {len(cache_map)} bản ghi từ bộ nhớ đệm {cache_path}.")

    # 3. Gọi Gemini API cho các câu còn thiếu (nếu không bật cờ --skip-llm)
    if not skip_llm:
        cache_map = fetch_missing_reviews_via_api(
            df_sample=df_sample,
            cache_map=cache_map,
            cache_path=cache_path,
            model_name=model_name,
            valid_aspects=valid_aspects,
            batch_size=batch_size,
            usage_path=usage_path,
        )
    else:
        print("    [i] Đang chạy ở chế độ --skip-llm: Chỉ sử dụng dữ liệu từ cache, không gọi API.")

    # 4. Gắn các cột dự đoán vào bảng, kiểm tra độ phủ và lưu kết quả
    df_llm = attach_llm_predictions(df_sample, cache_map, valid_aspects)
    check_llm_coverage(df_llm, min_coverage, strict=not skip_llm)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_llm.to_parquet(out_dir / "llm_results.parquet", index=False)
    df_llm.to_csv(out_dir / "llm_results.csv", index=False, encoding="utf-8-sig")

    df_ok = df_llm[df_llm["llm_ok"]]
    print(f"    -> Phân bổ cảm xúc LLM : {df_ok['llm_sentiment'].value_counts().to_dict()}")
    print(f"    -> Phân bổ ngôn ngữ LLM: {df_ok['llm_lang'].value_counts().to_dict()}")
    print("\n[+] HOÀN TẤT LLM! Đã lưu cache tại data/processed/llm_cache.jsonl và bảng tại data/processed/llm_results.parquet")
    return df_llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy trích xuất văn bản bằng Gemini API")
    parser.add_argument("--config", default="configs/vienna.yaml", help="Đường dẫn tệp cấu hình YAML")
    parser.add_argument("--skip-llm", action="store_true", help="Bỏ qua gọi API, chỉ đọc từ tệp cache")
    parser.add_argument("--batch-size", type=int, default=20, help="Số lượng bình luận mỗi lần gọi API")
    args = parser.parse_args()

    run_llm(config_path=args.config, skip_llm=args.skip_llm, batch_size=args.batch_size)
