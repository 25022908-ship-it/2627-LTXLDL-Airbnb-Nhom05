import pandas as pd
import yaml
from pathlib import Path

# Định nghĩa điều kiện, lý do và hành động cho từng quy tắc để xuất vào reports/qa_report.csv
RULE_DOCS: dict[str, tuple[str, str, str]] = {
    "flag_thieu_gia": ("price bị trống", "Không tính được KPI giá", "Gắn cờ; loại khỏi KPI giá"),
    "flag_sai_kieu_gia": ("price có giá trị nhưng không ép được sang số", "Chuỗi giá sai định dạng", "Gắn cờ; loại khỏi KPI giá"),
    "flag_gia_ngoai_lai": ("price_num <= 0 hoặc > price_max_eur", "Giá vô lý làm lệch trung bình", "Gắn cờ; loại khỏi KPI giá"),
    "flag_dem_ngoai_lai": ("minimum_nights <= 0 hoặc > min_nights_cap", "Số đêm tối thiểu phi thực tế", "Gắn cờ; loại khỏi KPI"),
    "flag_sai_suc_chua": ("accommodates <= 0 hoặc > 50; bedrooms < 0 hoặc > 20", "Sức chứa phi logic", "Gắn cờ; loại khỏi KPI"),
    "flag_sai_logic_review": ("Điểm ngoài [0,5] hoặc có điểm khi number_of_reviews = 0", "Mâu thuẫn dữ liệu đánh giá", "Gắn cờ; loại khỏi KPI"),
    "flag_sai_thong_tin_host": ("Thiếu host_id hoặc calculated_host_listings_count < 1", "Không xác định được chủ nhà", "Gắn cờ; loại khỏi KPI"),
    "flag_trung_id": ("Trùng khóa định danh (dedupe_keys trong config; id với reviews)", "Đếm lặp một đối tượng", "Gắn cờ; loại khỏi KPI"),
    "flag_lech_loai_phong": ("room_type ngoài 4 loại chuẩn", "Danh mục lệch", "Gắn cờ; loại khỏi KPI"),
    "flag_lech_id_giua_cac_moc": ("id không có ở mốc đầu tiên", "Listing mới xuất hiện sau mốc đầu (không phải lỗi)", "Chỉ ghi nhận, KHÔNG loại khỏi KPI"),
    "flag_thieu_available": ("available bị trống", "Không biết ngày trống hay đã khóa", "Gắn cờ; loại khỏi KPI"),
    "flag_sai_kieu_ngay": ("Cột ngày có dữ liệu nhưng không ép được sang datetime", "Ngày sai định dạng", "Gắn cờ; loại khỏi KPI"),
    "flag_sai_logic_dem": ("minimum_nights <= 0 hoặc > maximum_nights", "Ràng buộc số đêm phi logic", "Gắn cờ; loại khỏi KPI"),
    "flag_trung_ngay_phong": ("Trùng (listing_id, date)", "Một ngày bị đếm nhiều lần", "Gắn cờ; loại khỏi KPI"),
    "flag_lech_available": ("available ngoài {t, f}", "Danh mục lệch", "Gắn cờ; loại khỏi KPI"),
    "flag_thieu_noi_dung": ("comments trống", "Không có văn bản để phân tích", "Gắn cờ; loại khỏi mẫu LLM/baseline"),
    "flag_ngay_tuong_lai": ("Ngày review sau ngày chụp dữ liệu", "Ngày không thể xảy ra", "Gắn cờ; loại khỏi KPI"),
    "flag_review_tu_dong": ("comments chứa thông báo tự động của Airbnb (EN: 'automated posting')", "Không phải đánh giá của khách, làm lệch cảm xúc", "Gắn cờ; loại khỏi mẫu LLM/baseline"),
    "flag_lech_review_giua_cac_moc": ("Cùng review id nhưng listing_id khác với mốc đầu", "Dữ liệu không nhất quán giữa các mốc", "Chỉ ghi nhận, KHÔNG loại khỏi KPI"),
    "flag_ngoai_toa_do": ("Tọa độ thiếu hoặc nằm ngoài khung địa lý thành phố", "Vị trí sai/không đáng tin", "Gắn cờ; loại khỏi KPI"),
    "flag_lech_vung": ("Thiếu neighbourhood_cleansed hoặc analysis_area ngoài danh mục", "Không gán được vùng phân tích", "Gắn cờ; loại khỏi KPI"),
}

AUTO_REVIEW_PATTERN = r"automated posting"

REQUIRED_COLS = {
    "listings": ["id", "price", "room_type", "minimum_nights"],
    "calendar": ["listing_id", "date", "available", "minimum_nights", "maximum_nights"],
    "reviews": ["id", "listing_id", "date", "comments"],
}

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def empty_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index, dtype=bool)

def clean_price(val: object) -> float | None:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        cleaned = str(val).strip().replace("$", "").replace(",", "")
        return float(cleaned) if cleaned else None
    except (ValueError, AttributeError, TypeError):
        return None

def add_price_num_column(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    df_out = df.copy()
    if price_col not in df_out.columns:
        df_out["price_num"] = pd.Series(float("nan"), index=df_out.index, dtype="float64")
        return df_out
    df_out["price_num"] = pd.to_numeric(df_out[price_col].map(clean_price), errors="coerce")
    return df_out

def add_clean_date_column(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    fmt: str = "%Y-%m-%d",
) -> pd.DataFrame:
    df_out = df.copy()
    if date_col not in df_out.columns:
        df_out[target_col] = pd.Series(pd.NaT, index=df_out.index, dtype="datetime64[ns]")
        return df_out
    df_out[target_col] = pd.to_datetime(df_out[date_col], format=fmt, errors="coerce")
    return df_out

def check_missing_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return empty_mask(df)
    return df[col].isna()

def check_missing_text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return empty_mask(df)
    return df[col].isna() | (df[col].astype(str).str.strip() == "")

def check_wrong_type_after_parse(df: pd.DataFrame, raw_col: str, parsed_col: str) -> pd.Series:
    # Cột gốc có dữ liệu nhưng sau khi ép kiểu lại ra NaN/NaT
    if raw_col not in df.columns or parsed_col not in df.columns:
        return empty_mask(df)
    return df[raw_col].notna() & df[parsed_col].isna()

def check_out_of_bounds_num(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    allow_zero: bool = False,
) -> pd.Series:
    if col not in df.columns:
        return empty_mask(df)
    num_s = pd.to_numeric(df[col], errors="coerce")
    low_cond = (num_s < min_val) if allow_zero else (num_s <= min_val)
    return num_s.notna() & (low_cond | (num_s > max_val))

def check_coordinates_bounds(
    df: pd.DataFrame,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> pd.Series:
    if not {"latitude", "longitude"}.issubset(df.columns):
        return empty_mask(df)
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    missing = lat.isna() | lon.isna()
    if lat_bounds is None or lon_bounds is None:
        return missing
    out = ~lat.between(lat_bounds[0], lat_bounds[1]) | ~lon.between(lon_bounds[0], lon_bounds[1])
    return missing | out

def check_capacity_logic(df: pd.DataFrame) -> pd.Series:
    bad_acc = check_out_of_bounds_num(df, "accommodates", min_val=0, max_val=50, allow_zero=False)
    bad_bed = check_out_of_bounds_num(df, "bedrooms", min_val=0, max_val=20, allow_zero=True)
    return bad_acc | bad_bed

def check_review_score_logic(df: pd.DataFrame) -> pd.Series:
    bad_score = check_out_of_bounds_num(df, "review_scores_rating", min_val=0, max_val=5, allow_zero=True)
    if {"number_of_reviews", "review_scores_rating"}.issubset(df.columns):
        num_rev = pd.to_numeric(df["number_of_reviews"], errors="coerce")
        contradiction = (num_rev < 0) | ((num_rev == 0) & df["review_scores_rating"].notna())
        return bad_score | contradiction.fillna(False)
    return bad_score

def check_host_logic(df: pd.DataFrame) -> pd.Series:
    missing_host = check_missing_col(df, "host_id")
    bad_count = check_out_of_bounds_num(df, "calculated_host_listings_count", min_val=1, max_val=10000, allow_zero=True)
    return missing_host | bad_count

def check_min_max_nights_logic(df: pd.DataFrame) -> pd.Series:
    if not {"minimum_nights", "maximum_nights"}.issubset(df.columns):
        return empty_mask(df)
    min_n = pd.to_numeric(df["minimum_nights"], errors="coerce")
    max_n = pd.to_numeric(df["maximum_nights"], errors="coerce")
    return (min_n <= 0) | (min_n > max_n)

def check_future_date(df: pd.DataFrame, clean_date_col: str, snapshot_str: str) -> pd.Series:
    if clean_date_col not in df.columns:
        return empty_mask(df)
    snapshot_dt = pd.to_datetime(snapshot_str, format="%Y-%m-%d", errors="coerce")
    if pd.isna(snapshot_dt):
        return empty_mask(df)
    return df[clean_date_col].notna() & (df[clean_date_col] > snapshot_dt)

def check_duplicates(df: pd.DataFrame, subset_cols: list[str]) -> pd.Series:
    if not set(subset_cols).issubset(df.columns):
        return empty_mask(df)
    return df.duplicated(subset=subset_cols, keep=False)

def check_invalid_category(df: pd.DataFrame, col: str, valid_values: list | set) -> pd.Series:
    if col not in df.columns:
        return empty_mask(df)
    return ~df[col].isin(valid_values)

def check_auto_generated_review(df: pd.DataFrame) -> pd.Series:
    if "comments" not in df.columns:
        return empty_mask(df)
    return df["comments"].fillna("").astype(str).str.contains(AUTO_REVIEW_PATTERN, case=False, regex=True)

def check_review_listing_shift(df: pd.DataFrame, base_reviews: dict | None) -> pd.Series:
    # Kiểm tra cùng review id nhưng bị gắn sang listing_id khác so với mốc đầu
    if not base_reviews or not {"id", "listing_id"}.issubset(df.columns):
        return empty_mask(df)
    shifted = [
        (rid in base_reviews and base_reviews[rid] != lid)
        for rid, lid in zip(df["id"], df["listing_id"])
    ]
    return pd.Series(shifted, index=df.index, dtype=bool)

def build_listings_rules(df: pd.DataFrame, qa_cfg: dict, base_ids: set | None = None) -> dict[str, pd.Series]:
    price_max = float(qa_cfg.get("price_max_eur", 5000))
    min_nights_cap = float(qa_cfg.get("min_nights_cap", 365))
    dedupe_keys = qa_cfg.get("dedupe_keys", ["id"])
    valid_rooms = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    lat_b = qa_cfg.get("lat_bounds")
    lon_b = qa_cfg.get("lon_bounds")
    valid_areas = qa_cfg.get("valid_areas")

    area_invalid = check_invalid_category(df, "analysis_area", valid_areas) if valid_areas else empty_mask(df)

    rules = {
        "flag_thieu_gia": check_missing_col(df, "price"),
        "flag_sai_kieu_gia": check_wrong_type_after_parse(df, "price", "price_num"),
        "flag_gia_ngoai_lai": check_out_of_bounds_num(df, "price_num", 0, price_max, allow_zero=False),
        "flag_dem_ngoai_lai": check_out_of_bounds_num(df, "minimum_nights", 0, min_nights_cap, allow_zero=False),
        "flag_ngoai_toa_do": check_coordinates_bounds(df, lat_b, lon_b),
        "flag_sai_suc_chua": check_capacity_logic(df),
        "flag_sai_logic_review": check_review_score_logic(df),
        "flag_sai_thong_tin_host": check_host_logic(df),
        "flag_trung_id": check_duplicates(df, dedupe_keys),
        "flag_lech_loai_phong": check_invalid_category(df, "room_type", valid_rooms),
        "flag_lech_vung": area_invalid | check_missing_col(df, "neighbourhood_cleansed"),
    }
    if base_ids is not None and "id" in df.columns:
        rules["flag_lech_id_giua_cac_moc"] = ~df["id"].isin(base_ids)
    else:
        rules["flag_lech_id_giua_cac_moc"] = empty_mask(df)
    return rules

def build_calendar_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "flag_thieu_available": check_missing_col(df, "available"),
        "flag_sai_kieu_ngay": check_wrong_type_after_parse(df, "date", "date_clean"),
        "flag_sai_logic_dem": check_min_max_nights_logic(df),
        "flag_trung_ngay_phong": check_duplicates(df, ["listing_id", "date"]),
        "flag_lech_available": check_invalid_category(df, "available", ["t", "f"]),
    }

def build_reviews_rules(df: pd.DataFrame, snapshot_str: str, base_reviews: dict | None = None) -> dict[str, pd.Series]:
    return {
        "flag_thieu_noi_dung": check_missing_text(df, "comments"),
        "flag_sai_kieu_ngay": check_wrong_type_after_parse(df, "date", "date_clean"),
        "flag_ngay_tuong_lai": check_future_date(df, "date_clean", snapshot_str),
        "flag_trung_id": check_duplicates(df, ["id"]),
        "flag_review_tu_dong": check_auto_generated_review(df),
        "flag_lech_review_giua_cac_moc": check_review_listing_shift(df, base_reviews),
    }

def attach_qa_flags(
    df: pd.DataFrame,
    rules: dict[str, pd.Series],
    price_flags: list[str] | None = None,
    info_flags: list[str] | None = None,
) -> pd.DataFrame:
    # Gắn cờ cho từng quy tắc; info_flags chỉ dùng để ghi nhận chứ không loại dòng khỏi is_clean
    df_out = df.copy()
    price_flag_set = set(price_flags or [])
    info_flag_set = set(info_flags or [])

    structural_clean = pd.Series(True, index=df_out.index, dtype=bool)
    price_clean = pd.Series(True, index=df_out.index, dtype=bool)

    for rule_name, mask in rules.items():
        df_out[rule_name] = mask.fillna(False).astype(bool)
        if rule_name in info_flag_set:
            continue
        if rule_name in price_flag_set:
            price_clean = price_clean & (~df_out[rule_name])
        else:
            structural_clean = structural_clean & (~df_out[rule_name])

    df_out["is_clean"] = structural_clean
    if price_flag_set:
        df_out["is_clean_price"] = structural_clean & price_clean
    return df_out

def compute_listings_kpi_impact(df: pd.DataFrame) -> tuple[str, str]:
    if "price_num" not in df.columns or not df["price_num"].notna().any():
        return "NaN (NaN)", "NaN (NaN)"
    med_pre = round(float(df["price_num"].median()), 2)
    mean_pre = round(float(df["price_num"].mean()), 2)

    clean_col = "is_clean_price" if "is_clean_price" in df.columns else "is_clean"
    clean_s = df.loc[df[clean_col], "price_num"]
    if not clean_s.notna().any():
        return f"{med_pre} ({mean_pre})", "NaN (NaN)"
    med_post = round(float(clean_s.median()), 2)
    mean_post = round(float(clean_s.mean()), 2)
    return f"{med_pre} ({mean_pre})", f"{med_post} ({mean_post})"

def compute_calendar_kpi_impact(df: pd.DataFrame) -> tuple[str, str]:
    if "available" not in df.columns or len(df) == 0:
        return "NaN", "NaN"
    occ_pre = round(float((df["available"] == "f").mean() * 100), 2)
    clean_df = df.loc[df["is_clean"]]
    occ_post = round(float((clean_df["available"] == "f").mean() * 100), 2) if len(clean_df) > 0 else 0.0
    return str(occ_pre), str(occ_post)

def compute_reviews_kpi_impact(df: pd.DataFrame) -> tuple[str, str]:
    return str(len(df)), str(int(df["is_clean"].sum()))

def kpi_post_for_rule(compute_fn, df: pd.DataFrame, mask: pd.Series) -> str:
    keep = ~mask.fillna(False).astype(bool)
    tmp = df.copy()
    tmp["is_clean"] = keep
    if "is_clean_price" in tmp.columns:
        tmp["is_clean_price"] = keep
    return compute_fn(tmp)[1]

def compute_per_rule_kpi(
    compute_fn,
    df: pd.DataFrame,
    rules: dict[str, pd.Series],
    kpi_pre: str,
    info_flags: list[str] | None = None,
) -> dict[str, str]:
    # Các cờ chỉ ghi nhận (info_flags) không loại dòng nên giữ nguyên giá trị kpi_pre
    info_set = set(info_flags or [])
    return {
        name: (kpi_pre if name in info_set else kpi_post_for_rule(compute_fn, df, mask))
        for name, mask in rules.items()
    }

def build_report_rows(
    snapshot_str: str,
    table_name: str,
    rules: dict[str, pd.Series],
    kpi_name: str,
    kpi_pre: str,
    kpi_post: str,
    per_rule_post: dict[str, str] | None = None,
) -> list[dict]:
    rows = []
    for rule_name, mask in rules.items():
        dieu_kien, ly_do, hanh_dong = RULE_DOCS.get(rule_name, ("", "", ""))
        rows.append({
            "moc_thoi_gian": snapshot_str,
            "bang": table_name,
            "quy_tac": rule_name,
            "dieu_kien": dieu_kien,
            "ly_do": ly_do,
            "hanh_dong": hanh_dong,
            "so_dong_vi_pham": int(mask.sum()),
            "ty_le_phan_tram": round(float(mask.mean() * 100), 2) if len(mask) > 0 else 0.0,
            "kpi_ten": kpi_name,
            "kpi_truoc_qa": kpi_pre,
            "kpi_sau_qa": (per_rule_post or {}).get(rule_name, kpi_post),
        })
    return rows

def build_summary_row(
    snapshot_str: str,
    table_name: str,
    df: pd.DataFrame,
    kpi_name: str,
    kpi_pre: str,
    kpi_post: str,
) -> dict:
    clean_col = "is_clean_price" if "is_clean_price" in df.columns else "is_clean"
    removed = int((~df[clean_col]).sum())
    return {
        "moc_thoi_gian": snapshot_str,
        "bang": table_name,
        "quy_tac": "TONG_HOP_QA",
        "dieu_kien": f"Dòng vi phạm ít nhất một quy tắc loại dòng ({clean_col} = False)",
        "ly_do": "Tác động gộp của cả bộ quy tắc",
        "hanh_dong": "Loại khỏi KPI",
        "so_dong_vi_pham": removed,
        "ty_le_phan_tram": round(removed / len(df) * 100, 2) if len(df) > 0 else 0.0,
        "kpi_ten": kpi_name,
        "kpi_truoc_qa": kpi_pre,
        "kpi_sau_qa": kpi_post,
    }

def warn_if_mismatch(label: str, actual: object, expected: object) -> None:
    if expected is not None and actual != expected:
        print(f"    [!] Lệch kỳ vọng config - {label}: thực tế {actual} vs kỳ vọng {expected}")

def check_listings_expectations(df: pd.DataFrame, val_cfg: dict | None, snap_idx: int, is_last: bool) -> None:
    if not val_cfg:
        return
    warn_if_mismatch("số listing", len(df), val_cfg.get("expected_listings_per_snapshot"))

    no_rev = val_cfg.get("no_reviews_count")
    if no_rev and snap_idx < len(no_rev) and "number_of_reviews" in df.columns:
        n_zero = int((pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0) == 0).sum())
        warn_if_mismatch("số phòng chưa có review", n_zero, no_rev[snap_idx])

    if "room_type" in df.columns:
        actual_rooms = df["room_type"].value_counts().to_dict()
        for room, n_exp in (val_cfg.get("room_types") or {}).items():
            warn_if_mismatch(f"room_type '{room}'", int(actual_rooms.get(room, 0)), n_exp)

    if is_last and "price" in df.columns:
        warn_if_mismatch("số listing thiếu giá (mốc cuối)", int(df["price"].isna().sum()), val_cfg.get("missing_price_last_snapshot"))

    if "analysis_area" in df.columns:
        counts = df["analysis_area"].value_counts()
        warn_if_mismatch("số vùng", int(counts.size), val_cfg.get("expected_num_areas"))
        low, high = val_cfg.get("listings_per_area_range") or [None, None]
        if low is not None and high is not None:
            bad = counts[(counts < low) | (counts > high)]
            if not bad.empty:
                print(f"    [!] Lệch kỳ vọng config - vùng ngoài khoảng [{low}, {high}] listing: {bad.to_dict()}")

def validate_required_columns(df: pd.DataFrame, table_name: str) -> None:
    required = set(REQUIRED_COLS.get(table_name, []))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Bảng '{table_name}' thiếu các cột bắt buộc: {missing}")

def process_listings_qa(
    file_path: Path,
    snap_str: str,
    qa_cfg: dict,
    base_ids: set | None,
    val_cfg: dict | None = None,
    snap_idx: int = 0,
    is_last: bool = False,
) -> tuple[list[dict], set | None]:
    if not file_path.exists():
        return [], base_ids
    df = pd.read_parquet(file_path)
    validate_required_columns(df, "listings")
    df = add_price_num_column(df, "price")
    df = add_clean_date_column(df, "last_scraped", "last_scraped_clean")

    if base_ids is None and "id" in df.columns:
        base_ids = set(df["id"])

    rules = build_listings_rules(df, qa_cfg, base_ids)
    price_flags = ["flag_thieu_gia", "flag_sai_kieu_gia", "flag_gia_ngoai_lai"]
    info_flags = ["flag_lech_id_giua_cac_moc"]
    df = attach_qa_flags(df, rules, price_flags=price_flags, info_flags=info_flags)
    check_listings_expectations(df, val_cfg, snap_idx, is_last)

    kpi_pre, kpi_post = compute_listings_kpi_impact(df)
    per_rule = compute_per_rule_kpi(compute_listings_kpi_impact, df, rules, kpi_pre, info_flags=info_flags)
    df.to_parquet(file_path, index=False)
    print(f"    -> Listings: Thiếu giá = {int(rules['flag_thieu_gia'].sum())} | KPI Giá trước/sau = {kpi_pre} -> {kpi_post}")

    rows = build_report_rows(snap_str, "listings", rules, "price_median (mean)", kpi_pre, kpi_post, per_rule)
    rows.append(build_summary_row(snap_str, "listings", df, "price_median (mean)", kpi_pre, kpi_post))
    return rows, base_ids

def process_calendar_qa(file_path: Path, snap_str: str, val_cfg: dict | None = None) -> list[dict]:
    if not file_path.exists():
        return []
    df = pd.read_parquet(file_path)
    validate_required_columns(df, "calendar")
    df = add_clean_date_column(df, "date", "date_clean")
    if val_cfg:
        warn_if_mismatch("số dòng calendar", len(df), val_cfg.get("expected_calendar_rows_per_snapshot"))

    rules = build_calendar_rules(df)
    df = attach_qa_flags(df, rules)

    kpi_pre, kpi_post = compute_calendar_kpi_impact(df)
    per_rule = compute_per_rule_kpi(compute_calendar_kpi_impact, df, rules, kpi_pre)
    df.to_parquet(file_path, index=False)
    rows = build_report_rows(snap_str, "calendar", rules, "unavailable_rate_pct", kpi_pre, kpi_post, per_rule)
    rows.append(build_summary_row(snap_str, "calendar", df, "unavailable_rate_pct", kpi_pre, kpi_post))
    return rows

def process_reviews_qa(
    file_path: Path,
    snap_str: str,
    base_reviews: dict | None = None,
    val_cfg: dict | None = None,
) -> tuple[list[dict], dict | None]:
    if not file_path.exists():
        return [], base_reviews
    df = pd.read_parquet(file_path)
    validate_required_columns(df, "reviews")
    df = add_clean_date_column(df, "date", "date_clean")
    if val_cfg:
        warn_if_mismatch("số dòng reviews", len(df), val_cfg.get("expected_reviews_per_snapshot"))

    if base_reviews is None and {"id", "listing_id"}.issubset(df.columns):
        base_reviews = dict(zip(df["id"], df["listing_id"]))
        compare_with = None
    else:
        compare_with = base_reviews

    rules = build_reviews_rules(df, snap_str, compare_with)
    info_flags = ["flag_lech_review_giua_cac_moc"]
    df = attach_qa_flags(df, rules, info_flags=info_flags)

    kpi_pre, kpi_post = compute_reviews_kpi_impact(df)
    per_rule = compute_per_rule_kpi(compute_reviews_kpi_impact, df, rules, kpi_pre, info_flags=info_flags)
    df.to_parquet(file_path, index=False)
    rows = build_report_rows(snap_str, "reviews", rules, "valid_reviews_count", kpi_pre, kpi_post, per_rule)
    rows.append(build_summary_row(snap_str, "reviews", df, "valid_reviews_count", kpi_pre, kpi_post))
    return rows, base_reviews

def run_qa(config_path: str | Path = "configs/vienna.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    snapshots = config.get("snapshots", [])
    qa_cfg = config.get("qa", {})
    val_cfg = config.get("validation", {})

    all_report_rows = []
    base_listing_ids = None
    base_reviews = None

    for snap_idx, snapshot in enumerate(snapshots):
        snap_str = str(snapshot)
        print(f"- Đang kiểm định mốc: {snap_str}")
        proc_dir = Path("data/processed") / snap_str
        is_last = snap_idx == len(snapshots) - 1

        rows_l, base_listing_ids = process_listings_qa(
            proc_dir / "listings.parquet", snap_str, qa_cfg, base_listing_ids, val_cfg, snap_idx, is_last
        )
        rows_c = process_calendar_qa(proc_dir / "calendar.parquet", snap_str, val_cfg)
        rows_r, base_reviews = process_reviews_qa(proc_dir / "reviews.parquet", snap_str, base_reviews, val_cfg)

        all_report_rows.extend(rows_l + rows_c + rows_r)

    report_df = pd.DataFrame(all_report_rows)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(reports_dir / "qa_report.csv", index=False, encoding="utf-8-sig")
    print("\n[+] Hoàn tất QA! Đã lưu các tệp .parquet kèm cờ QA và xuất báo cáo ra reports/qa_report.csv")
    return report_df

if __name__ == "__main__":
    print("-" * 80)
    print("[Bước 2] Làm sạch và kiểm định chất lượng dữ liệu (QA)...")
    print("-" * 80)
    run_qa()
