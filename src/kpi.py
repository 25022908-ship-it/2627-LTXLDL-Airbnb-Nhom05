import yaml
import pandas as pd
from pathlib import Path

MIN_N_PRICE = 10
MIN_N_TEXT = 30

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    """Đọc tệp cấu hình YAML an toàn với bảng mã utf-8."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_iqr(series: pd.Series) -> float:
    """Tính khoảng tứ phân vị IQR (Q75 - Q25) an toàn, bỏ qua NaN."""
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    if clean_s.empty:
        return 0.0
    return round(float(clean_s.quantile(0.75) - clean_s.quantile(0.25)), 2)


def ensure_price_num(df: pd.DataFrame) -> pd.DataFrame:
    """Đảm bảo bảng listings luôn có cột số price_num ngay cả khi chưa chạy qa.py."""
    df_out = df.copy()
    if "price_num" in df_out.columns:
        return df_out
    if "price" in df_out.columns:
        cleaned = df_out["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        df_out["price_num"] = pd.to_numeric(cleaned, errors="coerce")
    else:
        df_out["price_num"] = float("nan")
    return df_out


def filter_clean_rows(df: pd.DataFrame, require_price: bool = False) -> pd.DataFrame:
    """Lọc các dòng đạt chuẩn QA (is_clean_price nếu tính giá, hoặc is_clean chung)."""
    if require_price and "is_clean_price" in df.columns:
        return df[df["is_clean_price"]].copy()
    if "is_clean" in df.columns:
        out = df[df["is_clean"]].copy()
        if require_price and "price_num" in out.columns:
            out = out[out["price_num"].notna() & (out["price_num"] > 0)].copy()
        return out
    return df.copy()


def resolve_area_column(df: pd.DataFrame, preferred_col: str = "analysis_area") -> str:
    """Chọn cột vùng phân tích hợp lệ (ưu tiên cột trong config, dự phòng neighbourhood_cleansed)."""
    if preferred_col in df.columns:
        return preferred_col
    if "neighbourhood_cleansed" in df.columns:
        return "neighbourhood_cleansed"
    raise KeyError("Không tìm thấy cột khu vực (analysis_area hoặc neighbourhood_cleansed) trong bảng.")


def compute_kpi1_price_by_area(df_last_listings: pd.DataFrame, area_pref: str = "analysis_area") -> pd.DataFrame:
    """KPI 1: Thống kê giá thuê (median, mean, std, IQR, min, max) và loại phòng theo vùng ở mốc cuối."""
    df_clean = ensure_price_num(df_last_listings)
    df_clean = filter_clean_rows(df_clean, require_price=True)
    area_col = resolve_area_column(df_clean, area_pref)

    df_clean["is_entire_home"] = df_clean.get("room_type", "") == "Entire home/apt"

    grouped = (
        df_clean.groupby(area_col, as_index=False)
        .agg(
            so_phong_sach=("id", "count"),
            gia_trung_vi_eur=("price_num", "median"),
            gia_trung_binh_eur=("price_num", "mean"),
            do_lech_chuan_eur=("price_num", "std"),
            gia_iqr_eur=("price_num", compute_iqr),
            gia_thap_nhat_eur=("price_num", "min"),
            gia_cao_nhat_eur=("price_num", "max"),
            ty_le_entire_home_pct=("is_entire_home", lambda s: round(float(s.mean() * 100), 2)),
            diem_danh_gia_tb=("review_scores_rating", "mean"),
        )
        .sort_values(by="gia_trung_vi_eur", ascending=False)
        .reset_index(drop=True)
    )

    for col in ["gia_trung_vi_eur", "gia_trung_binh_eur", "do_lech_chuan_eur", "diem_danh_gia_tb"]:
        grouped[col] = grouped[col].round(2)
    return grouped

def compute_kpi4_host_professionalization(df_last_listings: pd.DataFrame, area_pref: str = "analysis_area") -> pd.DataFrame:
    """KPI 4: Phân tích tỷ trọng và mức giá giữa Chủ nhà cá nhân (1 phòng) và Chủ nhà chuyên nghiệp (>=2 phòng)."""
    df_clean = ensure_price_num(df_last_listings)
    df_clean = filter_clean_rows(df_clean, require_price=False)
    area_col = resolve_area_column(df_clean, area_pref)

    host_count = pd.to_numeric(df_clean.get("calculated_host_listings_count", 1), errors="coerce").fillna(1)
    df_clean["is_multi_host"] = host_count >= 2
    # Tỷ trọng tính trên mọi dòng sạch; còn so sánh giá chỉ dùng dòng sạch về giá (cùng chuẩn với KPI 1)
    df_clean["price_ok"] = df_clean["is_clean_price"] if "is_clean_price" in df_clean.columns else True

    rows = []
    for area_val, grp in df_clean.groupby(area_col):
        total_rooms = len(grp)
        multi_grp = grp[grp["is_multi_host"]]
        single_grp = grp[~grp["is_multi_host"]]

        multi_prices = multi_grp.loc[multi_grp["price_ok"] & multi_grp["price_num"].notna() & (multi_grp["price_num"] > 0), "price_num"]
        single_prices = single_grp.loc[single_grp["price_ok"] & single_grp["price_num"].notna() & (single_grp["price_num"] > 0), "price_num"]

        rows.append({
            area_col: area_val,
            "tong_so_phong": total_rooms,
            "so_chu_nha_duy_nhat": int(grp["host_id"].nunique()) if "host_id" in grp.columns else total_rooms,
            "so_phong_chu_nha_chuyen_nghiep": len(multi_grp),
            "ty_le_chuyen_nghiep_pct": round(len(multi_grp) / total_rooms * 100, 2) if total_rooms > 0 else 0.0,
            "gia_median_chuyen_nghiep_eur": round(float(multi_prices.median()), 2) if not multi_prices.empty else None,
            "gia_median_ca_nhan_eur": round(float(single_prices.median()), 2) if not single_prices.empty else None,
        })

    return pd.DataFrame(rows).sort_values(by="ty_le_chuyen_nghiep_pct", ascending=False).reset_index(drop=True)


def compute_kpi7_price_by_neighbourhood(df_last_listings: pd.DataFrame) -> pd.DataFrame:
    """KPI 7: Giá trung vị theo neighbourhood_cleansed ở mốc cuối (dữ liệu cho bản đồ choropleth)."""
    df_clean = filter_clean_rows(ensure_price_num(df_last_listings), require_price=True)
    if "neighbourhood_cleansed" not in df_clean.columns:
        return pd.DataFrame()

    grouped = (
        df_clean.groupby("neighbourhood_cleansed", as_index=False)
        .agg(
            so_phong_sach=("id", "count"),
            gia_trung_vi_eur=("price_num", "median"),
            gia_trung_binh_eur=("price_num", "mean"),
        )
        .sort_values(by="gia_trung_vi_eur", ascending=False)
        .reset_index(drop=True)
    )
    for col in ["gia_trung_vi_eur", "gia_trung_binh_eur"]:
        grouped[col] = grouped[col].round(2)
    grouped["canh_bao_n_nho"] = grouped["so_phong_sach"] < MIN_N_PRICE
    return grouped

def compute_kpi2_occupancy_trend(snapshots: list) -> pd.DataFrame:
    """KPI 2: Tỷ lệ ngày không khả dụng (available == 'f') qua các mốc và theo khu vực.
    Lưu ý: gồm cả ngày chủ nhà tự chặn và là lịch tương lai ~365 ngày kể từ ngày chụp, nên chỉ là chỉ báo gần đúng của mức lấp đầy."""
    rows = []
    for snap in snapshots:
        snap_str = str(snap)
        proc_dir = Path("data/processed") / snap_str
        cal_path = proc_dir / "calendar.parquet"
        list_path = proc_dir / "listings.parquet"
        if not cal_path.exists() or not list_path.exists():
            continue

        df_c = filter_clean_rows(pd.read_parquet(cal_path), require_price=False)
        df_l = pd.read_parquet(list_path)[["id", "analysis_area"]].drop_duplicates(subset=["id"])

        merged = df_c.merge(df_l, left_on="listing_id", right_on="id", how="left")
        merged["is_booked"] = merged["available"] == "f"

        # Tổng hợp toàn thành phố ở mốc đó
        rows.append({
            "moc_thoi_gian": snap_str,
            "analysis_area": "ALL_VIENNA",
            "tong_ngay_lich": len(merged),
            "so_ngay_khong_kha_dung": int(merged["is_booked"].sum()),
            "ty_le_khong_kha_dung_pct": round(float(merged["is_booked"].mean() * 100), 2) if len(merged) > 0 else 0.0,
            "so_dem_toi_thieu_trung_vi": round(float(pd.to_numeric(merged["minimum_nights"], errors="coerce").median()), 2),
        })

        # Tổng hợp chi tiết theo từng vùng A01 - A08
        for area_val, grp in merged.groupby("analysis_area"):
            rows.append({
                "moc_thoi_gian": snap_str,
                "analysis_area": area_val,
                "tong_ngay_lich": len(grp),
                "so_ngay_khong_kha_dung": int(grp["is_booked"].sum()),
                "ty_le_khong_kha_dung_pct": round(float(grp["is_booked"].mean() * 100), 2) if len(grp) > 0 else 0.0,
                "so_dem_toi_thieu_trung_vi": round(float(pd.to_numeric(grp["minimum_nights"], errors="coerce").median()), 2),
            })

    return pd.DataFrame(rows)

def compute_kpi3_review_activity_trend(snapshots: list) -> pd.DataFrame:
    """KPI 3: Động lực tăng trưởng đánh giá và sự giảm dần số phòng chưa có đánh giá qua các mốc."""
    rows = []
    for snap in snapshots:
        snap_str = str(snap)
        list_path = Path("data/processed") / snap_str / "listings.parquet"
        rev_path = Path("data/processed") / snap_str / "reviews.parquet"
        if not list_path.exists():
            continue

        df_l = filter_clean_rows(pd.read_parquet(list_path), require_price=False)
        num_rev = pd.to_numeric(df_l.get("number_of_reviews", 0), errors="coerce").fillna(0)
        ratings = pd.to_numeric(df_l.get("review_scores_rating"), errors="coerce").dropna()

        valid_rev_sample = 0
        if rev_path.exists():
            df_r = filter_clean_rows(pd.read_parquet(rev_path), require_price=False)
            valid_rev_sample = len(df_r)

        rows.append({
            "moc_thoi_gian": snap_str,
            "so_phong_sach": len(df_l),
            "so_phong_chua_co_review": int((num_rev == 0).sum()),
            "so_phong_da_co_review": int((num_rev > 0).sum()),
            "tong_review_tich_luy_tren_listings": int(num_rev.sum()),
            "trung_vi_so_review_moi_phong": round(float(num_rev.median()), 1),
            "diem_rating_trung_binh": round(float(ratings.mean()), 3) if not ratings.empty else None,
            "diem_rating_trung_vi": round(float(ratings.median()), 3) if not ratings.empty else None,
            "so_review_hop_le_trong_mau": valid_rev_sample,
        })

    return pd.DataFrame(rows)

def load_text_extraction_results(proc_dir: Path = Path("data/processed")) -> tuple[pd.DataFrame, str]:
    """Đọc bảng kết quả LLM (llm_results.parquet). Không có thì báo lỗi, KHÔNG dùng baseline thay thế."""
    llm_path = proc_dir / "llm_results.parquet"
    if llm_path.exists():
        return pd.read_parquet(llm_path), "llm"

    combined_path = proc_dir / "reviews_llm_extracted.parquet"
    if combined_path.exists():
        return pd.read_parquet(combined_path), "llm"

    raise FileNotFoundError("Không tìm thấy llm_results.parquet trong data/processed/. Hãy chạy llm.py trước (KPI 5 & 6 phải dùng kết quả LLM).")

def keep_valid_text_rows(df_text: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ các bình luận có kết quả LLM hợp lệ (llm_ok=True), không điền giả cho dòng thiếu."""
    if "llm_ok" in df_text.columns:
        return df_text[df_text["llm_ok"].astype(bool)].copy()
    return df_text.copy()

def compute_kpi5_sentiment_by_area(df_text: pd.DataFrame, df_last_listings: pd.DataFrame, prefix: str = "llm") -> pd.DataFrame:
    """KPI 5: Phân bổ tỷ lệ cảm xúc (positive/mixed/negative) và ngôn ngữ theo khu vực analysis_area."""
    sent_col = f"{prefix}_sentiment" if f"{prefix}_sentiment" in df_text.columns else "base_sentiment"
    lang_col = f"{prefix}_lang" if f"{prefix}_lang" in df_text.columns else "base_lang"

    df_text = keep_valid_text_rows(df_text)
    df_map = df_last_listings[["id", "analysis_area", "neighbourhood_cleansed"]].drop_duplicates(subset=["id"])
    merged = df_text.merge(df_map, left_on="listing_id", right_on="id", how="left")

    rows = []
    for area_val, grp in merged.groupby("analysis_area"):
        n_total = len(grp)
        sent_s = grp[sent_col].astype(str).str.lower()
        lang_s = grp[lang_col].astype(str).str.lower()

        pos_pct = round(float((sent_s == "positive").mean() * 100), 2)
        mix_pct = round(float((sent_s == "mixed").mean() * 100), 2)
        neg_pct = round(float((sent_s == "negative").mean() * 100), 2)

        rows.append({
            "analysis_area": area_val,
            "so_binh_luan_mau": n_total,
            "ty_le_positive_pct": pos_pct,
            "ty_le_mixed_pct": mix_pct,
            "ty_le_negative_pct": neg_pct,
            "diem_cam_xuc_rong_net_pct": round(pos_pct - neg_pct, 2),
            "ty_le_tieng_anh_pct": round(float((lang_s == "en").mean() * 100), 2),
            "ty_le_tieng_duc_pct": round(float((lang_s == "de").mean() * 100), 2),
            "canh_bao_n_nho": n_total < MIN_N_TEXT,
        })

    return pd.DataFrame(rows).sort_values(by="diem_cam_xuc_rong_net_pct", ascending=False).reset_index(drop=True)

def compute_kpi6_aspects_summary(df_text: pd.DataFrame, valid_aspects: list[str], prefix: str = "llm") -> pd.DataFrame:
    """KPI 6: Thống kê lượt khen (pos), lượt chê (neg) và tỷ lệ hài lòng ròng cho các khía cạnh."""
    pos_col = f"{prefix}_aspects_pos" if f"{prefix}_aspects_pos" in df_text.columns else "base_aspects_pos"
    neg_col = f"{prefix}_aspects_neg" if f"{prefix}_aspects_neg" in df_text.columns else "base_aspects_neg"

    df_text = keep_valid_text_rows(df_text)
    n_total = max(len(df_text), 1)
    pos_sets = [
        {x.strip().lower() for x in str(val).split(",") if x.strip() and str(val).lower() != "nan"}
        for val in df_text.get(pos_col, [])
    ]
    neg_sets = [
        {x.strip().lower() for x in str(val).split(",") if x.strip() and str(val).lower() != "nan"}
        for val in df_text.get(neg_col, [])
    ]

    rows = []
    for aspect in valid_aspects:
        asp_lower = aspect.strip().lower()
        p_cnt = sum(1 for s in pos_sets if asp_lower in s)
        n_cnt = sum(1 for s in neg_sets if asp_lower in s)
        total_mentions = p_cnt + n_cnt

        rows.append({
            "khia_canh": aspect,
            "so_luot_khen": p_cnt,
            "ty_le_khen_pct": round(p_cnt / n_total * 100, 2),
            "so_luot_che": n_cnt,
            "ty_le_che_pct": round(n_cnt / n_total * 100, 2),
            "tong_luot_nhac": total_mentions,
            "ty_le_nhac_toi_pct": round(total_mentions / n_total * 100, 2),
            "ty_le_khen_tren_tong_nhac_pct": round(p_cnt / total_mentions * 100, 2) if total_mentions > 0 else 0.0,
            "so_binh_luan_hop_le": len(df_text),
        })

    return pd.DataFrame(rows).sort_values(by="tong_luot_nhac", ascending=False).reset_index(drop=True)

def build_kpi_definitions() -> pd.DataFrame:
    """Bảng định nghĩa KPI để đính kèm báo cáo. Cột ben_lien_quan: sửa theo bên đặt hàng mà nhóm đã chọn."""
    stakeholder = "Cơ quan quản lý du lịch/đô thị Vienna (sửa theo bên đặt hàng của nhóm)"
    rows = [
        ("KPI 1", "Giá thuê theo khu vực", "So sánh mặt bằng giá giữa các vùng", "median/mean/IQR của price_num theo analysis_area, mốc cuối, dòng sạch về giá", "EUR/đêm"),
        ("KPI 2", "Tỷ lệ ngày không khả dụng", "Theo dõi mức khai thác theo thời gian và vùng (chỉ báo gần đúng, gồm ngày chủ nhà chặn)", "số ngày available='f' / tổng ngày lịch sạch", "%"),
        ("KPI 3", "Động lực đánh giá", "Theo dõi tăng trưởng review và số phòng chưa có review", "tổng number_of_reviews; số phòng có number_of_reviews = 0", "lượt; phòng"),
        ("KPI 4", "Chuyên nghiệp hóa chủ nhà", "Đo tỷ trọng chủ nhà nhiều phòng và chênh lệch giá", "tỷ lệ phòng có calculated_host_listings_count >= 2; median giá hai nhóm", "%; EUR/đêm"),
        ("KPI 5", "Cảm xúc khách theo khu vực (LLM)", "So sánh mức hài lòng giữa các vùng", "tỷ lệ positive/mixed/negative trên bình luận LLM hợp lệ; net = positive - negative", "%"),
        ("KPI 6", "Khen/chê theo khía cạnh (LLM)", "Xác định khía cạnh mạnh/yếu của dịch vụ", "số bình luận khen/chê mỗi khía cạnh / số bình luận hợp lệ", "lượt; %"),
        ("KPI 7", "Giá theo quận (neighbourhood_cleansed)", "Dữ liệu cho bản đồ giá theo ranh giới hành chính", "median price_num theo neighbourhood_cleansed, mốc cuối", "EUR/đêm"),
    ]
    return pd.DataFrame(
        [{"kpi": k, "ten": t, "muc_tieu": g, "cong_thuc": f, "don_vi": u, "ben_lien_quan": stakeholder} for k, t, g, f, u in rows]
    )

def run_kpi(config_path: str | Path = "configs/vienna.yaml") -> dict[str, pd.DataFrame]:
    """Điều phối tính toán toàn bộ bảng KPI và lưu kết quả tại data/processed/."""
    print("--- BẮT ĐẦU TÍNH TOÁN BỘ CHỈ SỐ NGHIỆP VỤ (KPI PIPELINE) ---")
    config = load_config(config_path)
    snapshots = config.get("snapshots", [])
    last_snap = str(snapshots[-1])
    valid_aspects = config.get("llm", {}).get("aspects", ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"])
    area_pref = config.get("kpi", {}).get("area_col", "analysis_area")

    proc_dir = Path("data/processed")
    last_listings_path = proc_dir / last_snap / "listings.parquet"
    if not last_listings_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {last_listings_path}. Hãy chạy filter.py và qa.py trước.")

    df_last_listings = pd.read_parquet(last_listings_path)

    # 1. Tính các KPI từ dữ liệu có cấu trúc (Listings & Calendar)
    kpi1_df = compute_kpi1_price_by_area(df_last_listings, area_pref)
    kpi2_df = compute_kpi2_occupancy_trend(snapshots)
    kpi3_df = compute_kpi3_review_activity_trend(snapshots)
    kpi4_df = compute_kpi4_host_professionalization(df_last_listings, area_pref)
    kpi7_df = compute_kpi7_price_by_neighbourhood(df_last_listings)

    # 2. Tính 2 KPI từ dữ liệu văn bản đã trích xuất bằng LLM
    df_text, prefix = load_text_extraction_results(proc_dir)
    kpi5_df = compute_kpi5_sentiment_by_area(df_text, df_last_listings, prefix=prefix)
    kpi6_df = compute_kpi6_aspects_summary(df_text, valid_aspects, prefix=prefix)

    # 3. Lưu toàn bộ tệp CSV ra data/processed/
    kpi_outputs = {
        "kpi_1_price_by_area.csv": kpi1_df,
        "kpi_2_occupancy_trend.csv": kpi2_df,
        "kpi_3_review_activity_trend.csv": kpi3_df,
        "kpi_4_host_professionalization.csv": kpi4_df,
        "kpi_5_llm_sentiment_by_area.csv": kpi5_df,
        "kpi_6_llm_aspects_summary.csv": kpi6_df,
        "kpi_7_price_by_neighbourhood.csv": kpi7_df,
    }

    for filename, df_out in kpi_outputs.items():
        out_path = proc_dir / filename
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"    [+] Đã xuất {out_path} ({len(df_out)} dòng)")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    build_kpi_definitions().to_csv(reports_dir / "kpi_definitions.csv", index=False, encoding="utf-8-sig")

    print("\n--- TÓM TẮT NHANH KPI 1 (GIÁ THEO VÙNG MỐC CUỐI) ---")
    area_col = resolve_area_column(kpi1_df, area_pref)
    print(kpi1_df[[area_col, "so_phong_sach", "gia_trung_vi_eur", "gia_trung_binh_eur", "gia_iqr_eur"]].to_string(index=False))

    print("\n--- TÓM TẮT NHANH KPI 6 (CÁC KHÍA CẠNH TỪ LLM) ---")
    print(kpi6_df[["khia_canh", "so_luot_khen", "so_luot_che", "ty_le_nhac_toi_pct"]].to_string(index=False))

    print("\n[+] HOÀN TẤT TÍNH KPI! Tất cả tệp CSV đã sẵn sàng cho bước vẽ biểu đồ (src/viz.py).")
    return kpi_outputs

if __name__ == "__main__":
    run_kpi()
