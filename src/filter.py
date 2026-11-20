import pandas as pd
import yaml
from pathlib import Path

CALENDAR_COLS = ["listing_id", "date", "available", "minimum_nights", "maximum_nights"]

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_listing(path: str | Path = "data/samples/danh_sach_listing.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df[["listing_id", "analysis_area"]].drop_duplicates(subset=["listing_id"])

def load_review(path: str | Path = "data/samples/danh_sach_review.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df[["review_id", "Mẫu LLM", "Nhãn tay"]].drop_duplicates(subset=["review_id"])

def filter_listing(raw_path: Path, out_path: Path, listing_df: pd.DataFrame) -> None:
    if not raw_path.exists():
        return

    valid_listing_ids = set(listing_df["listing_id"])
    df_l = pd.read_csv(raw_path, compression="gzip", low_memory=False)
    df_l = df_l[df_l["id"].isin(valid_listing_ids)].copy()

    if "analysis_area" in df_l.columns:
        df_l = df_l.drop(columns=["analysis_area"])

    # Ghép vùng phân tích và bỏ cột khóa phụ để giữ nguyên số cột gốc
    df_l = df_l.merge(
        listing_df[["listing_id", "analysis_area"]],
        left_on="id",
        right_on="listing_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["listing_id"])

    df_l.to_parquet(out_path, index=False)
    print(f"    -> Listings: {len(df_l)} dòng, {df_l.shape[1]} cột.")

def filter_review(raw_path: Path, out_path: Path, review_df: pd.DataFrame) -> None:
    if not raw_path.exists():
        return

    valid_review_ids = set(review_df["review_id"])
    df_r = pd.read_csv(raw_path, compression="gzip", low_memory=False)
    df_r = df_r[df_r["id"].isin(valid_review_ids)].copy()

    for col in ["Mẫu LLM", "Nhãn tay"]:
        if col in df_r.columns:
            df_r = df_r.drop(columns=[col])

    # Ghép cột đánh dấu mẫu và bỏ review_id phụ, giữ lại 'id' và 'listing_id' gốc
    df_r = df_r.merge(
        review_df[["review_id", "Mẫu LLM", "Nhãn tay"]],
        left_on="id",
        right_on="review_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["review_id"])

    df_r.to_parquet(out_path, index=False)
    print(f"    -> Reviews : {len(df_r)} dòng, {df_r.shape[1]} cột.")

def filter_calendar(
    raw_path: Path,
    out_path: Path,
    valid_listing_ids: set,
    chunksize: int = 500_000,
) -> None:
    if not raw_path.exists():
        return

    # Đọc theo từng lô kết hợp usecols để tránh tràn RAM với bảng calendar lớn
    filtered_chunks = []
    chunk_iterator = pd.read_csv(
        raw_path,
        compression="gzip",
        usecols=CALENDAR_COLS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in chunk_iterator:
        sub = chunk[chunk["listing_id"].isin(valid_listing_ids)]
        filtered_chunks.append(sub)

    df_c = pd.concat(filtered_chunks, ignore_index=True)
    df_c.to_parquet(out_path, index=False)
    print(f"    -> Calendar: {len(df_c)} dòng, {df_c.shape[1]} cột.")

def run_filter(config_path: str | Path = "configs/vienna.yaml") -> None:
    config = load_config(config_path)
    snapshots = config["snapshots"]

    sample_listings_df = load_listing("data/samples/danh_sach_listing.csv")
    sample_reviews_df = load_review("data/samples/danh_sach_review.csv")
    valid_listing_ids = set(sample_listings_df["listing_id"])

    print(f"- Đã nạp {len(sample_listings_df)} listing_id và {len(sample_reviews_df)} review_id.")

    for snapshot in snapshots:
        print(f"\n- Đang xử lý mốc: {snapshot}...")
        raw_dir = Path("data/raw") / str(snapshot)
        proc_dir = Path("data/processed") / str(snapshot)
        proc_dir.mkdir(parents=True, exist_ok=True)

        filter_listing(raw_dir / "listings.csv.gz", proc_dir / "listings.parquet", sample_listings_df)
        filter_calendar(raw_dir / "calendar.csv.gz", proc_dir / "calendar.parquet", valid_listing_ids)
        filter_review(raw_dir / "reviews.csv.gz", proc_dir / "reviews.parquet", sample_reviews_df)

    print("\n[+] Hoàn tất lọc dữ liệu! Tất cả tệp đã lưu tại data/processed/")

if __name__ == "__main__":
    print("-" * 80)
    print("[Bước 1] Lọc dữ liệu theo danh sách phân công...")
    print("-" * 80)
    run_filter()
