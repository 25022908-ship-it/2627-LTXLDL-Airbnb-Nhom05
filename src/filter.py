import pandas as pd
import yaml
from pathlib import Path

CALENDAR_COLS = ["listing_id", "date", "available", "minimum_nights", "maximum_nights"]

def load_config(config_path="configs/vienna.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_listing(path="data/samples/danh_sach_listing.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df[["listing_id", "analysis_area"]].drop_duplicates(subset=["listing_id"])

def load_review(path = "data/samples/danh_sach_review.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df[["review_id", "Mẫu LLM", "Nhãn tay"]].drop_duplicates(subset=["review_id"])

def filter_listing(raw_path,out_path,listing_df):
    if not raw_path.exists():
        return
        
    valid_listing_ids = set(listing_df["listing_id"])
    
    df_l = pd.read_csv(raw_path, compression="gzip", low_memory=False)
    df_l = df_l[df_l["id"].isin(valid_listing_ids)].copy()
    
    # Xóa cột analysis_area cũ nếu có trước khi ghép
    if "analysis_area" in df_l.columns:
        df_l = df_l.drop(columns=["analysis_area"])
        
    # Ghép bảng bằng left_on="id" và right_on="listing_id"
    df_l = df_l.merge(
        listing_df[["listing_id", "analysis_area"]],
        left_on="id",
        right_on="listing_id",
        how="left",
        validate="one_to_one"
    ).drop(columns=["listing_id"]) # Bỏ cột khóa phụ sau khi merge để giữ đúng 80 cột
    
    df_l.to_parquet(out_path, index=False)
    print(f"    -> Listings: {len(df_l)} dòng, {df_l.shape[1]} cột.")

def filter_review(raw_path,out_path,review_df):
    if not raw_path.exists():
        return
        
    valid_review_ids = set(review_df["review_id"])
    
    df_r = pd.read_csv(raw_path, compression="gzip", low_memory=False)
    df_r = df_r[df_r["id"].isin(valid_review_ids)].copy()
    
    # Xóa cột cũ nếu có trước khi ghép
    for col in ["Mẫu LLM", "Nhãn tay"]:
        if col in df_r.columns:
            df_r = df_r.drop(columns=[col])

    df_r = df_r.merge(
        review_df[["review_id", "Mẫu LLM", "Nhãn tay"]],
        left_on="id",
        right_on="review_id",
        how="left",
        validate="one_to_one"
    ).drop(columns=["review_id"]) # Bỏ cột review_id phụ để giữ nguyên cột 'id' và 'listing_id' gốc
    
    df_r.to_parquet(out_path, index=False)
    print(f"    -> Reviews : {len(df_r)} dòng, {df_r.shape[1]} cột.")

def filter_calendar(raw_path,out_path,valid_listing_ids,chunksize = 500000):
    if not raw_path.exists():
        return

    filtered_chunks = []
    chunk_iterator = pd.read_csv(
        raw_path,
        compression="gzip",
        usecols=CALENDAR_COLS,
        chunksize=chunksize,
        low_memory=False
    )

    for chunk in chunk_iterator:
        sub = chunk[chunk["listing_id"].isin(valid_listing_ids)]
        filtered_chunks.append(sub)
        
    df_c = pd.concat(filtered_chunks, ignore_index=True)
    df_c.to_parquet(out_path, index=False)
    print(f"    -> Calendar: {len(df_c)} dòng, {df_c.shape[1]} cột (Đã đọc bằng chunksize).")

def run_filter(config_path="configs/vienna.yaml"):
    print("--- BẮT ĐẦU LỌC DỮ LIỆU THEO PHÂN CÔNG ---")
    config = load_config(config_path)
    snapshots = config["snapshots"]
    
    # Đọc 2 tệp danh sách 
    sample_listings_df = load_listing("data/samples/danh_sach_listing.csv")
    sample_reviews_df  = load_review("data/samples/danh_sach_review.csv")
    
    valid_listing_ids = set(sample_listings_df["listing_id"])
    print(f"Đã nạp {len(sample_listings_df)} listing_id và {len(sample_reviews_df)} review_id.")
    
    # Quét qua 4 mốc thời gian
    for snapshot in snapshots:
        print(f"\n - Đang xử lý mốc: {snapshot}...")
        raw_dir = Path("data/raw") / str(snapshot)
        proc_dir = Path("data/processed") / str(snapshot)
        proc_dir.mkdir(parents=True, exist_ok=True)
        
        filter_listing(raw_dir / "listings.csv.gz", proc_dir / "listings.parquet", sample_listings_df)
        filter_calendar(raw_dir / "calendar.csv.gz", proc_dir / "calendar.parquet", valid_listing_ids)
        filter_review(raw_dir / "reviews.csv.gz", proc_dir / "reviews.parquet", sample_reviews_df)

    print("\n HOÀN TẤT LỌC DỮ LIỆU! Tất cả tệp đã được lưu tại data/processed/")

if __name__ == "__main__":
    run_filter()