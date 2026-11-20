import json
import argparse
import requests
import yaml
import pandas as pd
from pathlib import Path

BASE_URL = "https://data.insideairbnb.com"
GEOJSON_FILE = "neighbourhoods.geojson"

# Các cột khóa tối thiểu để kiểm tra tệp tải về không bị rỗng hay hỏng
REQUIRED_COLUMNS = {
    "listings.csv.gz": ["id"],
    "calendar.csv.gz": ["listing_id", "date", "available"],
    "reviews.csv.gz": ["id", "listing_id", "date", "comments"],
}

def load_config(config_path: str | Path = "configs/vienna.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_url(data_root: str, snapshot: object, file: str) -> str:
    # GeoJSON nằm ở thư mục visualisations, các bảng dữ liệu nằm ở thư mục data
    folder = "visualisations" if file == GEOJSON_FILE else "data"
    return f"{BASE_URL}/{data_root}/{snapshot}/{folder}/{file}"

def is_valid_file(file_path: Path) -> bool:
    try:
        if file_path.name == GEOJSON_FILE:
            with open(file_path, "r", encoding="utf-8") as f:
                return "features" in json.load(f)
        header = pd.read_csv(file_path, compression="gzip", nrows=0)
        return set(REQUIRED_COLUMNS[file_path.name]).issubset(header.columns)
    except Exception:
        return False

def download_file(url: str, file_path: Path) -> None:
    # Ghi tạm vào tệp .part rồi mới đổi tên để tránh giữ lại tệp lỗi nếu đứt mạng giữa chừng
    tmp_path = file_path.with_name(file_path.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp_path.replace(file_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def download_data(config: dict) -> list[str]:
    data_root = config["data_root"]
    snapshots = config["snapshots"]
    files = list(REQUIRED_COLUMNS) + [GEOJSON_FILE]
    failed: list[str] = []

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshots:
        print(f"- Đang tải mốc thời gian: {snapshot}")
        snapshot_dir = raw_dir / str(snapshot)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            url = build_url(data_root, snapshot, file)
            file_path = snapshot_dir / file
            if file_path.exists():
                if is_valid_file(file_path):
                    print(f"    + Đã tồn tại file {file} tại {snapshot}, bỏ qua")
                    continue
                print(f"    + File {file} tại {snapshot} bị hỏng hoặc thiếu cột khóa, tải lại")
                file_path.unlink()
            print(f"    + Đang tải: {file} tại {snapshot}")
            try:
                download_file(url, file_path)
            except (requests.exceptions.RequestException, OSError) as e:
                print(f"    [!] Lỗi khi tải {file}: {e}")
                failed.append(f"{snapshot}/{file}")
                continue
            if not is_valid_file(file_path):
                print(f"    [!] File {file} tại {snapshot} không hợp lệ (thiếu cột khóa)")
                file_path.unlink()
                failed.append(f"{snapshot}/{file}")
    return failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải dữ liệu Inside Airbnb theo cấu hình YAML")
    parser.add_argument("--config", default="configs/vienna.yaml", help="Đường dẫn tệp cấu hình YAML")
    args = parser.parse_args()

    print("-" * 80)
    print("[Bước 0] Kiểm tra và tải dữ liệu thô theo mốc trong config...")
    print("-" * 80)

    failed_files = download_data(load_config(args.config))
    if failed_files:
        raise RuntimeError(f"Các tệp tải lỗi: {failed_files}")
    print("\n[+] Hoàn tất quá trình tải dữ liệu!")
