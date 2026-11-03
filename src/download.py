import requests
import time
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_data(config):
    base_url = "https://data.insideairbnb.com"
    data_root = config["data_root"]
    snapshots = config["snapshots"]
    files = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True,exist_ok=True)

    for snapshot in snapshots:
        print(f"- Đang tải mốc thời gian: {snapshot}")
        snapshot_dir = raw_dir/str(snapshot)
        snapshot_dir.mkdir(parents=True,exist_ok=True)
        for file in files:
            url = f"{base_url}/{data_root}/{snapshot}/data/{file}"
            file_path = snapshot_dir/file 
            if(file_path.exists()):
                print(f"    + Đã tồn tại file {file} tại {snapshot}, bỏ qua")
                continue
            print(f"    + Đang tải : {file} tại {snapshot}")
            try:
                reponse = requests.get(url,timeout=20)
                reponse.raise_for_status()
                with open(file_path,"wb") as f:
                    f.write(reponse.content)
            except requests.exceptions.RequestException() as e:
                print(f"  [!] Lỗi khi tải {file}: {e}")

if __name__ == "__main__":
    config_data = load_config("configs/vienna.yaml")
    download_data(config_data)
    print("\nHoàn tất quá trình tải dữ liệu!")