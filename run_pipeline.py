import time
import argparse
import subprocess
import sys
from pathlib import Path

from src.download import download_data, load_config as load_download_config
from src.filter import run_filter
from src.qa import run_qa
from src.baseline import run_baseline
from src.llm import run_llm
from src.evaluate import run_evaluate
from src.kpi import run_kpi
from src.viz import run_viz

def print_step_banner(step_title: str, major: bool = False) -> None:
    line = "=" * 80 if major else "-" * 80
    print(f"\n{line}")
    print(step_title)
    print(line)

def ensure_raw_data_downloaded(config_path: str | Path) -> None:
    print_step_banner("[Bước 0] Kiểm tra và tải dữ liệu thô theo mốc trong config...")
    failed = download_data(load_download_config(config_path))
    critical = [f for f in failed if not f.endswith("neighbourhoods.geojson")]
    if critical:
        raise RuntimeError(f"Không tải được các tệp bắt buộc: {critical}")
    if failed:
        print(f"    [!] Thiếu geojson ({failed}); biểu đồ bản đồ sẽ bị bỏ qua.")

def run_full_pipeline(config_path: str = "configs/vienna.yaml", skip_llm: bool = False) -> None:
    start_time = time.time()
    print_step_banner(
        f"[Pipeline] Bắt đầu chạy quy trình (config={config_path}, skip_llm={skip_llm})",
        major=True,
    )

    # Tải dữ liệu thô nếu chưa có
    ensure_raw_data_downloaded(config_path)

    # Lọc dữ liệu theo danh sách ID phân công của nhóm
    print_step_banner("[Bước 1] Lọc dữ liệu theo danh sách phân công...")
    run_filter(config_path=config_path)

    # Kiểm tra nhanh cấu trúc cột nếu có script check_data.py
    check_script = Path("src/check_data.py")
    if check_script.exists():
        print_step_banner("[Bước 1b] Kiểm tra cấu trúc cột dữ liệu sau lọc...")
        subprocess.run([sys.executable, str(check_script)], check=False)

    # Làm sạch và gắn cờ kiểm định chất lượng (QA)
    print_step_banner("[Bước 2] Làm sạch và kiểm định chất lượng dữ liệu (QA)...")
    run_qa(config_path=config_path)

    # Phân loại cảm xúc & khía cạnh bằng bộ từ khóa đối chứng
    print_step_banner("[Bước 3] Chạy phương pháp đối chứng bằng từ khóa (Baseline)...")
    run_baseline(config_path=config_path)

    # Trích xuất cảm xúc & khía cạnh qua Gemini API (hoặc đọc từ cache)
    print_step_banner("[Bước 4] Trích xuất cảm xúc và khía cạnh bằng LLM...")
    run_llm(config_path=config_path, skip_llm=skip_llm)

    # Chấm điểm Accuracy/F1 của Baseline và LLM trên 100 nhãn tay
    print_step_banner("[Bước 5] Đánh giá chất lượng LLM và Baseline trên bộ nhãn tay...")
    run_evaluate(config_path=config_path)

    # Tính toán các bảng chỉ số KPI
    print_step_banner("[Bước 6] Tính toán bộ chỉ số nghiệp vụ (KPI)...")
    run_kpi(config_path=config_path)

    # Vẽ bộ biểu đồ tĩnh và lưu vào thư mục figures/
    print_step_banner("[Bước 7] Vẽ bộ biểu đồ tĩnh lưu vào figures/...")
    run_viz(config_path=config_path)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"[Pipeline] Hoàn tất toàn bộ quy trình trong {elapsed} giây!")
    print("    - Dữ liệu sạch & KPI : data/processed/")
    print("    - Báo cáo QA & LLM   : reports/")
    print("    - Biểu đồ PNG        : figures/")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy quy trình xử lý và phân tích dữ liệu Inside Airbnb")
    parser.add_argument("--config", default="configs/vienna.yaml", help="Đường dẫn tệp cấu hình YAML")
    parser.add_argument("--skip-llm", action="store_true", help="Dùng kết quả có sẵn trong cache, không gọi API")
    args = parser.parse_args()

    run_full_pipeline(config_path=args.config, skip_llm=args.skip_llm)