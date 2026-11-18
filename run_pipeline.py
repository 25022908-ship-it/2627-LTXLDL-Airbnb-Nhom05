import time
import argparse
from pathlib import Path

from src.download import download_data, load_config as load_download_config
from src.filter import run_filter
from src.qa import run_qa
from src.baseline import run_baseline
from src.llm import run_llm
from src.evaluate import run_evaluate
from src.kpi import run_kpi
from src.viz import run_viz


def ensure_raw_data_downloaded(config_path: str | Path) -> None:
    """Kiểm tra/tải từng tệp của từng mốc theo config (tệp đã hợp lệ sẽ được bỏ qua)."""
    print("\n[Bước 0] Kiểm tra và tải dữ liệu thô theo mốc trong config...")
    failed = download_data(load_download_config(config_path))
    critical = [f for f in failed if not f.endswith("neighbourhoods.geojson")]
    if critical:
        raise RuntimeError(f"Không tải được các tệp bắt buộc: {critical}")
    if failed:
        print(f"    [!] Thiếu geojson ({failed}); biểu đồ bản đồ sẽ bị bỏ qua.")


def run_full_pipeline(config_path: str = "configs/vienna.yaml", skip_llm: bool = False) -> None:
    """Điều phối chạy tuần tự toàn bộ quy trình phân tích dữ liệu Vienna từ đầu đến cuối."""
    start_time = time.time()
    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE VIENNA (config={config_path}, skip_llm={skip_llm})")
    print("=" * 80)

    # Bước 0: Đảm bảo dữ liệu thô đã tải về
    ensure_raw_data_downloaded(config_path)

    # Bước 1: Lọc dữ liệu theo phân công của Nhóm 5 (usecols & chunksize)
    print("\n" + "-" * 80)
    run_filter(config_path=config_path)

    # Bước 2: Chạy check_data.py (nếu có) ngay sau khi lọc để xuất báo cáo kiểm tra 80 cột chuẩn
    check_script = Path("src/check_data.py")
    if check_script.exists():
        print("\n" + "-" * 80)
        import subprocess
        import sys
        subprocess.run([sys.executable, str(check_script)], check=False)

    # Bước 3: Làm sạch và kiểm định chất lượng dữ liệu (QA)
    print("\n" + "-" * 80)
    run_qa(config_path=config_path)

    # Bước 4: Chạy phương pháp đối chứng từ khóa (Baseline)
    print("\n" + "-" * 80)
    run_baseline(config_path=config_path)

    # Bước 5: Trích xuất có cấu trúc bằng Gemini LLM (hoặc đọc từ cache nếu --skip-llm)
    print("\n" + "-" * 80)
    run_llm(config_path=config_path, skip_llm=skip_llm)

    # Bước 6: Đánh giá đối chiếu LLM vs Baseline trên bộ 100 nhãn tay
    print("\n" + "-" * 80)
    run_evaluate(config_path=config_path)

    # Bước 7: Tính toán các bảng chỉ số nghiệp vụ (KPI)
    print("\n" + "-" * 80)
    run_kpi(config_path=config_path)

    # Bước 8: Tự động vẽ bộ biểu đồ tĩnh lưu vào figures/
    print("\n" + "-" * 80)
    run_viz(config_path=config_path)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"[+] HOÀN TẤT TOÀN BỘ PIPELINE TRONG {elapsed} GIÂY!")
    print("    - Dữ liệu sạch & KPI : data/processed/")
    print("    - Báo cáo QA & LLM   : reports/")
    print("    - Biểu đồ PNG        : figures/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline phân tích dữ liệu Inside Airbnb - Vienna")
    parser.add_argument("--config", default="configs/vienna.yaml", help="Đường dẫn tới tệp cấu hình YAML")
    parser.add_argument("--skip-llm", action="store_true", help="Không gọi Gemini API mới, sử dụng kết quả từ cache")
    args = parser.parse_args()

    run_full_pipeline(config_path=args.config, skip_llm=args.skip_llm)
