# Phân tích Dữ liệu Inside Airbnb — Vienna (Nhóm 05)

Quy trình xử lý dữ liệu tự động (data pipeline) phân tích thị trường lưu trú ngắn hạn Airbnb tại Vienna qua 4 mốc thời gian: thu thập dữ liệu thô, lọc theo danh sách phân công, kiểm định chất lượng (QA), trích xuất cảm xúc & khía cạnh đánh giá bằng Mô hình Ngôn ngữ Lớn (Gemini LLM), tính toán 7 chỉ số KPI và trực quan hóa biểu đồ phục vụ báo cáo phân tích.

- **Mã phân công:** `DSE2049-C04-G05-vienna`
- **Nguồn dữ liệu:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) (Giấy phép CC BY 4.0)
- **Các mốc chụp (Snapshots):** `2025-09-14`, `2025-12-16`, `2026-03-24`, `2026-06-20`

---

## 1. Cấu trúc thư mục dự án

```text
.
├── configs/
│   └── vienna.yaml               # Cấu hình thành phố, mốc chụp, ngưỡng QA, KPI và LLM
├── data/
│   ├── raw/                      # Dữ liệu thô tải từ Inside Airbnb (không commit lên Git)
│   ├── processed/                # Dữ liệu sạch (.parquet), cache LLM (.jsonl), bảng KPI (.csv)
│   └── samples/                  # Danh sách ID phân công & bộ 100 câu gán nhãn tay chuẩn
├── src/
│   ├── download.py               # Tải tự động dữ liệu thô & kiểm tra tính toàn vẹn
│   ├── filter.py                 # Lọc dữ liệu theo danh sách ID phân công của nhóm
│   ├── qa.py                     # Làm sạch, gắn cờ QA (flag_*, is_clean) & xuất qa_report.csv
│   ├── baseline.py               # Phương pháp đối chứng không dùng LLM (Regex/Từ khóa)
│   ├── llm.py                    # Trích xuất cảm xúc & khía cạnh bằng Gemini API (có cache)
│   ├── evaluate.py               # Đánh giá Accuracy/F1 của Baseline & LLM trên 100 nhãn tay
│   ├── kpi.py                    # Tính toán 7 bảng chỉ số KPI & xuất bảng định nghĩa KPI
│   └── viz.py                    # Vẽ tự động 9 biểu đồ tĩnh (.png) vào thư mục figures/
├── figures/                      # Các biểu đồ sinh tự động phục vụ báo cáo
├── reports/                      # Báo cáo PDF cuối kỳ, qa_report.csv, llm_eval_report.csv...
├── run_pipeline.py               # Kịch bản điều phối chạy toàn bộ quy trình bằng 1 lệnh
├── .python-version               # Phiên bản Python đã kiểm thử (3.12.3)
├── .env.example                  # Mẫu khai báo biến môi trường GEMINI_API_KEY
├── .gitignore                    # Loại trừ data/raw/, .venv/, .env khỏi Git
├── requirements.txt              # Danh sách thư viện đã ghim phiên bản (==)
├── AI_USAGE.md                   # Báo cáo minh bạch việc sử dụng AI
└── README.md                     # Tài liệu hướng dẫn sử dụng (tệp này)
```

---

## 2. Cài đặt môi trường

1. **Phiên bản Python:** Cài đặt đúng phiên bản Python ghi trong tệp `.python-version` (`3.13.12`).
2. **Tạo và kích hoạt môi trường ảo:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
3. **Cài đặt thư viện đã ghim phiên bản:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 3. Khai báo khóa API (Biến môi trường)

Mô-đun trích xuất đánh giá bằng Gemini (`src/llm.py`) đọc khóa API từ biến môi trường `GEMINI_API_KEY` (không ghi trực tiếp trong mã nguồn):

- **Cách 1 — Dùng tệp `.env` ở thư mục gốc** (sao chép từ `.env.example`, tệp `.env` đã được đưa vào `.gitignore`):
  ```env
  GEMINI_API_KEY=your_actual_gemini_api_key_here
  ```
- **Cách 2 — Đặt biến môi trường trực tiếp trên Terminal:**
  - **Linux / macOS:**
    ```bash
    export GEMINI_API_KEY="your_actual_gemini_api_key_here"
    ```
  - **Windows (PowerShell):**
    ```powershell
    $env:GEMINI_API_KEY="your_actual_gemini_api_key_here"
    ```

---

## 4. Hướng dẫn chạy quy trình

### Chạy toàn bộ quy trình bằng một lệnh duy nhất

- **Chạy đầy đủ** (tải dữ liệu nếu thiếu, lọc, QA, baseline, gọi API cho câu chưa có trong cache, đánh giá, tính KPI và vẽ biểu đồ):
  ```bash
  python run_pipeline.py --config configs/vienna.yaml
  ```
- **Chạy ở chế độ không cần khóa API** (đọc trực tiếp kết quả LLM đã lưu tại `data/processed/llm_cache.jsonl`):
  ```bash
  python run_pipeline.py --config configs/vienna.yaml --skip-llm
  ```

### Chạy từng bước riêng lẻ (khi cần kiểm tra từng mô-đun)

```bash
python src/download.py --config configs/vienna.yaml
python src/filter.py
python src/qa.py
python src/baseline.py
python src/llm.py --config configs/vienna.yaml --skip-llm
python src/evaluate.py
python src/kpi.py
python src/viz.py
```

---

## 5. Danh sách tệp kết quả đầu ra để đối chiếu

Sau khi chạy xong `run_pipeline.py`, toàn bộ số liệu và hình ảnh để đối chiếu với báo cáo PDF được lưu tại:

| Nhóm kết quả | Đường dẫn tệp | Nội dung chính |
| :--- | :--- | :--- |
| **Kiểm định chất lượng (QA)** | `reports/qa_report.csv` | Số dòng vi phạm từng quy tắc QA (kể cả cờ ngoại lai IQR và lỗi mã hóa MacRoman) và giá trị KPI trước/sau làm sạch |
| **Đánh giá mô hình LLM** | `reports/llm_eval_report.csv` | Bảng điểm Accuracy, Macro-F1, Micro-F1, Recall lớp `mixed` kèm khoảng tin cậy 95% (Wilson CI) của Baseline và Gemini LLM trên 100 nhãn tay |
| **Phân tích lỗi & Ma trận nhầm lẫn** | `reports/llm_error_analysis.csv`<br>`reports/llm_error_types_summary.csv`<br>`reports/llm_error_summary.csv`<br>`reports/confusion_*_sentiment.csv` | Chi tiết các câu dự đoán lệch so với nhãn tay, phân loại nguyên nhân lỗi (`error_type`), thống kê tỷ lệ từng nhóm lỗi và ma trận nhầm lẫn |
| **Ước tính chi phí API** | `reports/llm_cost_estimate.csv` | Lượng token trung bình thực tế từ log và chi phí ước tính cho toàn bộ review thành phố |
| **Bảng chỉ số KPI** | `data/processed/kpi_1_price_by_area.csv` → `kpi_7_price_by_neighbourhood.csv`<br>`reports/kpi_definitions.csv` | 7 bảng số liệu KPI (giá thuê, tỷ lệ ngày không trống xấp xỉ, tăng trưởng review, chủ nhà nhiều phòng, cảm xúc & khía cạnh từ LLM) và bảng định nghĩa KPI gắn với từng bên liên quan |
| **Biểu đồ trực quan hóa** | `figures/fig1_price_by_area.png`<br>`figures/fig1b_price_boxplot_log.png`<br>`figures/fig2_occupancy_trend.png`<br>`figures/fig3_review_activity.png`<br>`figures/fig3b_monthly_reviews.png`<br>`figures/fig4_host_professionalization.png`<br>`figures/fig5_llm_sentiment_by_area.png`<br>`figures/fig6_llm_aspects_diverging.png`<br>`figures/fig7_price_map.png` | Bộ 9 biểu đồ sử dụng trực tiếp trong báo cáo phân tích |