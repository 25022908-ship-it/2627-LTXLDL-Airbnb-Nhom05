# AI_USAGE.md

Báo cáo minh bạch việc sử dụng Trí tuệ Nhân tạo (AI) của **Nhóm 05** trong Bài tập lớn môn *Lập trình xử lý dữ liệu* (`DSE2049-C04-G05-vienna`).

---

## Công cụ đã dùng (vào việc gì)

1. **Google Gemini (giao diện hội thoại):**
   - Giải thích khái niệm và tra cứu cú pháp `pandas`, `pathlib`, `requests`, `matplotlib`, `pydantic`.
   - Hỗ trợ viết khung mã nguồn ban đầu (boilerplate) cho các mô-đun `src/download.py`, `src/filter.py`, `src/qa.py`, `src/baseline.py`, `src/llm.py`, `src/evaluate.py`, `src/kpi.py`, `src/viz.py` và `run_pipeline.py`.
   - Gỡ lỗi ngoại lệ (`AttributeError`, `KeyError`, `MergeError`, cảnh báo font/Regex).
2. **Claude:**
   - Rà soát lỗi cú pháp và lỗi logic; đối chiếu code với đề bài tập lớn và nội dung các bài giảng (Bài 5–14); gợi ý hướng sửa để nhóm tự chỉnh và gửi lại kiểm tra.
   - Tư vấn chọn dạng biểu đồ theo Bài 12–13 và giải thích bảng kết quả đánh giá.
3. **Gemini API (`gemini-3.5-flash-lite` qua thư viện `google-genai` trong `src/llm.py`):**
   - Dùng trực tiếp trong quy trình để trích xuất có cấu trúc (cảm xúc `sentiment`, khía cạnh khen `aspects_pos`, khía cạnh chê `aspects_neg`, ngôn ngữ `lang`) trên tập 1.000 bình luận mẫu.

---

## Prompt then chốt (5-10): bước nào, vào tệp nào

| STT | Giai đoạn / Bước | Tệp áp dụng | Nội dung Prompt then chốt |
| :-- | :--- | :--- | :--- |
| 1 | **Tải dữ liệu an toàn** | `src/download.py` | *"Viết kịch bản tải 3 tệp `.csv.gz` và `neighbourhoods.geojson` theo 4 mốc chụp trong `configs/vienna.yaml`, tải qua tệp tạm `.part` rồi đổi tên để tránh hỏng file và kiểm tra đủ cột khóa bắt buộc."* |
| 2 | **Lọc dữ liệu & tối ưu RAM** | `src/filter.py` | *"Tệp `calendar.csv.gz` rất nặng. Hãy viết hàm lọc theo danh sách `listing_id` bằng `chunksize=500000` và `usecols`, phân biệt rõ `id` với `listing_id` / `review_id` khi nối bảng và lưu ra `.parquet`."* |
| 3 | **Kiểm định chất lượng (QA)** | `src/qa.py` | *"Tách riêng bộ quy tắc QA cho 3 bảng `listings`, `calendar`, `reviews` bao phủ đủ 5 nhóm lỗi (thiếu, ngoại lai, trùng lặp, kiểu sai, danh mục lệch), gắn cờ `flag_*` và `is_clean` thay vì xóa dòng, đối chiếu kỳ vọng cấu hình và xuất `reports/qa_report.csv`."* |
| 4 | **Phương pháp đối chứng** | `src/baseline.py` | *"Xây dựng bộ phân loại dựa trên Regex và từ điển từ khóa đa ngôn ngữ (Anh, Đức, Pháp, Tây Ban Nha) có xử lý cửa sổ phủ định để phân loại cảm xúc, 7 khía cạnh chuẩn và ngôn ngữ làm mốc đối chứng."* |
| 5 | **Trích xuất văn bản bằng LLM** | `src/llm.py` | *"Gọi Gemini API theo lô 20 bình luận với `Pydantic` (`BatchReviewExtraction`) để ép kiểu JSON đầu ra, ghi bộ nhớ đệm nối tiếp vào `llm_cache.jsonl`, ghi log token thực tế và hỗ trợ cờ `--skip-llm`."* |
| 6 | **Đánh giá & phân tích lỗi** | `src/evaluate.py` | *"Viết kịch bản đối chiếu dự đoán của Baseline và LLM với 100 câu nhãn tay (`nhan_tay_100.csv`), tính Accuracy, Macro-F1, Micro-F1, xuất ma trận nhầm lẫn, bảng câu đoán sai và ước tính chi phí token cho toàn bộ thành phố."* |
| 7 | **Tính KPI & vẽ biểu đồ** | `src/kpi.py`, `src/viz.py` | *"Tính 7 bảng KPI từ dữ liệu sạch sau QA (chỉ tính `price` ở mốc cuối, dùng trung vị và IQR) và vẽ 9 biểu đồ PNG tự động tính số liệu động trên tiêu đề, không dùng trục Y kép, có cỡ mẫu `n` và dòng ghi nguồn."* |
| 8 | **Rà soát toàn bộ code với đề và bài giảng** | toàn bộ `src/`, `run_pipeline.py` | *"Đây là toàn bộ code trong `src`. Hãy đối chiếu với nội dung bài tập lớn và lý thuyết các bài học, cho biết code có phải chỉnh sửa không; chỉ nêu chỗ cần sửa và hướng sửa ngắn."* |

---

## AI sai ở đâu (ví dụ thật)

1. **Dùng chung một bộ quy tắc QA cho cả 3 bảng khác cấu trúc (`src/qa.py`):**
   - *Lỗi của AI:* Bản nháp đầu dùng chung một hàm kiểm tra cột `price_num`, `room_type` cho cả `calendar.parquet` và `reviews.parquet`, gây `KeyError`.
   - *Cách sửa:* Tách thành 3 hàm `build_listings_rules`, `build_calendar_rules`, `build_reviews_rules` theo từng bảng.
2. **Giả định cột `price` luôn là chuỗi (`src/qa.py`):**
   - *Lỗi của AI:* `df["price"].str.replace("$", "")` văng `AttributeError` ở mốc `2025-12-16` vì cột `price` trống 100% nên pandas đọc thành `float64`.
   - *Cách sửa:* Viết `clean_price(val)` kiểm tra `pd.isna(val)` và kiểu dữ liệu trước khi xử lý chuỗi.
3. **Trả về sai kiểu khi thiếu `GEMINI_API_KEY` (`src/llm.py`):**
   - *Lỗi của AI:* Khi đổi `fetch_missing_reviews_via_api` sang trả về `(cache_map, api_ok)`, AI quên sửa nhánh `if missing_df.empty: return cache_map` (trả 1 giá trị thay vì 2) và đặt `strict=not skip_llm`, nên pipeline vẫn dừng khi chạy không có khóa API (trái đề: giảng viên không có khóa).
   - *Cách sửa:* Trả `return cache_map, True` ở nhánh đó, dùng `cache_map, api_ok = ...` trong `run_llm` và đặt `strict=(not skip_llm and api_ok)`.
4. **Regex baseline gây nhầm ngữ cảnh (`src/baseline.py`):**
   - *Lỗi của AI:* Dùng nhóm bắt `(...)` trong `str.contains()` gây `UserWarning`; đưa `"hot"` vào từ chê khiến `"hot shower"` bị tính là chê; đưa `"la"` vào nhận diện ngôn ngữ gây trùng Pháp/Tây Ban Nha. Khi sửa dấu nháy cong `’`, AI còn viết `re.sub(..., str(text), ...)` nên ghi đè kết quả vừa chuẩn hóa, `wasn’t` vẫn không khớp phủ định.
   - *Cách sửa:* Đổi sang nhóm không bắt `(?:...)`, thay `"hot"` bằng `"too hot"`, bỏ `"la"`, thêm cửa sổ phủ định 3 từ và cho `re.sub` chạy trên chuỗi đã chuẩn hóa.
5. **Ghi cứng kết luận trong tiêu đề biểu đồ, font và cửa sổ trượt (`src/viz.py`):**
   - *Lỗi của AI:* Viết cứng `"37%–40%"`, `"Trên 85%"` vào tiêu đề (sai khi đổi thành phố/mốc chụp); đặt font `Segoe UI` (không có trên Linux); tính `rolling(3, min_periods=1)` sau `.tail(36)`.
   - *Cách sửa:* Tính min/max/đỉnh từ bảng KPI rồi đưa vào f-string; đưa `DejaVu Sans` lên đầu; tính `rolling(3, min_periods=3)` trên cả chuỗi rồi mới cắt 36 tháng.
6. **Điền sẵn nhãn tay bằng dự đoán của Baseline (`src/baseline.py`):**
   - *Lỗi của AI:* Bản đầu của `init_hand_label_file_if_missing` copy `base_*` sang `human_*`, gây thiên lệch neo và làm điểm Baseline cao giả tạo.
   - *Cách sửa:* Để trống các cột `human_*`; `evaluate.py` có `validate_hand_labels()` báo lỗi nếu còn ô trống hoặc nhãn ngoài tập cho phép; đọc nhãn bằng `encoding="utf-8-sig"`.
7. **Tự đổi tên model và quên truyền tham số (`src/llm.py`):**
   - *Lỗi của AI:* `resolve_gemini_model_name` âm thầm đổi `gemini-3.5-flash-lite` trong config thành `gemini-2.5-flash-lite`. Khi thêm tham số `city`, AI chỉ sửa chữ ký hàm, lệnh gọi `build_batch_prompt(...)` không truyền `city` nên prompt vẫn ghi "địa phương". Thời gian nghỉ giữa các lô chỉ 1 giây, vượt hạn mức 15 lời gọi/phút của gói miễn phí.
   - *Cách sửa:* Dùng đúng tên model trong config; truyền `city=city`; đổi `time.sleep` thành 4 giây.
8. **Thiếu cột thì báo "0 vi phạm" và báo cáo QA tự mâu thuẫn (`src/qa.py`):**
   - *Lỗi của AI:* `empty_mask` trả toàn `False` khi thiếu cột nên báo cáo ghi 0 lỗi; `pd.to_datetime` không có `format`; hai cờ chỉ ghi nhận (`flag_lech_id_giua_cac_moc`, `flag_lech_review_giua_cac_moc`) vẫn có `kpi_sau_qa` như thể đã loại dòng, trong khi `hanh_dong` ghi "KHÔNG loại"; khóa `RULE_DOCS` bị lặp nên bản sau ghi đè bản trước; khung tọa độ và `A01`–`A08` ghi cứng.
   - *Cách sửa:* `validate_required_columns()` ném `ValueError`; thêm `format="%Y-%m-%d"`; cờ `info_flags` giữ `kpi_sau_qa = kpi_truoc_qa`; xóa khóa lặp; đưa `lat_bounds`, `lon_bounds`, `valid_areas` vào `configs/vienna.yaml`.
9. **Lỗi trình bày biểu đồ (`src/viz.py`):**
   - *Lỗi của AI:* Hình 3 dùng trục y kép (lỗi điển hình ở Bài 13); hình 4 dùng `fillna(0)` nên vẽ cột 0 cho nhóm thiếu dữ liệu giá; hình 7 dùng `set_array`/`set_clim` làm quận thiếu dữ liệu không còn màu xám; hình thiếu cỡ mẫu `n` và dòng nguồn.
   - *Cách sửa:* Tách hình 3 thành hai khung chung trục x; bỏ `fillna(0)`; tô màu thủ công bằng `Normalize` và tô xám quận `n < 10`; thêm `n` và dòng "Nguồn: Inside Airbnb, mốc …".
10. **Điền giả dữ liệu thiếu và dùng Baseline thay thế LLM (`src/llm.py`, `src/kpi.py`):**
    - *Lỗi của AI:* Bản ghi LLM hỏng hoặc chưa có trong cache bị gán `positive`; KPI 5 tự chuyển sang Baseline khi không có `llm_results.parquet`. Cả hai làm số liệu "từ LLM" sai mà không báo.
    - *Cách sửa:* Bản ghi không hợp lệ giữ `None` với `llm_ok=False`; KPI 5 và 6 chỉ tính dòng `llm_ok`; thêm `check_llm_coverage()` báo lỗi khi độ phủ dưới ngưỡng; KPI không còn dùng Baseline thay thế.

---

## Nhóm kiểm chứng thế nào

1. **Đối chiếu với thông số kiểm tra của đề bài (`src/qa.py`):**
   - Hàm `check_listings_expectations()` đối chiếu tự động cả 4 mốc chụp: `1.000` listings (`875` Entire home/apt, `125` Private room), `365.000` dòng calendar, `5.000` reviews; số phòng chưa có đánh giá lần lượt `50, 30, 25, 25`; mỗi vùng `A01`–`A08` có `120–130` chỗ ở; mốc cuối thiếu đúng `20` dòng giá.
   - Mốc `2025-12-16` thiếu `1.000/1.000` giá: nhóm kiểm tra trực tiếp tệp gốc bằng `df['price'].notna().sum()` để xác nhận lỗi từ nguồn trước khi chỉ tính KPI giá ở mốc cuối.
2. **Kiểm chứng quy tắc QA và phép ghép bảng:**
   - `reports/qa_report.csv` xuất đủ tên quy tắc, điều kiện, lý do, hành động và KPI trước/sau QA (kể cả quy tắc có 0 vi phạm).
   - Mọi phép `merge` trong `filter.py`, `evaluate.py`, `kpi.py` dùng `validate=...` hoặc đối chiếu số dòng trước/sau.
3. **Gán nhãn thủ công độc lập và đo lường khách quan (`src/evaluate.py`):**
   - Bốn thành viên gán nhãn độc lập cho cùng bộ 100 câu trong `data/samples/nhan_tay_100.csv` (đồng thuận 82/100 câu, thảo luận chốt các câu bất đồng), không điền sẵn gợi ý từ mô hình để tránh thiên lệch neo.
   - `evaluate.py` đo Accuracy, Macro-F1, Micro-F1 của Baseline và Gemini trên cùng tập 100 câu, xuất ma trận nhầm lẫn và phân loại nguyên nhân sai trong `reports/llm_error_analysis.csv`.
4. **Rà soát chéo bằng AI rồi tự sửa và kiểm lại:**
   - Mỗi lần AI nêu lỗi, nhóm tự sửa, chạy lại `python run_pipeline.py --skip-llm`, rồi mới gửi lại phần đã đổi để rà soát; các tiêu đề hình được đối chiếu thủ công với `kpi_*.csv`.
5. **Kiểm thử trên môi trường sạch (clean-room test):**
   - Trước khi gắn tag `final`, nhóm clone repo sang thư mục mới, dựng môi trường theo `.python-version` và `requirements.txt`, chạy đúng một lệnh `python run_pipeline.py --config configs/vienna.yaml --skip-llm` để xác nhận bảng số liệu và biểu đồ tái tạo khớp với báo cáo PDF.
