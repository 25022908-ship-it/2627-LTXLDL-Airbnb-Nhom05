# Từ điển dữ liệu (Data Dictionary) – Airbnb Vienna, Nhóm 05

Mã dữ liệu: `DSE2049-C04-G05-vienna` · Nguồn: Inside Airbnb (CC BY 4.0) · 4 mốc chụp: `2025-09-14`, `2025-12-16`, `2026-03-24`, `2026-06-20` · Tiền tệ: EUR.

**Cách đọc.** Kiểu dữ liệu lấy từ schema của tệp parquet mốc `2026-06-20` (kiểu pandas). *(Biến phái sinh)* là cột do pipeline tạo ra; cột còn lại là cột gốc của Inside Airbnb hoặc của danh sách phân công. Quy mô mỗi mốc: 1.000 chỗ ở, 365.000 dòng lịch (1.000 × 365 ngày), 5.000 đánh giá (cố định giữa các mốc).

**Lưu ý về schema giữa các mốc.** Bảng `calendar` và `reviews` có schema giống nhau ở cả 4 mốc; bảng `listings` thì khác: mốc `2025-09-14` có 98 cột, `2025-12-16` có 104 cột, hai mốc sau có 109 cột (các cột `price_quote_*` chỉ có ở hai mốc sau). Một cột văn bản có kiểu `float64` nghĩa là toàn bộ giá trị của cột đó trong mốc ấy bị trống; ví dụ `price` ở mốc `2025-12-16` có kiểu `float64` (trống 100%), đúng với kết quả QA.

## 1. Khóa và cột thời gian


| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `id` / `listing_id` | listings, calendar, reviews | int64 | Mã định danh | Khóa định danh chỗ ở: `listings.id` = `calendar.listing_id` = `reviews.listing_id`. Dùng để lọc theo danh sách phân công, nối bảng (kiểm tra quan hệ 1–1/1–n) và chống trùng. |
| `id` | reviews | int64 | Mã định danh | Khóa định danh đánh giá; trùng với `review_id` trong danh sách phân công và `id` trong kết quả LLM/baseline. |
| `last_scraped` | listings | string | YYYY-MM-DD | Ngày chụp dữ liệu của dòng. Chỉ có ở `listings` (không có trong `calendar`, `reviews`); cùng `id` tạo khóa chống trùng. |
| `last_scraped_clean` | listings | datetime64 | Ngày | *(Biến phái sinh)* `last_scraped` đã ép sang kiểu ngày; dòng không ép được bị gắn cờ. |
| `date` | calendar, reviews | string | YYYY-MM-DD | Ngày của dòng lịch (`calendar`) hoặc ngày đánh giá (`reviews`), giữ nguyên dạng chuỗi gốc. Mỗi mốc lịch phủ 365 ngày phía trước ngày chụp; ngày đánh giá của 5.000 đánh giá nằm trong 2012 đến 2025-09-13. |
| `date_clean` | calendar, reviews | datetime64 | Ngày | *(Biến phái sinh)* `date` đã ép sang kiểu ngày (`%Y-%m-%d`); dòng không ép được bị gắn cờ `flag_sai_kieu_ngay`. |


## 2. Bảng `listings` – cột dùng trong phân tích

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `analysis_area` | listings | string | A01 – A08 | *(Biến phái sinh)* Mã vùng phân tích ẩn danh, gán từ `data/samples/danh_sach_listing.csv` theo `listing_id`. Đơn vị tổng hợp của KPI 1–5. Giá trị ngoài `A01`–`A08` bị gắn cờ `flag_lech_vung`. |
| `neighbourhood_cleansed` | listings | string | Tên quận (23 quận) | Tên quận hành chính Vienna, dùng cho KPI 7 và bản đồ giá. Nguồn có lỗi bảng mã MacRoman (ví dụ `D\x9abling`); pipeline gắn cờ `flag_loi_ma_hoa_quan` rồi chuẩn hóa sang UTF-8 (`Döbling`). |
| `latitude`, `longitude` | listings | float64 | Độ (WGS84) | Tọa độ chỗ ở. Thiếu hoặc ngoài khung Vienna (lat 48,10–48,35; lon 16,15–16,60) bị gắn cờ `flag_ngoai_toa_do`. |
| `room_type` | listings | string | Entire home/apt, Private room, Shared room, Hotel room | Loại chỗ ở. Giá trị ngoài 4 loại chuẩn bị gắn cờ `flag_lech_loai_phong`. Dùng nhóm khi tính ngưỡng IQR và tỷ lệ nguyên căn (KPI 1). |
| `accommodates` | listings | int64 | Số khách | Sức chứa. Ngoài (0; 50] bị gắn cờ `flag_sai_suc_chua`. |
| `bedrooms` | listings | float64 | Số phòng ngủ | Kiểu `float64`, thường do có giá trị thiếu. Ngoài [0; 20] bị gắn cờ `flag_sai_suc_chua`. |
| `price` | listings | string | Chuỗi giá (ví dụ `$120.00`) | Giá một đêm dạng chuỗi gốc. Mốc `2025-12-16` kiểu `float64` và trống hoàn toàn nên KPI giá chỉ tính ở mốc cuối `2026-06-20`. |
| `price_num` | listings | float64 | EUR/đêm | *(Biến phái sinh)* `price` đã bỏ ký hiệu tiền tệ và dấu phân tách rồi ép sang số. Trống hoặc không ép được bị gắn cờ `flag_thieu_gia` / `flag_sai_kieu_gia`; ≤ 0 hoặc > 5.000 bị gắn cờ `flag_gia_ngoai_lai`. Đầu vào KPI 1, 4, 7. |
| `minimum_nights`, `maximum_nights` | listings | float64 | Số đêm | Số đêm tối thiểu/tối đa. `minimum_nights` ≤ 0 hoặc > 365 bị gắn cờ `flag_dem_ngoai_lai`. Kiểu `float64` ở hai mốc cuối (hai mốc đầu là `int64`), thường do có giá trị thiếu. |
| `host_id` | listings | int64 | Mã định danh | Mã chủ nhà; thiếu bị gắn cờ `flag_sai_thong_tin_host`. Đếm số chủ nhà duy nhất ở KPI 4. |
| `calculated_host_listings_count` | listings | int64 | Số chỗ ở | Số chỗ ở của chủ nhà trong toàn bộ dữ liệu thành phố. ≥ 2 được coi là chủ nhà nhiều phòng (chuyên nghiệp hóa, KPI 4); < 1 bị gắn cờ `flag_sai_thong_tin_host`. |
| `number_of_reviews` | listings | int64 | Lượt đánh giá | Tổng đánh giá tích lũy của chỗ ở. KPI 3 cộng tổng và đếm chỗ ở có giá trị 0 (chưa có đánh giá). |
| `review_scores_rating` | listings | float64 | Điểm, thang 0–5 | Điểm đánh giá tổng thể; có giá trị thiếu khi chưa có đánh giá. Ngoài [0; 5] hoặc có điểm khi `number_of_reviews` = 0 bị gắn cờ `flag_sai_logic_review`. |


## 3. Bảng `listings` – cột gốc không dùng trong phân tích

Các cột này được giữ nguyên từ Inside Airbnb và **không đi vào KPI**. Kiểu dữ liệu ghi cạnh tên cột (mốc `2026-06-20`); cột kiểu `float64` mà ở mốc khác là văn bản thì trống hoàn toàn trong mốc này.

| Nhóm cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Danh sách cột (kiểu) |
|---|---|---|---|---|
| Thông tin chủ nhà (`host_*`, `hosts_time_*`) | listings | xem danh sách | – | `host_url` (string); `host_profile_id` (float64); `host_profile_url` (string); `host_name` (string); `host_since` (float64); `hosts_time_as_user_years` (float64); `hosts_time_as_user_months` (float64); `hosts_time_as_host_years` (float64); `hosts_time_as_host_months` (float64); `host_location` (string); `host_about` (string); `host_response_time` (float64); `host_response_rate` (float64); `host_acceptance_rate` (float64); `host_is_superhost` (string); `host_thumbnail_url` (float64); `host_picture_url` (string); `host_neighbourhood` (float64); `host_listings_count` (float64); `host_total_listings_count` (float64); `host_verifications` (float64); `host_has_profile_pic` (string); `host_identity_verified` (string) |
| Báo giá mẫu (`price_quote_*`) | listings | xem danh sách | – | `price_quote_checkin_date` (string); `price_quote_checkout_date` (string); `price_quote_total_price` (float64); `price_quote_price_per_night` (float64); `price_quote_raw` (string) |
| Khả dụng (`availability_*`, `has_availability`, `calendar_*`) | listings | xem danh sách | – | `calendar_updated` (float64); `has_availability` (string); `availability_30` (int64); `availability_60` (int64); `availability_90` (int64); `availability_365` (int64); `calendar_last_scraped` (string); `availability_eoy` (int64) |
| Biến thể số đêm (`minimum_*_nights`, `*_nights_avg_ntm`) | listings | xem danh sách | – | `minimum_minimum_nights` (float64); `maximum_minimum_nights` (float64); `minimum_maximum_nights` (float64); `maximum_maximum_nights` (float64); `minimum_nights_avg_ntm` (float64); `maximum_nights_avg_ntm` (float64) |
| Điểm đánh giá và thống kê đánh giá khác | listings | xem danh sách | – | `number_of_reviews_ltm` (int64); `number_of_reviews_l30d` (int64); `number_of_reviews_ly` (int64); `first_review` (string); `last_review` (string); `review_scores_accuracy` (float64); `review_scores_cleanliness` (float64); `review_scores_checkin` (float64); `review_scores_communication` (float64); `review_scores_location` (float64); `review_scores_value` (float64); `reviews_per_month` (float64) |
| Số chỗ ở của chủ nhà theo loại (`calculated_host_listings_count_*`) | listings | xem danh sách | – | `calculated_host_listings_count_entire_homes` (int64); `calculated_host_listings_count_private_rooms` (int64); `calculated_host_listings_count_shared_rooms` (int64) |
| Ước tính doanh thu và công suất (`estimated_*`) | listings | xem danh sách | – | `estimated_occupancy_l365d` (int64); `estimated_revenue_l365d` (float64) |
| Mô tả, đường dẫn và đặc điểm chỗ ở (còn lại) | listings | xem danh sách | – | `listing_url` (string); `scrape_id` (int64); `source` (string); `name` (string); `description` (string); `neighborhood_overview` (float64); `picture_url` (string); `neighbourhood` (float64); `neighbourhood_group_cleansed` (float64); `property_type` (string); `bathrooms` (float64); `bathrooms_text` (string); `beds` (float64); `amenities` (string); `license` (float64); `instant_bookable` (float64) |


## 4. Bảng `calendar`

Chỉ giữ 5 cột gốc cần dùng; tệp gốc được đọc theo lô 500.000 dòng để tiết kiệm bộ nhớ.

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `listing_id` | calendar | int64 | Mã định danh | Khóa chỗ ở (xem mục 1). |
| `date` | calendar | string | YYYY-MM-DD | Ngày của dòng lịch (xem mục 1). Trùng cặp (`listing_id`, `date`) bị gắn cờ `flag_trung_ngay_phong`. |
| `available` | calendar | string | `t` / `f` | `t` = còn trống; `f` = không khả dụng (đã đặt **hoặc** chủ nhà tự khóa). KPI 2 = số ngày `f` / tổng ngày lịch sạch, là chỉ báo gần đúng của mức khai thác, không phải tỷ lệ đặt phòng thực tế. Trống bị gắn cờ `flag_thieu_available`; ngoài {`t`,`f`} bị gắn cờ `flag_lech_available`. |
| `minimum_nights`, `maximum_nights` | calendar | int64 | Số đêm | Số đêm tối thiểu/tối đa áp dụng cho ngày đó. `minimum_nights` ≤ 0 hoặc > `maximum_nights` bị gắn cờ `flag_sai_logic_dem`. |
| `date_clean` | calendar | datetime64 | Ngày | *(Biến phái sinh)* xem mục 1. |
| `flag_*`, `is_clean` | calendar | bool | True / False | *(Biến phái sinh)* Cờ QA (mục 7). `is_clean` = True khi dòng không vi phạm cờ cấu trúc nào. |


## 5. Bảng `reviews`

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `id` | reviews | int64 | Mã định danh | Mã đánh giá; trùng khóa bị gắn cờ `flag_trung_id`. |
| `listing_id` | reviews | int64 | Mã định danh | Chỗ ở được đánh giá. Cùng `id` nhưng `listing_id` khác mốc đầu bị gắn cờ `flag_lech_review_giua_cac_moc` (chỉ ghi nhận). |
| `date` | reviews | string | YYYY-MM-DD | Ngày đánh giá (xem mục 1). Sau ngày chụp bị gắn cờ `flag_ngay_tuong_lai`. |
| `reviewer_id`, `reviewer_name` | reviews | int64 / string | Mã / tên | Người đánh giá; không dùng trong phân tích. |
| `comments` | reviews | string | Văn bản | Nội dung đánh giá, có thể chứa thẻ HTML như `<br/>`. Trống bị gắn cờ `flag_thieu_noi_dung`; chứa thông báo tự động của Airbnb ("automated posting") bị gắn cờ `flag_review_tu_dong`. Cả hai bị loại khỏi mẫu LLM và baseline. |
| `Mẫu LLM` | reviews | string | `x` hoặc trống | Từ danh sách phân công: `x` = thuộc mẫu 1.000 đánh giá chạy LLM (4.000 dòng còn lại để trống). |
| `Nhãn tay` | reviews | string | `x` hoặc trống | Từ danh sách phân công: `x` = thuộc 100 đánh giá gán nhãn tay (nằm trong mẫu LLM). |
| `date_clean` | reviews | datetime64 | Ngày | *(Biến phái sinh)* xem mục 1. |
| `flag_*`, `is_clean` | reviews | bool | True / False | *(Biến phái sinh)* Cờ QA (mục 7). |


## 6. Kết quả baseline và LLM (`data/processed/`)

Hai tệp (`baseline_results.*`, `llm_results.*`) có cùng các cột của bảng `reviews` (mục 5), cộng thêm các cột dưới đây.

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `base_sentiment` | baseline_results | string | positive / mixed / negative | Cảm xúc dự đoán bằng từ khóa/regex đa ngôn ngữ (cửa sổ phủ định 3 từ). Phương pháp đối chứng. |
| `base_aspects_pos`, `base_aspects_neg` | baseline_results | string | Danh sách phân tách dấu phẩy | Khía cạnh được khen / bị chê theo baseline. |
| `base_lang` | baseline_results | string | Mã ngôn ngữ (ví dụ `en`) | Ngôn ngữ dự đoán theo từ khóa. |
| `has_<khía cạnh>` (7 cột) | baseline_results | bool | True / False | Văn bản có nhắc tới khía cạnh: `has_vị_trí`, `has_sạch_sẽ`, `has_chủ_nhà`, `has_giá_trị`, `has_tiện_nghi`, `has_độ_ồn`, `has_an_toàn`. |
| `llm_sentiment` | llm_results | string | positive / mixed / negative | Cảm xúc do LLM (gemini-3.5-flash-lite) gán, trả JSON theo schema Pydantic. |
| `llm_aspects_pos`, `llm_aspects_neg` | llm_results | string | Danh sách phân tách dấu phẩy (ví dụ `vị trí, chủ nhà`) | Khía cạnh được khen / bị chê trong 7 khía cạnh: vị trí, sạch sẽ, chủ nhà, giá trị, tiện nghi, độ ồn, an toàn. |
| `llm_lang` | llm_results | string | Mã ngôn ngữ | Ngôn ngữ LLM nhận diện (cả 1.000 đánh giá là `en`). |
| `llm_ok` | llm_results | bool | True / False | Bản ghi LLM hợp lệ. Bản ghi lỗi để trống, không điền giả, và bị loại khỏi KPI 5–6. |
| `llm_pos_<khía cạnh>`, `llm_neg_<khía cạnh>` (14 cột) | llm_results | bool | True / False | Cột nhị phân khen/chê cho từng khía cạnh (ví dụ `llm_pos_vị_trí`, `llm_neg_độ_ồn`), tách từ hai cột danh sách; đầu vào KPI 6. |


Tệp phụ: `llm_cache.jsonl` (mỗi dòng: `review_id` int, `sentiment`, `aspects_pos`, `aspects_neg`, `lang` – dùng để chạy lại không cần khóa API) và `llm_usage.jsonl` (mỗi lời gọi API: `model`, `n_reviews`, `prompt_tokens`, `output_tokens`).


## 7. Cờ QA (`flag_*`, `is_clean`)

Cờ `True` = dòng vi phạm quy tắc. **Loại khỏi KPI** = dòng bị bỏ khỏi phép tính (`is_clean` = False; `is_clean_price` = False nếu là cờ về giá). **Chỉ ghi nhận** = giữ lại trong KPI. Tác động từng quy tắc theo mốc nằm trong `reports/qa_report.csv`.

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `flag_thieu_gia` | listings | bool | True / False | *(Biến phái sinh)* `price` trống. Hành động: Loại khỏi KPI giá. |
| `flag_sai_kieu_gia` | listings | bool | True / False | *(Biến phái sinh)* `price` có giá trị nhưng không ép được sang số. Hành động: Loại khỏi KPI giá. |
| `flag_gia_ngoai_lai` | listings | bool | True / False | *(Biến phái sinh)* `price_num` ≤ 0 hoặc > 5.000 € (`qa.price_max_eur`). Hành động: Loại khỏi KPI giá. |
| `flag_gia_ngoai_lai_iqr` | listings | bool | True / False | *(Biến phái sinh)* `price_num` > Q3 + 1,5×IQR theo từng `room_type`. Hành động: Chỉ ghi nhận. |
| `flag_dem_ngoai_lai` | listings | bool | True / False | *(Biến phái sinh)* `minimum_nights` ≤ 0 hoặc > 365 (`qa.min_nights_cap`). Hành động: Loại khỏi KPI. |
| `flag_sai_suc_chua` | listings | bool | True / False | *(Biến phái sinh)* `accommodates` ≤ 0 hoặc > 50; `bedrooms` < 0 hoặc > 20. Hành động: Loại khỏi KPI. |
| `flag_sai_logic_review` | listings | bool | True / False | *(Biến phái sinh)* Điểm ngoài [0; 5] hoặc có điểm khi `number_of_reviews` = 0. Hành động: Loại khỏi KPI. |
| `flag_sai_thong_tin_host` | listings | bool | True / False | *(Biến phái sinh)* Thiếu `host_id` hoặc `calculated_host_listings_count` < 1. Hành động: Loại khỏi KPI. |
| `flag_trung_id` | listings, reviews | bool | True / False | *(Biến phái sinh)* Trùng khóa (`id` + `last_scraped` ở listings; `id` ở reviews). Hành động: Loại khỏi KPI. |
| `flag_lech_loai_phong` | listings | bool | True / False | *(Biến phái sinh)* `room_type` ngoài 4 loại chuẩn. Hành động: Loại khỏi KPI. |
| `flag_ngoai_toa_do` | listings | bool | True / False | *(Biến phái sinh)* Tọa độ thiếu hoặc ngoài khung Vienna. Hành động: Loại khỏi KPI. |
| `flag_lech_vung` | listings | bool | True / False | *(Biến phái sinh)* Thiếu `neighbourhood_cleansed` hoặc `analysis_area` ngoài A01–A08. Hành động: Loại khỏi KPI. |
| `flag_lech_id_giua_cac_moc` | listings | bool | True / False | *(Biến phái sinh)* `id` không có ở mốc đầu tiên (listing mới, không phải lỗi). Hành động: Chỉ ghi nhận. |
| `flag_loi_ma_hoa_quan` | listings | bool | True / False | *(Biến phái sinh)* `neighbourhood_cleansed` chứa ký tự lỗi MacRoman (`\x80`–`\x9f`, `§`). Hành động: Chỉ ghi nhận; chuẩn hóa tên về UTF-8. |
| `flag_thieu_available` | calendar | bool | True / False | *(Biến phái sinh)* `available` trống. Hành động: Loại khỏi KPI. |
| `flag_lech_available` | calendar | bool | True / False | *(Biến phái sinh)* `available` ngoài {`t`, `f`}. Hành động: Loại khỏi KPI. |
| `flag_sai_kieu_ngay` | calendar, reviews | bool | True / False | *(Biến phái sinh)* Ngày có giá trị nhưng không ép được sang kiểu ngày. Hành động: Loại khỏi KPI. |
| `flag_sai_logic_dem` | calendar | bool | True / False | *(Biến phái sinh)* `minimum_nights` ≤ 0 hoặc > `maximum_nights`. Hành động: Loại khỏi KPI. |
| `flag_trung_ngay_phong` | calendar | bool | True / False | *(Biến phái sinh)* Trùng (`listing_id`, `date`). Hành động: Loại khỏi KPI. |
| `flag_thieu_noi_dung` | reviews | bool | True / False | *(Biến phái sinh)* `comments` trống. Hành động: Loại khỏi mẫu LLM / baseline. |
| `flag_ngay_tuong_lai` | reviews | bool | True / False | *(Biến phái sinh)* Ngày đánh giá sau ngày chụp. Hành động: Loại khỏi KPI. |
| `flag_review_tu_dong` | reviews | bool | True / False | *(Biến phái sinh)* `comments` chứa thông báo tự động ("automated posting"). Hành động: Loại khỏi mẫu LLM / baseline. |
| `flag_lech_review_giua_cac_moc` | reviews | bool | True / False | *(Biến phái sinh)* Cùng `id` nhưng `listing_id` khác mốc đầu. Hành động: Chỉ ghi nhận. |


| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `is_clean` | listings, calendar, reviews | bool | True / False | *(Biến phái sinh)* True khi dòng không vi phạm cờ "loại khỏi KPI" nào. Phép tính KPI chỉ dùng dòng `is_clean` = True. |
| `is_clean_price` | listings | bool | True / False | *(Biến phái sinh)* `is_clean` và không vi phạm cờ về giá (`flag_thieu_gia`, `flag_sai_kieu_gia`, `flag_gia_ngoai_lai`). Dùng cho KPI giá (1, 4, 7). |


## 8. Danh sách phân công và nhãn tay (`data/samples/`)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `Lớp`, `Nhóm`, `Mã dữ liệu` | danh_sach_listing, danh_sach_review | string, int64, string | Ví dụ `UET.DSE2049 4`, `5`, `DSE2049-C04-G05-vienna` | Thông tin lớp, nhóm, mã dữ liệu được phân công. |
| `listing_id` | danh_sach_listing, danh_sach_review, nhan_tay_100 | int64 | Mã định danh | 1.000 chỗ ở được phân công (khóa lọc dữ liệu). |
| `analysis_area` | danh_sach_listing | string | A01 – A08 | Vùng phân tích (8 vùng, khoảng 120–130 chỗ ở mỗi vùng). |
| `review_id` | danh_sach_review | int64 | Mã định danh | 5.000 đánh giá được phân công (khóa lọc). |
| `Mẫu LLM`, `Nhãn tay` | danh_sach_review | string | `x` hoặc trống | `x` ở 1.000 dòng (mẫu LLM) và 100 dòng (nhãn tay). |
| `id`, `listing_id`, `date`, `comments` | nhan_tay_100 | int64, int64, string, string | – | 100 đánh giá được gán nhãn tay, kèm nội dung. |
| `human_sentiment` | nhan_tay_100 | string | positive / mixed / negative | Nhãn cảm xúc chốt cuối sau khi 4 thành viên gán độc lập và thảo luận (hướng dẫn: `huong_dan_gan_nhan.md`; nhãn từng người: `raw_annotations/`). |
| `human_aspects_pos`, `human_aspects_neg` | nhan_tay_100 | string | Danh sách phân tách dấu phẩy; trống nếu không có | Khía cạnh được khen / bị chê theo nhãn tay (7 khía cạnh chuẩn). |
| `human_lang` | nhan_tay_100 | string | Mã ngôn ngữ | Ngôn ngữ theo nhãn tay. |


## 9. Bảng KPI (`data/processed/kpi_*.csv`)

Kiểu dữ liệu là kiểu khi đọc tệp CSV bằng pandas. Chỉ tính trên dòng sạch (`is_clean` / `is_clean_price`). Giá: EUR/đêm; tỷ lệ: phần trăm (0–100).


### `kpi_1_price_by_area.csv` – KPI 1: giá theo vùng (mốc cuối)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `analysis_area` | kpi_1_price_by_area | string | A01 – A08 | Vùng phân tích. |
| `so_phong_sach` | kpi_1_price_by_area | int64 | Số chỗ ở | Số chỗ ở sạch về giá (cỡ mẫu n). |
| `gia_trung_vi_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Giá trung vị `price_num`. |
| `gia_trung_binh_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Giá trung bình; nhạy với ngoại lai nên báo cáo ưu tiên trung vị. |
| `do_lech_chuan_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Độ lệch chuẩn của giá. |
| `gia_iqr_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Khoảng tứ phân vị Q3 − Q1. |
| `gia_thap_nhat_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Giá nhỏ nhất. |
| `gia_cao_nhat_eur` | kpi_1_price_by_area | float64 | EUR/đêm | Giá lớn nhất. |
| `ty_le_entire_home_pct` | kpi_1_price_by_area | float64 | % | Tỷ lệ chỗ ở loại `Entire home/apt`. |
| `diem_danh_gia_tb` | kpi_1_price_by_area | float64 | Điểm 0–5 | Trung bình `review_scores_rating`. |


### `kpi_2_occupancy_trend.csv` – KPI 2: tỷ lệ ngày không khả dụng theo thời gian

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `moc_thoi_gian` | kpi_2_occupancy_trend | string | YYYY-MM-DD | Mốc chụp. |
| `analysis_area` | kpi_2_occupancy_trend | string | A01 – A08, `ALL` | Vùng; dòng `ALL` là tổng hợp toàn mẫu. |
| `tong_ngay_lich` | kpi_2_occupancy_trend | int64 | Số ngày | Tổng số ngày lịch sạch. |
| `so_ngay_khong_kha_dung` | kpi_2_occupancy_trend | int64 | Số ngày | Số ngày `available` = `f`. |
| `ty_le_khong_kha_dung_pct` | kpi_2_occupancy_trend | float64 | % | Tỷ lệ = không khả dụng / tổng ngày lịch (chỉ báo gần đúng mức khai thác, gồm cả ngày chủ nhà tự khóa). |
| `so_dem_toi_thieu_trung_vi` | kpi_2_occupancy_trend | float64 | Số đêm | Trung vị `minimum_nights` của lịch. |


### `kpi_3_review_activity_trend.csv` – KPI 3: động lực đánh giá theo thời gian

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `moc_thoi_gian` | kpi_3_review_activity_trend | string | YYYY-MM-DD | Mốc chụp. |
| `so_phong_sach` | kpi_3_review_activity_trend | int64 | Số chỗ ở | Số chỗ ở sạch. |
| `so_phong_chua_co_review` | kpi_3_review_activity_trend | int64 | Số chỗ ở | Số chỗ ở có `number_of_reviews` = 0. |
| `so_phong_da_co_review` | kpi_3_review_activity_trend | int64 | Số chỗ ở | Số chỗ ở có `number_of_reviews` > 0. |
| `tong_review_tich_luy_tren_listings` | kpi_3_review_activity_trend | int64 | Lượt | Tổng `number_of_reviews`. |
| `trung_vi_so_review_moi_phong` | kpi_3_review_activity_trend | float64 | Lượt | Trung vị số đánh giá mỗi chỗ ở. |
| `diem_rating_trung_binh` | kpi_3_review_activity_trend | float64 | Điểm 0–5 | Trung bình `review_scores_rating`. |
| `diem_rating_trung_vi` | kpi_3_review_activity_trend | float64 | Điểm 0–5 | Trung vị `review_scores_rating`. |
| `so_review_hop_le_trong_mau` | kpi_3_review_activity_trend | int64 | Lượt | Số đánh giá hợp lệ trong mẫu cố định (5.000; giống nhau ở mọi mốc). |


### `kpi_4_host_professionalization.csv` – KPI 4: chuyên nghiệp hóa chủ nhà (mốc cuối)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `analysis_area` | kpi_4_host_professionalization | string | A01 – A08 | Vùng phân tích. |
| `tong_so_phong` | kpi_4_host_professionalization | int64 | Số chỗ ở | Tổng số chỗ ở của vùng. |
| `so_chu_nha_duy_nhat` | kpi_4_host_professionalization | int64 | Số chủ nhà | Số `host_id` khác nhau. |
| `so_phong_chu_nha_chuyen_nghiep` | kpi_4_host_professionalization | int64 | Số chỗ ở | Số chỗ ở của chủ nhà có `calculated_host_listings_count` ≥ 2. |
| `ty_le_chuyen_nghiep_pct` | kpi_4_host_professionalization | float64 | % | Tỷ lệ chỗ ở thuộc chủ nhà nhiều phòng. |
| `gia_median_chuyen_nghiep_eur` | kpi_4_host_professionalization | float64 | EUR/đêm | Giá trung vị của chủ nhà ≥ 2 chỗ ở. |
| `gia_median_ca_nhan_eur` | kpi_4_host_professionalization | float64 | EUR/đêm | Giá trung vị của chủ nhà 1 chỗ ở. |


### `kpi_5_llm_sentiment_by_area.csv` – KPI 5: cảm xúc khách theo vùng (LLM)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `analysis_area` | kpi_5_llm_sentiment_by_area | string | A01 – A08 | Vùng phân tích. |
| `so_binh_luan_mau` | kpi_5_llm_sentiment_by_area | int64 | Số bình luận | Số bình luận LLM hợp lệ của vùng. |
| `ty_le_positive_pct` | kpi_5_llm_sentiment_by_area | float64 | % | Tỷ lệ positive. |
| `ty_le_mixed_pct` | kpi_5_llm_sentiment_by_area | float64 | % | Tỷ lệ mixed. |
| `ty_le_negative_pct` | kpi_5_llm_sentiment_by_area | float64 | % | Tỷ lệ negative. |
| `diem_cam_xuc_rong_net_pct` | kpi_5_llm_sentiment_by_area | float64 | % (điểm) | Điểm ròng = positive − negative. |
| `ty_le_tieng_anh_pct` | kpi_5_llm_sentiment_by_area | float64 | % | Tỷ lệ bình luận tiếng Anh theo LLM. |
| `ty_le_tieng_duc_pct` | kpi_5_llm_sentiment_by_area | float64 | % | Tỷ lệ bình luận tiếng Đức theo LLM. |
| `canh_bao_n_nho` | kpi_5_llm_sentiment_by_area | bool | True / False | True nếu dưới 30 bình luận (cỡ mẫu nhỏ). |


### `kpi_6_llm_aspects_summary.csv` – KPI 6: khen/chê theo khía cạnh (LLM)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `khia_canh` | kpi_6_llm_aspects_summary | string | 7 khía cạnh | Khía cạnh: vị trí, sạch sẽ, chủ nhà, giá trị, tiện nghi, độ ồn, an toàn. |
| `so_luot_khen` | kpi_6_llm_aspects_summary | int64 | Số bình luận | Số bình luận khen khía cạnh. |
| `ty_le_khen_pct` | kpi_6_llm_aspects_summary | float64 | % | Tỷ lệ khen trên số bình luận hợp lệ. |
| `so_luot_che` | kpi_6_llm_aspects_summary | int64 | Số bình luận | Số bình luận chê khía cạnh. |
| `ty_le_che_pct` | kpi_6_llm_aspects_summary | float64 | % | Tỷ lệ chê trên số bình luận hợp lệ. |
| `tong_luot_nhac` | kpi_6_llm_aspects_summary | int64 | Số bình luận | Khen + chê. |
| `ty_le_nhac_toi_pct` | kpi_6_llm_aspects_summary | float64 | % | Tỷ lệ bình luận nhắc tới khía cạnh. |
| `ty_le_khen_tren_tong_nhac_pct` | kpi_6_llm_aspects_summary | float64 | % | Khen / (khen + chê). |
| `so_binh_luan_hop_le` | kpi_6_llm_aspects_summary | int64 | Số bình luận | Mẫu số: số bình luận hợp lệ. |


### `kpi_7_price_by_neighbourhood.csv` – KPI 7: giá theo quận (mốc cuối)

| Tên cột | Bảng | Kiểu dữ liệu | Đơn vị / Giá trị | Ý nghĩa & Ghi chú xử lý |
|---|---|---|---|---|
| `neighbourhood_cleansed` | kpi_7_price_by_neighbourhood | string | 23 quận | Tên quận đã chuẩn hóa UTF-8. |
| `so_phong_sach` | kpi_7_price_by_neighbourhood | int64 | Số chỗ ở | Số chỗ ở sạch về giá. |
| `gia_trung_vi_eur` | kpi_7_price_by_neighbourhood | float64 | EUR/đêm | Giá trung vị. |
| `gia_trung_binh_eur` | kpi_7_price_by_neighbourhood | float64 | EUR/đêm | Giá trung bình. |
| `canh_bao_n_nho` | kpi_7_price_by_neighbourhood | bool | True / False | True nếu dưới 10 chỗ ở. |


## 10. Báo cáo (`reports/`)

| Tệp | Thư mục | Kiểu dữ liệu | Đơn vị / Giá trị | Nội dung |
|---|---|---|---|---|
| `qa_report.csv` | reports | mixed | – | Tác động từng quy tắc QA theo mốc và bảng: `moc_thoi_gian`, `bang`, `quy_tac`, `dieu_kien`, `ly_do`, `hanh_dong`, `so_dong_vi_pham`, `ty_le_phan_tram`, `kpi_ten`, `kpi_truoc_qa`, `kpi_sau_qa` (dạng `trung vị (trung bình)`). Dòng `TONG_HOP_QA` là tổng hợp mỗi bảng. |
| `kpi_definitions.csv` | reports | string | – | Định nghĩa 7 KPI: `kpi`, `ten`, `muc_tieu`, `cong_thuc`, `don_vi`, `ben_lien_quan`. |
| `llm_eval_report.csv` | reports | mixed | – | So sánh mốc tối thiểu, Baseline và LLM với 100 nhãn tay: accuracy, macro-F1, F1 khía cạnh (có/không khen-chê), accuracy ngôn ngữ, khoảng tin cậy 95%, recall lớp mixed. |
| `confusion_base_sentiment.csv`, `confusion_llm_sentiment.csv` | reports | int64 | Số câu | Ma trận nhầm lẫn cảm xúc (hàng `nhan_tay`; cột positive/mixed/negative). |
| `llm_error_analysis.csv` | reports | mixed | – | Từng đánh giá kèm nhãn tay (`human_*`), dự đoán baseline (`base_*`) và LLM (`llm_*`), cờ lỗi (`err_*`, `disagree_base_vs_llm`) và nguyên nhân (`error_type`). |
| `llm_error_summary.csv`, `llm_error_types_summary.csv` | reports | mixed | Số câu; % | Tổng số lỗi theo phương pháp; phân loại nguyên nhân lỗi (`error_type`, `so_luong`, `ty_le_pct`). |
| `inter_annotator_agreement.csv` | reports | mixed | – | Độ đồng thuận 4 người gán nhãn: `so_nguoi_gan`, `so_cau`, `dong_thuan_tuyet_doi_sent`, `dong_thuan_cap_sent_tb`, `f1_khia_canh_cap_tb`. |
| `llm_cost_estimate.csv` | reports | mixed | USD | Ước tính chi phí LLM: token vào/ra trung bình mỗi đánh giá, chi phí mẫu và toàn thành phố (đơn giá trong `configs/vienna.yaml`). |


## 11. Quy ước chung

- **Giá trị thiếu:** để trống (`NaN`), không điền giả; dòng lỗi được gắn cờ chứ không xóa âm thầm.
- **Nhãn cảm xúc:** `positive`, `mixed`, `negative`.
- **Mã hóa tệp CSV đầu ra:** UTF-8 có BOM (đọc đúng tiếng Việt trong Excel).
- **Tái lập:** mọi tham số (ngưỡng QA, mô hình, seed = 42) nằm trong `configs/vienna.yaml`.

