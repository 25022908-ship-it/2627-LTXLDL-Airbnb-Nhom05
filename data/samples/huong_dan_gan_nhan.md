# Hướng dẫn gán nhãn cảm xúc và khía cạnh (Dùng chung cho Nhãn tay, Baseline và LLM)

1. **Cảm xúc tổng thể (`sentiment`)**:
   - `positive`: Bình luận khen hoàn toàn hoặc khen chủ đạo, không có phàn nàn đáng kể.
   - `negative`: Bình luận chê chủ đạo hoặc bày tỏ sự không hài lòng rõ rệt.
   - `mixed`: Bình luận vừa khen vừa chê rõ rệt (ví dụ: khen vị trí đẹp nhưng chê phòng ồn).

2. **Danh mục 7 khía cạnh chuẩn (`aspects_pos`, `aspects_neg`)**:
   - `vị trí`: khu vực, giao thông, khoảng cách đi bộ tới trung tâm/tàu điện.
   - `sạch sẽ`: vệ sinh phòng, mùi, bụi bẩn, nấm mốc.
   - `chủ nhà`: giao tiếp, đón tiếp, hỗ trợ, thủ tục check-in.
   - `giá trị`: mức giá tương xứng với chất lượng nhận được.
   - `tiện nghi`: nội thất, giường, bếp, wifi, điều hòa, nước nóng.
   - `độ ồn`: tiếng ồn xung quanh hoặc sự yên tĩnh.
   - `an toàn`: an ninh khu phố, khóa cửa, cảm giác an toàn.

3. **Quy ước xử lý trường hợp biên**:
   - Chỉ gán khía cạnh khi bình luận nhắc rõ ràng; không suy đoán từ điểm số.
   - Nếu một khía cạnh vừa được khen vừa bị chê trong cùng bình luận (ví dụ: *"giường êm nhưng phòng tắm hơi nhỏ"* -> khía cạnh `tiện nghi`), ghi tên khía cạnh đó vào **cả hai cột** `aspects_pos` và `aspects_neg`, đồng thời đặt `sentiment` là `mixed`.