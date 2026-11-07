# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604024
- **Họ và Tên:** Nguyễn Hữu An
- **SEED:** 2317

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

| Phương án | F1 test (775 tin) | F1 SUB (40 tin) | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.869 | 0.788 | 0.016 | 0 |
| BoW + MLP | 0.935 | 1.000 | 0.070 | 0 |
| MiniLM + kNN | 0.919 | 1.000 | 35.051 | 0 |
| MiniLM + LogReg | 0.926 | 0.947 | 8.698 | 0 |
| MiniLM + MLP head | 0.925 | 0.947 | 7.308 | 0 |
| LLM nhỏ zero-shot | Chưa đo | 0.895 | 1496.523 | 0.007987 |
| LLM lớn zero-shot | Chưa đo | 0.947 | 1244.101 | 0.079867 |
| LLM nhỏ few-shot | Chưa đo | 0.947 | 1440.404 | 0.021053 |

## 3. Dự đoán và thực tế

| Dự đoán | Kết quả | Bài học |
| --- | --- | --- |
| `ml_lr`: lr = 50 dao động mạnh và hội tụ chậm hơn lr = 0.5. | Loss tăng mạnh lúc đầu nhưng sau đó giảm xuống mức thấp nhanh hơn lr = 0.5. | Learning rate lớn có thể gây vượt bước, nhưng không đồng nghĩa luôn hội tụ chậm; còn phụ thuộc gradient và dữ liệu. |
| `dl_relu`: bỏ ReLU sẽ cho F1 gần LogReg tự code. | F1 không ReLU = 0.941, cao hơn LogReg = 0.869. | Hai lớp affine vẫn tương đương một phép affine, nhưng cách tối ưu khác nhau: Adam theo mini-batch so với gradient descent toàn bộ dữ liệu, cùng khác biệt khởi tạo và lịch học, có thể cho kết quả khác. |
| `emb_tricky`: Encoder + LogReg đúng nhiều hơn BoW. | Cả hai đúng 5/6 và cùng bỏ sót tin “Last chance! Your account rewards expire tonight, confirm details to keep them”. | Có biểu diễn ngữ nghĩa chưa đảm bảo nhận ra ý đồ lừa đảo; cả hai vẫn nhầm tin dụ xác nhận thông tin với thông báo tài khoản. |

## 4. Phân tích 50 case khó

| Phương án | F1 SUB → F1 case khó | Accuracy case khó |
| --- | ---: | ---: |
| BoW + LogReg | 0.788 → 0.368 | 52% |
| BoW + MLP | 1.000 → 0.400 | 52% |
| MiniLM + kNN | 1.000 → 0.679 | 64% |
| MiniLM + LogReg | 0.947 → 0.571 | 58% |
| MiniLM + MLP head | 0.947 → 0.571 | 58% |
| LLM nhỏ zero-shot | 0.895 → 1.000 | 100% |
| LLM nhỏ + prompt phòng thủ | Chưa đo SUB riêng → 1.000 | 100% |

BoW + MLP tụt nhiều nhất so với SUB, giảm 0.600 F1; nhóm né từ khoá sai cả 6/6 tin. BoW + LogReg khó nhất ở nhóm thao túng/lừa đảo, chỉ đúng 1/8 tin. MiniLM + kNN khó nhất ở nhóm nói về spam (25% đúng); hai đầu LogReg/MLP khó nhất ở spam giọng tự nhiên (10%/20% đúng).

Hai ví dụ BoW + MLP phân loại sai thành ham trong nhóm né từ khoá:

- **Case #29:** tin báo trúng quà dùng `C0ngr4tulat10ns`, `Y0u`, `w0n`, `Cl4im`. Các từ biến dạng không khớp từ điển, làm mất tín hiệu spam.
- **Case #34:** tin mời nhận điện thoại dùng `W.I.N`, `iPh0ne`, `rep|y`. Dấu chấm và ký tự thay thế làm tokenizer tách sai các từ quan trọng; MLP vẫn bị giới hạn bởi đầu vào BoW.

LLM nhỏ đúng cả 50 case với cả hai prompt, kể cả nhóm injection. Prompt thường của tôi đã yêu cầu bỏ qua chỉ dẫn trong SMS, nên kết quả này chưa chứng minh prompt phòng thủ bổ sung có tác dụng hơn. Bộ test viết tay giúp tìm điểm yếu, nhưng không đủ đại diện để kết luận LLM luôn đúng ngoài thực tế; test thường cũng có thể lạc quan vì còn các mẫu spam gần trùng với train.

## 5. Thử thách cá nhân

**Chưa làm.** Với SEED 2317, `2317 % 6 = 1`, tương ứng **đề số 2**: giảm vocab xuống 100 từ, tăng lên 5000 từ, train lại và so sánh F1 test.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày (khoảng 116 tin/giây trung bình), tôi chọn "BoW + MLP làm phương án ban đầu để thử nghiệm triển khai": F1 test = 0.935, precision = 0.966, recall = 0.905, train khoảng 1.518 giây, throughput cả lô khoảng 140226 tin/giây và p95 = 0.070 ms/tin. Cả năm phương án BoW/embedding đạt p95 < 50 ms trong phép đo; ba phương án LLM đều vượt ngưỡng.

Trên biểu đồ SUB, BoW + LogReg và BoW + MLP nằm trên đường Pareto. Đánh đổi của lựa chọn MLP là độ bền kém: F1 chỉ còn 0.400 trên case khó. Tôi chưa kiểm thử tải thực tế; throughput cả lô và p95 đo trong lab chưa bảo đảm SLA khi có tải cao điểm, hàng đợi và độ trễ hệ thống.

Tôi đề xuất thử hệ thống lai: BoW + MLP xử lý tin chắc chắn, chuyển tin có xác suất trong khoảng [0.3, 0.7] sang LLM để xét thêm. Nhánh LLM cần xử lý bất đồng bộ hoặc có cơ chế chờ riêng vì không đạt SLA 50 ms; ngưỡng router và chất lượng toàn hệ thống chưa được thử nghiệm. Cũng cần kiểm tra độc lập các đường link đáng ngờ vì mô hình có thể tự tin nhưng vẫn sai.

Nếu gửi toàn bộ 10 triệu tin/ngày cho LLM trong 30 ngày, lấy USD/1000 tin chưa làm tròn trong JSON nhân 300000, chi phí ước tính là 2396 USD/tháng (nhỏ zero-shot), 23960 USD/tháng (lớn zero-shot) hoặc 6316 USD/tháng (nhỏ few-shot), chưa gồm hạ tầng và retry.

## 7. AI log và điều đã học

Qua bài lab, tôi hiểu rõ hơn rằng cùng khả năng biểu diễn chưa chắc cho cùng kết quả nếu cách tối ưu khác nhau, như thí nghiệm bỏ ReLU. Điều làm tôi thay đổi cách nhìn nhiều nhất là BoW + MLP đạt F1 = 1.000 trên SUB nhưng chỉ còn 0.400 trên case khó: chọn mô hình cần xem cả độ bền, độ trễ và chi phí, thay vì chỉ nhìn một điểm F1 cao.
