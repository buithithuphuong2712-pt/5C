# 🔮 Ứng Dụng Web Dự Báo Phân Loại Chỉ Số Đối Tượng (Streamlit Web App)

Dự án này chuyển đổi tự động quy trình xây dựng, huấn luyện và kiểm định mô hình học máy phân loại từ định dạng Jupyter Notebook sang nền tảng giao diện web tương tác phân vùng hiện đại bằng **Streamlit**.

## 📌 Khái quát về ứng dụng & thuật toán sử dụng
- **Bài toán:** Phân loại nhị phân (Binary Classification) dự đoán giá trị của nhãn mục tiêu `PD`.
- **Tập biến đầu vào (X):** Bao gồm 24 đặc trưng khảo sát đại diện: `TC1` -> `TC5`, `NL1` -> `NL4`, `DK1` -> `DK5`, `V1` -> `V6`, `TS1` -> `TS4`.
- **Mô hình AI:** Sử dụng thuật toán Rừng ngẫu nhiên (**RandomForestClassifier**) làm giải pháp học máy chính tối ưu cho các dạng dữ liệu bảng số liệu khảo sát, có khả năng xử lý tốt các quan hệ phi tuyến và tránh hiện tượng quá khớp (overfitting).

## 🗂️ Cấu trúc các Phân vùng chức năng giao diện (Tabs)
1. **📊 Tổng quan dữ liệu:** Xem nhanh cấu trúc kích thước file, các dòng bản ghi thô đầu tiên và thông số thống kê toán học mô tả quan trọng của các tập biến.
2. **📈 Trực quan hóa dữ liệu:** Sử dụng thư viện đồ họa động **Plotly** hiển thị biểu đồ phân phối tần suất của biến mục tiêu `PD` và hỗ trợ đa chọn tùy biến hiển thị của các trường dữ liệu đầu vào.
3. **🎯 Kết quả & Kiểm định:** Đánh giá độ chính xác tổng quan trên tập kiểm định ẩn (Test split), thể hiện qua các chỉ số `Accuracy`, `Precision`, `Recall`, `F1-score`, trực quan hóa biểu đồ Ma trận nhầm lẫn màu sắc (Confusion Matrix).
4. **🔮 Sử dụng mô hình:** Tích hợp bộ suy luận hai chế độ tiện lợi:
   - *Nhập dữ liệu trực tiếp:* Form điền thông số thủ công có tự động điền sẵn giá trị trung vị lịch sử.
   - *Dự báo hàng loạt:* Tải lên tệp danh sách mới chứa đầy đủ các cột thuộc tính đặc trưng và xuất kết quả file đính kèm nút tải xuống tiện lợi.

## 🛠️ Hướng dẫn thiết lập & Chạy ứng dụng

### 1. Chuẩn bị môi trường
Yêu cầu hệ thống đã cài đặt sẵn **Python 3.9+**. Mở Terminal (hoặc Command Prompt) tại thư mục chứa mã nguồn dự án và cài đặt toàn bộ các thư viện bổ trợ cần thiết:
```bash
pip install -r requirements.txt
