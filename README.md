# TTUD_CK

Đồ án cuối kỳ môn **Thuật toán ứng dụng**: *Tìm đường tối ưu trong mạng giao thông động quận Hà Đông với 24 khung giờ —
so sánh thuật toán cổ điển và học tăng cường.*

| Đường dẫn | Nội dung |
|---|---|
| [`hadong-routing/`](hadong-routing/) | Mã nguồn, dữ liệu, kết quả và báo cáo |
| [`hadong-routing/README.md`](hadong-routing/README.md) | Cài đặt và cách chạy (pipeline, giao diện Streamlit, Docker, kiểm thử) |
| [`hadong-routing/FEATURES.md`](hadong-routing/FEATURES.md) | Mô tả chức năng, thuật toán và kết quả chính |
| [`hadong-routing/reports/paper/main.pdf`](hadong-routing/reports/paper/main.pdf) | Báo cáo LaTeX (nguồn: `main.tex`, `sec1.tex`–`sec6.tex`) |

Chạy nhanh:

```bash
cd hadong-routing
pip install -r requirements.txt
python run_all.py --mode smoke                     # ~3 phút
python -m streamlit run src/app/streamlit_app.py   # giao diện tại http://localhost:8501
```
