# hadong-routing

Tìm đường tối ưu trong mạng giao thông **động** quận Hà Đông (Hà Nội) với 24 khung giờ: so sánh
* tìm kiếm phụ thuộc thời gian (Dijkstra, A\*, TD-Dijkstra, TD-A\* với heuristic chim bay / cận dưới / ALT, bản Numba, Yen),
* tái lập kế hoạch (replan, replan lười, D\* Lite), dự báo ngắn hạn bằng GCN từ cảm biến, định tuyến theo P90,
* 7 biến thể học tăng cường (cục bộ, +cost-to-go, định hình ctg, cây-GNN, LSTM, cây-GNN + behavior cloning, lai RL+TD-A\*),
* thí nghiệm chọn giờ xuất phát (Q-learning) và mô phỏng nhiều xe cùng được dẫn đường (BPR).

> ⚠️ Vận tốc giao thông là **mô phỏng** (không phải dữ liệu đo). Xem `FEATURES.md` mục "Giả định cần người dùng hiệu chỉnh" và "Hạn chế đã biết".

## Cài đặt

Yêu cầu Python ≥ 3.10 (đã kiểm thử 3.13.7, Windows 11, chỉ CPU).

```bash
cd hadong-routing
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell/cmd); Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Lần chạy đầu cần Internet để tải dữ liệu OpenStreetMap (≈ 1–12 phút tuỳ máy chủ Overpass; có sẵn 3 máy chủ dự phòng).
Bản đồ được cache tại `data/raw/hadong_drive.graphml`; các lần sau chạy offline. Nếu không có mạng và không có cache,
chương trình báo lỗi rõ ràng kèm đường dẫn cần đặt file graphml.

## Chạy nhanh (smoke, ~3 phút)

```bash
python run_all.py --mode smoke
```

Kết quả smoke ghi riêng vào `results/smoke/`, `reports/smoke/`, `data/eval/smoke/` (không ghi đè kết quả full).

## Chạy đầy đủ (full, khoảng 3–3,5 giờ trên laptop Intel 16 luồng, chỉ CPU — phần lớn là huấn luyện 21 lượt RL)

```bash
python run_all.py --mode full               # toàn bộ
python run_all.py --mode full --skip-rl     # bỏ huấn luyện RL
python run_all.py --mode full --steps benchmark plots   # chỉ chạy lại vài bước
python run_all.py --mode full --force-download          # tải lại OSM
python -m src.rl.train --mode full --variants rl_tree rl_hybrid   # chỉ huấn luyện lại vài biến thể RL
python -m src.evaluation.run_benchmark --mode full --only main stress runtime  # chỉ chạy lại một phần đánh giá
```

Các bước (theo thứ tự): `graph → speeds → scenarios → forecast → train_rl → eval_rl → departure → benchmark → multiagent → plots → tests`.
Mỗi bước cũng chạy riêng được, ví dụ `python -m src.speed_model --mode full`.

Mọi tham số nằm trong `configs/config.yaml` (khối `modes.smoke` ghi đè cho chế độ smoke); các điểm nóng ùn tắc
trong `configs/hotspots.yaml`. Có thể trỏ tới file cấu hình khác bằng biến môi trường `HADONG_CONFIG`.

## Giao diện Streamlit

```bash
python -m streamlit run src/app/streamlit_app.py   # mở http://localhost:8501, Ctrl+C để tắt
# mở sẵn một cặp O–D: http://localhost:8501/?o=100&d=2500&h=8
```

Mặc định ứng dụng chạy **theo giờ thực**: lấy giờ hiện tại (GMT+7), tự cập nhật mỗi phút, nên mức ùn tắc và tuyến
đường đúng với thời điểm đang chạy; tắt công tắc "Theo giờ thực" (hoặc thêm `?h=8` vào URL) để chọn giờ thủ công.
Nhấp bản đồ để đặt điểm đi/đến hoặc thêm sự cố; chọn giờ, phương pháp (Dijkstra, TD-A\*, replan, D\* Lite, P90, các
biến thể RL), số tuyến thay thế; bật/tắt lớp vận tốc (gộp theo màu, có chú thích) và dải tin cậy P10–P90; tab
"Toàn mạng" vẽ cả 8.944 cạnh bằng WebGL (pydeck). Nếu chưa huấn luyện RL, các lựa chọn RL tự ẩn.

## Docker

```bash
docker build -t hadong-routing .
docker run --rm -p 8501:8501 hadong-routing            # giao diện tại http://localhost:8501
docker compose run --rm pipeline                       # chạy pipeline smoke trong container
```
Ảnh dùng PyTorch bản CPU (~2,9 GB). Đã kiểm tra: build thành công, giao diện trả health `ok`, 55/55 test qua trong container.

## Kiểm thử

```bash
python -m pytest -q tests
```

Test cần dữ liệu đã sinh ở giai đoạn 1–2 (nếu chưa có sẽ tự `skip`).

## TensorBoard

```bash
tensorboard --logdir results/rl/tb
```

## Báo cáo LaTeX

```bash
cd reports/paper
latexmk main                                     # latexmkrc chọn sẵn XeLaTeX
# hoặc thủ công: xelatex main, bibtex main, rồi xelatex main hai lần
```

Báo cáo dùng `fontspec` nên **phải** biên dịch bằng XeLaTeX (pdflatex báo lỗi *fontspec requires XeTeX or LuaTeX*).
File `latexmkrc` ép latexmk dùng XeLaTeX kể cả khi gọi `latexmk -pdf` (mặc định của LaTeX Workshop trên VS Code).

Báo cáo theo mẫu "Báo cáo Thuật toán ứng dụng" (Đại học Phenikaa): `main.tex` (bìa, trang lót, mục lục, danh mục
hình/bảng) + các chương `sec1.tex`–`sec6.tex` + `appendix.tex`, tài liệu tham khảo `ref.bib` (kiểu `ieeetr`).
Logo bìa đặt tại `reports/paper/image/phenikaa_image.png`. Font TeX Gyre Pagella; với MiKTeX cần các gói `makecell`,
`tocloft`, `ntheorem`, `fancyhdr`, `fancybox`, `jknappen`, `was`, `tex-gyre` (cài bằng `miktex packages install <gói>`).
Bảng và số liệu trong báo cáo được `\input` từ `reports/tables/*.tex` (sinh bởi `src/evaluation/plots.py`),
hình lấy từ `reports/figures/`.

## Cấu trúc

Xem `FEATURES.md` mục "Cấu trúc thư mục".
