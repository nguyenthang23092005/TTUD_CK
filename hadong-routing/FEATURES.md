# FEATURES — Mô tả chức năng hệ thống `hadong-routing`

Tài liệu mô tả các chức năng đã xây dựng, kiến trúc, mô hình mô phỏng, thuật toán, thiết kế RL, cách chạy,
kết quả chính (sinh tự động từ `results/`) và cấu trúc thư mục.

---

## 1. Tổng quan

Bài toán: **tìm đường nhanh nhất phụ thuộc thời gian** trên mạng đường lái xe quận Hà Đông (OSM), vận tốc thay đổi theo
**24 khung giờ** (nội suy theo phút), môi trường thực có nhiễu tương quan và sự cố.

| Nhóm | Phương pháp | Mã nguồn |
|---|---|---|
| Tĩnh | Dijkstra / A\* theo khoảng cách; A\* theo thời gian tự do; Yen k tuyến | `algorithms/dijkstra.py`, `astar.py`, `yen.py` |
| Phụ thuộc thời gian | TD-Dijkstra, TD-A\* (heuristic chim bay / **cận dưới LB** / **ALT**), **TD-A\* Numba** | `td_astar.py`, `heuristics.py`, `numba_td.py` |
| Trong môi trường thực | Plan-once, **Replan**, **Replan lười**, **Replan D\* Lite**, Oracle | `replan.py`, `dstar_lite.py` |
| Dự báo | **Cảm biến + GCN**: plan-once + GCN, replan + GCN; đối chứng plan-once + cảm biến | `forecast.py` |
| Rủi ro | **Định tuyến P90** (ngân hàng kịch bản + ứng viên Yen) | `risk.py` |
| Học tăng cường | 7 biến thể: cục bộ, +ctg (ablation), **định hình ctg**, **cây-GNN**, **LSTM**, **cây-GNN + BC**, **lai RL + TD-A\*** | `env/`, `rl/` |
| Thí nghiệm phụ | Chọn giờ xuất phát (Q-learning, bước 10 phút, đặc trưng mức ùn theo O–D); **mô phỏng nhiều xe** (BPR) | `rl/departure_time.py`, `evaluation/multiagent.py` |

Kèm: đánh giá thống kê (bootstrap CI, Wilcoxon), **tập stress** riêng, **đo thời gian đơn luồng**, biểu đồ, bảng LaTeX,
macro số liệu, giao diện Streamlit (folium + **pydeck**, **dải P10–P90**), **Docker**, báo cáo LaTeX theo mẫu.

## 2. Kiến trúc

```
OSM ─► graph_builder ─► data/processed (CSR + parquet)
                           │
            speed_model ◄──┘─► V_base (E×24)  +  NoiseModel (nhiễu tương quan) + sự cố (FIFO hoá)
                           │
            scenarios ─────┘─► data/eval/{train,val,test}_set.json  (2 bộ sự cố: chính / stress)
                           │
   ┌───────────────┬───────┴──────────┬──────────────────┬───────────────────────┐
algorithms/      forecast (GCN)     env/ (Gymnasium)    rl/departure_time   evaluation/multiagent
(TD-A*, LB, ALT, (cảm biến 15%)     routing_env (local/ (Q-learning)        (BPR, naive vs
 Numba, replan,                      tree), hybrid_env                       phối hợp)
 lười, D* Lite,                      rl/train (7 biến thể
 P90)                                × 3 seed, BC), evaluate_rl
   └───────────────┴──────────────────┴─────────► evaluation/run_benchmark ─► plots ─► reports/ ─► paper/main.tex
                                                                             app/streamlit_app (demo)
```

## 3. Đồ thị (`src/graph_builder.py`)

* Geocode "Quận Hà Đông"; kiểm tra diện tích ≥ 20 km² (từ 1/7/2025 OSM có thể chỉ trả về *phường* Hà Đông ~9 km²);
  fallback: hợp 5 phường mới → ranh giới cache → bounding box. Buffer 500 m.
* OSMnx `drive`, thử 3 máy chủ Overpass; cache graphml; mất mạng → dùng cache hoặc báo lỗi rõ ràng.
* Làm sạch (self-loop, cạnh song song, SCC lớn nhất), chuẩn hoá thuộc tính (nhóm đường, parse maxspeed, điền thiếu),
  CSR numpy, thống kê, bản đồ.

## 4. Mô hình vận tốc & môi trường thực (`src/speed_model.py`)

```
v(e,h) = max( v_free(e)·(1 − c(e,h)), v_min )
c(e,h) = clip( p(h)·s(g_e)·hotspot(e,h)·dir(e,h) + noise(e,h), 0, c_max )
```
* `p(h)` Gaussian tuần hoàn 24h; `s(g)` theo loại đường; `hotspot` từ `configs/hotspots.yaml`; `dir` hướng vào/ra
  trung tâm; `noise` làm mượt qua cạnh kề + biến thiên theo ngày.
* **Nhiễu môi trường thực có tương quan** (`NoiseModel`): log ε = σ·Z, Z có biên N(0,1) (ε vẫn LogNormal(0,σ)),
  tương quan không gian (làm mượt qua cạnh kề) và thời gian (AR(1) theo giờ, ρ = 0.8), 30% i.i.d.
  Cấu hình `speed.real_noise.model: iid` để quay về nhiễu độc lập như bản đầu.
* FIFO: giới hạn độ dốc −τ' ≤ 0.9; sự cố được FIFO hoá `ã(t) = min_{t'≥t} a(t')`; kiểm tra giải tích + số.

## 5. Kịch bản (`src/scenarios.py`)

* 1000 truy vấn (smoke 50), chia 60/20/20 **theo cặp O–D**; cạnh sự cố test chưa xuất hiện ở train.
* **Hai bộ sự cố trên cùng truy vấn** (cùng số lượng/thời điểm/mức độ, khác vị trí):
  `incidents` (tập chính — vị trí ngẫu nhiên trong hành lang O–D, không thiên lệch) và
  `incidents_stress` (ưu tiên trên tuyến dự kiến của plan-once, xác suất 0.7).

## 6. Thuật toán (`src/algorithms/`)

| Tệp | Nội dung |
|---|---|
| `td_astar.py` | TD-Dijkstra / TD-A\* (nhãn = thời điểm đến) |
| `heuristics.py` | **LB**: Dijkstra ngược trên τ_min = L/max_h v; **ALT**: 8 điểm mốc farthest-point |
| `numba_td.py` | TD-A\* JIT (Numba), cùng số nút mở rộng với bản Python. Contraction Hierarchies **không** cài: không cần ở quy mô ~3.500 nút |
| `replan.py` | plan-once, oracle, replan; **replan lười** (`lazy_threshold`); hook `base_provider` cho dự báo GCN |
| `dstar_lite.py` | **D\* Lite** với chi phí snapshot, cập nhật tăng dần, dựng lại mỗi 15 phút |
| `risk.py` | **Định tuyến P90**: ngân hàng 30 kịch bản/σ, ứng viên Yen + TD-A\*, đánh giá vector hoá |
| `yen.py`, `dijkstra.py`, `astar.py` | tĩnh |

## 7. Dự báo ngắn hạn (`src/forecast.py`)

* Cảm biến cố định 15% cạnh (mọi trunk/primary + ngẫu nhiên) + cạnh trong bán kính quan sát quanh xe.
* GCN 3 lớp × 32 trên đồ thị cạnh–cạnh, dự đoán log(v_real/v_base) tại mốc h và h+1 cho mọi cạnh; các mốc sau tắt dần ρ/giờ.
* Huấn luyện trên "ngày" mô phỏng có seed riêng; đánh giá RMSE trên cạnh chưa quan sát; lưu `results/forecast/gcn.pt`.

## 8. Học tăng cường (`src/env/`, `src/rl/`)

| Biến thể | Quan sát | Thuật toán | Điểm khác | Bước |
|---|---|---|---|---|
| `rl` | cục bộ (K=5 cạnh ra) | MaskablePPO | đúng đặc tả gốc, định hình chim bay | 400k |
| `rl_ctg` | cục bộ + cost-to-go tự do | MaskablePPO | ablation (thông tin toàn cục tĩnh) | 400k |
| `rl_shape` | cục bộ | MaskablePPO | **định hình thưởng theo ctg** (thế năng, chỉ trong thưởng) — B5 | 1M |
| `rl_tree` | **cây lân cận 3 bước** (155 khe) | MaskablePPO + **GNN trên cây** | phát hiện ngõ cụt phía trước, định hình ctg — B1 | 800k |
| `rl_lstm` | cục bộ | **RecurrentPPO (LSTM)** | bộ nhớ, định hình ctg — B2; ánh xạ a mod bậc_ra (không có mask) | 500k |
| `rl_tree_bc` | cây 3 bước | MaskablePPO + GNN | **behavior cloning từ oracle** (DAgger nhẹ) rồi PPO, định hình ctg — B4 | 800k |
| `rl_hybrid` | 12 đặc trưng tóm tắt quan sát | MaskablePPO | **RL chọn β (tin quan sát), TD-A\* tìm đường** — B3 | 60k |

Mọi biến thể: 3 seed, curriculum 3 mức, đánh giá định kỳ trên val, checkpoint tốt nhất theo val, TensorBoard.
`python -m src.rl.train --variants ...` để huấn luyện lại một phần.

## 9. Thí nghiệm phụ

* **Chọn giờ xuất phát** (`rl/departure_time.py`): chờ 10 phút × tối đa 6; w ∈ {1, 0.5, 0.25}; trạng thái thêm nhóm
  mức ùn dự báo của chính cặp O–D; so với "đi ngay" và vét cạn.
* **Nhiều xe** (`evaluation/multiagent.py`): 50–800 tác tử × 20 xe, 7:30–8:00; tải dẫn đường làm chậm cạnh theo BPR;
  so sánh dẫn đường giống nhau (naive) vs phối hợp tuần tự; điểm bất động tải ↔ lịch trình.

## 10. Đánh giá (`src/evaluation/`)

* `run_benchmark.py`: tập chính (mọi phương pháp), **tập stress**, nhanh nhất vs ngắn nhất theo giờ, quét σ, **độ tin
  cậy tuyến P90** (100 kịch bản độc lập/truy vấn), **thời gian đơn luồng** (một tiến trình, tuần tự, sau làm nóng).
  `--only main stress saving noise risk runtime` để chạy một phần.
* `metrics.py`: regret, tổng hợp, bootstrap CI ghép cặp, Wilcoxon, xuất LaTeX/macro.
* `plots.py`: mọi hình/bảng/macro + tự cập nhật mục kết quả dưới đây.

## 11. Giao diện (`src/app/streamlit_app.py`)

* Folium: nhấp đặt điểm đi/đến/sự cố; URL `?o=&d=&h=&mode=&all=1`; lớp vận tốc **gộp theo 5 lớp màu + chú thích**.
* **Chế độ giờ thực** (mặc định bật): lấy giờ:phút hiện tại theo giờ Việt Nam, tự cập nhật mỗi phút; lớp ùn tắc nội suy
  vận tốc theo phút như mô hình TD. Tắt công tắc (hoặc truyền `?h=`) để chọn giờ thủ công.
* Tab **pydeck** vẽ toàn bộ 8.944 cạnh bằng WebGL (độ rộng theo mét + `width_min_pixels`; không dùng `width_units`
  vì bản deck.gl đi kèm Streamlit 1.62 không nhận, làm nét vẽ phình thành mảng lớn).
* Phương pháp: Dijkstra, TD-A\*, Replan, D\* Lite, P90, mọi biến thể RL có checkpoint; Yen k tuyến.
* Bảng có **P10/P50/P90** của từng tuyến (ngân hàng kịch bản); biểu đồ 24 giờ có **dải P10–P90**.
* `tools/screenshot_app.py`: chụp màn hình tự động (Playwright + Edge/Chrome).

## 12. Docker

`Dockerfile` (python:3.13-slim, PyTorch CPU), `docker-compose.yml` (dịch vụ `app`, `pipeline`), `.dockerignore`.
Đã kiểm tra: build thành công (~2,9 GB), giao diện trả health `ok`, 55/55 test qua trong container Linux.

## 13. Hướng dẫn chạy

```bash
pip install -r requirements.txt
python run_all.py --mode smoke        # vài phút
python run_all.py --mode full         # ~3–3,5 giờ (21 lượt RL)
python -m streamlit run src/app/streamlit_app.py   # http://localhost:8501
python -m pytest -q tests
cd reports/paper && latexmk main   # latexmkrc chọn sẵn XeLaTeX (hoặc: xelatex, bibtex, xelatex ×2)
```

## 14. Kết quả chính (chế độ full)

<!-- RESULTS:START -->
*Tự động sinh bởi `src/evaluation/plots.py` từ `results/` (chế độ full). Không sửa tay.*

**Đồ thị:** 3482 nút, 8944 cạnh, 944.5 km. **Tập test:** 200 truy vấn.

### Tìm kiếm / dự báo (tập chính)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Dijkstra (ngắn nhất) | 100.00 | 19.01 | 33.12 | 11.03 | 26.73 |
| A* (thời gian tự do) | 100.00 | 18.33 | 31.42 | 5.21 | 16.49 |
| Plan-once TD-A* | 100.00 | 17.51 | 31.05 | 1.29 | 4.64 |
| Plan-once + cảm biến | 100.00 | 17.43 | 30.89 | 0.92 | 2.85 |
| Plan-once + GCN | 100.00 | 17.39 | 31.04 | 0.59 | 1.22 |
| Định tuyến P90 | 100.00 | 17.53 | 31.08 | 1.47 | 4.75 |
| Replan TD-A* | 100.00 | 17.35 | 30.89 | 0.47 | 1.18 |
| Replan lười | 100.00 | 17.36 | 30.89 | 0.48 | 1.18 |
| Replan D* Lite | 100.00 | 17.37 | 31.04 | 0.44 | 1.44 |
| Replan + GCN | 100.00 | 17.31 | 31.04 | 0.19 | 0.01 |
| Oracle TD-A* | 100.00 | 17.26 | 30.89 | 0.00 | 0.00 |


### RL (tập chính)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Tham lam cục bộ | 37.50 | 21.02 | 38.70 | 62.88 | 135.28 |
| RL cục bộ | 35.00 | 22.27 | 42.88 | 63.24 | 90.16 |
| RL + ctg (quan sát) | 100.00 | 18.32 | 31.52 | 5.19 | 17.79 |
| RL + định hình ctg | 57.00 | 24.18 | 43.68 | 76.11 | 183.88 |
| RL cây-GNN | 67.00 | 20.55 | 35.43 | 39.38 | 87.93 |
| RL LSTM | 2.50 | 22.83 | 29.17 | 219.06 | 307.75 |
| RL cây-GNN + BC | 77.00 | 21.59 | 40.35 | 37.36 | 73.33 |
| RL lai (β + TD-A*) | 100.00 | 17.37 | 30.89 | 0.56 | 1.67 |
| Replan TD-A* | 100.00 | 17.35 | 30.89 | 0.47 | 1.18 |
| Oracle TD-A* | 100.00 | 17.26 | 30.89 | 0.00 | 0.00 |


### Tập stress (sự cố trên tuyến dự kiến)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Dijkstra (ngắn nhất) | 100.00 | 19.38 | 33.60 | 11.38 | 26.36 |
| A* (thời gian tự do) | 100.00 | 18.80 | 33.10 | 6.38 | 19.05 |
| Plan-once TD-A* | 100.00 | 18.10 | 31.84 | 3.37 | 11.44 |
| Plan-once + GCN | 100.00 | 17.88 | 31.84 | 2.10 | 5.99 |
| Định tuyến P90 | 100.00 | 18.11 | 31.85 | 3.51 | 11.73 |
| Replan TD-A* | 100.00 | 17.71 | 31.84 | 0.67 | 1.93 |
| Replan lười | 100.00 | 17.71 | 31.84 | 0.67 | 1.93 |
| Replan D* Lite | 100.00 | 17.72 | 31.84 | 0.66 | 2.20 |
| Replan + GCN | 100.00 | 17.64 | 31.84 | 0.31 | 0.28 |
| RL lai (β + TD-A*) | 100.00 | 17.73 | 31.84 | 0.80 | 2.33 |
| RL cây-GNN + BC | 77.50 | 21.91 | 38.82 | 40.77 | 75.24 |
| Oracle TD-A* | 100.00 | 17.57 | 31.71 | 0.00 | 0.00 |


### Thời gian tính đơn luồng

| Phương pháp | Nút/bước TB | ms TB | Tiền xử lý ms |
|---|---|---|---|
| Dijkstra (ngắn nhất) | 1758.96 | 1.91 | — |
| A* (ngắn nhất) | 523.64 | 0.65 | — |
| A* (thời gian tự do) | 741.56 | 0.80 | — |
| TD-Dijkstra (V_base) | 1733.58 | 2.94 | — |
| TD-A* (chim bay) = plan-once | 1101.06 | 1.77 | — |
| TD-A* + heuristic LB | 497.84 | 0.83 | 3.24 |
| TD-A* + ALT | 883.94 | 1.36 | 1.55 |
| TD-A* (Numba) | 1101.06 | 0.32 | 0.00 |
| Oracle TD-A* | 1333.80 | 2.73 | — |
| Plan-once + cảm biến | 1097.20 | 2.19 | — |
| Plan-once + GCN | 1091.96 | 21.77 | — |
| Định tuyến P90 | 1101.06 | 214.83 | — |
| Replan TD-A* | 28424.50 | 53.16 | — |
| Replan lười | 11995.92 | 27.10 | — |
| Replan D* Lite | 2244.06 | 35.31 | — |
| Replan + GCN | 27617.88 | 124.38 | — |
| Tham lam cục bộ | 124.52 | 31.37 | — |
| RL cục bộ | 121.38 | 75.85 | — |
| RL + ctg (quan sát) | 46.76 | 43.51 | — |
| RL + định hình ctg | 109.48 | 72.52 | — |
| RL cây-GNN | 86.16 | 102.03 | — |
| RL LSTM | 162.72 | 127.95 | — |
| RL cây-GNN + BC | 86.36 | 108.22 | — |
| RL lai (β + TD-A*) | 47.30 | 104.62 | — |


### So sánh ghép cặp

| Cặp (A − B) | n | Hiệu TB (phút) | CI thấp | CI cao | p |
|---|---|---|---|---|---|
| Plan-once TD-A* − Dijkstra (ngắn nhất) | 200 | -1.499 | -1.738 | -1.270 | < 1e-4 |
| Replan TD-A* − Plan-once TD-A* | 200 | -0.159 | -0.250 | -0.084 | < 1e-4 |
| Oracle TD-A* − Replan TD-A* | 200 | -0.090 | -0.142 | -0.050 | < 1e-4 |
| Replan lười − Replan TD-A* | 200 | 0.002 | 0.000 | 0.004 | 0.0687 |
| Replan D* Lite − Replan TD-A* | 200 | 0.015 | -0.036 | 0.077 | 0.3037 |
| Replan + GCN − Replan TD-A* | 200 | -0.048 | -0.105 | 0.005 | 0.0186 |
| Plan-once + cảm biến − Plan-once TD-A* | 200 | -0.081 | -0.155 | -0.013 | 0.0297 |
| Plan-once + GCN − Plan-once TD-A* | 200 | -0.124 | -0.199 | -0.055 | < 1e-4 |
| Plan-once + GCN − Plan-once + cảm biến | 200 | -0.043 | -0.093 | 0.008 | 0.0002 |
| Định tuyến P90 − Plan-once TD-A* | 200 | 0.015 | -0.010 | 0.038 | 0.0027 |
| RL lai (β + TD-A*) − Replan TD-A* | 200 | 0.013 | 0.004 | 0.025 | 0.0009 |
| RL cây-GNN + BC − Plan-once TD-A* | 154 | 5.527 | 4.473 | 6.798 | < 1e-4 |
| RL + ctg (quan sát) − A* (thời gian tự do) | 200 | -0.001 | -0.079 | 0.063 | 0.0002 |


### Dự báo GCN

| Mô hình | RMSE chưa quan sát | RMSE mọi cạnh |
|---|---|---|
| Dự báo nền V_base (log-tỉ số = 0) | 0.196 | 0.196 |
| GCN chưa huấn luyện | 0.238 | 0.238 |
| GCN đã huấn luyện | 0.160 | 0.153 |


### Định tuyến P90

| Tuyến | TB | P90 | Std |
|---|---|---|---|
| Plan-once TD-A* | 16.79 | 17.69 | 0.71 |
| Định tuyến P90 | 16.80 | 17.71 | 0.71 |


### Nhiều xe

| Số xe | Không tương tác | Naive | Phối hợp |
|---|---|---|---|
| 1000 | 28.45 | 29.26 | 29.18 |
| 2000 | 28.52 | 31.35 | 30.69 |
| 4000 | 27.78 | 31.49 | 30.38 |
| 8000 | 27.45 | 34.83 | 32.56 |
| 16000 | 27.04 | 43.46 | 36.80 |


### Chọn giờ xuất phát

| w | Chính sách | Chi phí TB | Tỉ lệ chờ | Regret |
|---|---|---|---|---|
| 0.250 | Đi ngay | 17.264 | 0.000 | 0.183 |
| 0.250 | Giờ tối ưu (vét cạn) | 17.080 | 0.050 | 0.000 |
| 0.250 | Q-learning | 17.729 | 0.160 | 0.649 |
| 0.500 | Đi ngay | 17.264 | 0.000 | 0.007 |
| 0.500 | Giờ tối ưu (vét cạn) | 17.256 | 0.010 | 0.000 |
| 0.500 | Q-learning | 17.550 | 0.060 | 0.294 |
| 1.000 | Đi ngay | 17.264 | 0.000 | 0.000 |
| 1.000 | Giờ tối ưu (vét cạn) | 17.264 | 0.000 | 0.000 |
| 1.000 | Q-learning | 17.264 | 0.000 | 0.000 |


Hình: `reports/figures/*.png|pdf`; bảng LaTeX: `reports/tables/*.tex`; báo cáo: `reports/paper/main.pdf`.
<!-- RESULTS:END -->

### Giả định cần người dùng hiệu chỉnh

* `configs/hotspots.yaml`: hệ số nhân và khung giờ của 10 trục/điểm nóng là **giả định** (toạ độ từ Nominatim).
* `speed.profile`, `speed.road_sensitivity`, `speed.direction.beta`, `graph.free_flow_factor`, `graph.default_maxspeed_kmh`
  (chỉ ~6.5% cạnh có tag maxspeed thật).
* Mô hình nhiễu `speed.real_noise` (ρ, mức làm mượt, tỉ lệ i.i.d.), `scenarios.sigma_levels`, mô hình sự cố.
* `forecast.sensor_fraction` (giả định 15% cạnh có cảm biến/xe thăm dò), `algorithms.replan.observation_radius_m` (400 m).
* `multiagent.capacity_vph`, `bpr_alpha/beta`, `vehicles_per_agent`.

### Hạn chế đã biết

* Dữ liệu vận tốc **tự mô phỏng**; kết luận định lượng chỉ đúng trong mô phỏng.
* Không mô hình hoá đèn tín hiệu, hạn chế rẽ, xe máy; mô phỏng nhiều xe dùng BPR đơn giản.
* D\* Lite dùng chi phí snapshot (xấp xỉ, không tối ưu theo thời gian).
* RecurrentPPO không hỗ trợ mặt nạ hành động (dùng ánh xạ a mod bậc_ra).

## 15. Cấu trúc thư mục

```
hadong-routing/
├── configs/config.yaml, hotspots.yaml
├── data/raw | processed | speeds | eval (+ smoke/)
├── src/
│   ├── config.py, utils.py, graph_data.py, graph_builder.py, speed_model.py, scenarios.py, forecast.py
│   ├── algorithms/{common,dijkstra,astar,td_astar,heuristics,numba_td,replan,dstar_lite,risk,yen}.py
│   ├── env/{routing_env,hybrid_env}.py
│   ├── rl/{train,evaluate_rl,policies,departure_time}.py
│   ├── evaluation/{metrics,run_benchmark,multiagent,plots,parallel}.py
│   └── app/streamlit_app.py
├── tests/ (graph/speed, algorithms, algorithms_v2, scenarios, env, improvements, metrics)
├── tools/screenshot_app.py
├── reports/{figures,tables,paper/}  (+ reports/smoke/)
├── results/  (json/csv; rl/; forecast/; departure/; models/; logs/; smoke/)
├── run_all.py, requirements.txt, Dockerfile, docker-compose.yml, README.md, FEATURES.md
```
