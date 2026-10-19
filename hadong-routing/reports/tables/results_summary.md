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
