*Tự động sinh bởi `src/evaluation/plots.py` từ `results/` (chế độ smoke). Không sửa tay.*

**Đồ thị:** 3482 nút, 8944 cạnh, 944.5 km. **Tập test:** 10 truy vấn.

### Tìm kiếm / dự báo (tập chính)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Dijkstra (ngắn nhất) | 100.00 | 13.49 | 18.83 | 10.38 | 25.62 |
| A* (thời gian tự do) | 100.00 | 13.39 | 19.06 | 6.33 | 18.18 |
| Plan-once TD-A* | 100.00 | 13.02 | 19.08 | 1.99 | 6.18 |
| Plan-once + cảm biến | 100.00 | 12.79 | 18.83 | 1.54 | 4.56 |
| Plan-once + GCN | 100.00 | 13.02 | 19.08 | 1.99 | 6.18 |
| Định tuyến P90 | 100.00 | 13.14 | 19.08 | 3.44 | 7.44 |
| Replan TD-A* | 100.00 | 12.90 | 19.07 | 0.81 | 2.54 |
| Replan lười | 100.00 | 12.90 | 19.07 | 0.81 | 2.54 |
| Replan D* Lite | 100.00 | 12.90 | 19.07 | 0.85 | 2.93 |
| Replan + GCN | 100.00 | 12.90 | 19.07 | 0.85 | 2.93 |
| Oracle TD-A* | 100.00 | 12.63 | 18.83 | 0.00 | 0.00 |


### RL (tập chính)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Tham lam cục bộ | 30.00 | 13.94 | 19.50 | 100.39 | 210.99 |
| RL cục bộ | 0.00 | nan | nan | nan | nan |
| RL + ctg (quan sát) | 10.00 | 9.63 | 9.63 | 35.43 | 35.43 |
| RL + định hình ctg | 0.00 | nan | nan | nan | nan |
| RL cây-GNN | 0.00 | nan | nan | nan | nan |
| RL LSTM | 0.00 | nan | nan | nan | nan |
| RL cây-GNN + BC | 0.00 | nan | nan | nan | nan |
| RL lai (β + TD-A*) | 100.00 | 12.71 | 18.88 | 0.34 | 1.37 |
| Replan TD-A* | 100.00 | 12.90 | 19.07 | 0.81 | 2.54 |
| Oracle TD-A* | 100.00 | 12.63 | 18.83 | 0.00 | 0.00 |


### Tập stress (sự cố trên tuyến dự kiến)

| Phương pháp | Tới đích % | TB (phút) | P90 | Regret TB % | Regret P90 % |
|---|---|---|---|---|---|
| Dijkstra (ngắn nhất) | 100.00 | 14.09 | 19.95 | 12.01 | 25.53 |
| A* (thời gian tự do) | 100.00 | 13.99 | 20.18 | 8.40 | 18.18 |
| Plan-once TD-A* | 100.00 | 13.62 | 20.20 | 4.07 | 7.82 |
| Plan-once + GCN | 100.00 | 13.62 | 20.20 | 4.07 | 7.82 |
| Định tuyến P90 | 100.00 | 13.74 | 20.19 | 5.51 | 15.19 |
| Replan TD-A* | 100.00 | 13.25 | 20.19 | 0.80 | 2.50 |
| Replan lười | 100.00 | 13.25 | 20.19 | 0.80 | 2.50 |
| Replan D* Lite | 100.00 | 13.26 | 20.19 | 0.85 | 2.89 |
| Replan + GCN | 100.00 | 13.26 | 20.19 | 0.85 | 2.89 |
| RL lai (β + TD-A*) | 90.00 | 12.56 | 18.10 | 0.38 | 1.45 |
| RL cây-GNN + BC | 0.00 | nan | nan | nan | nan |
| Oracle TD-A* | 100.00 | 12.99 | 19.95 | 0.00 | 0.00 |


### Thời gian tính đơn luồng

| Phương pháp | Nút/bước TB | ms TB | Tiền xử lý ms |
|---|---|---|---|
| Dijkstra (ngắn nhất) | 1469.20 | 1.65 | nan |
| A* (ngắn nhất) | 379.70 | 0.49 | nan |
| A* (thời gian tự do) | 549.90 | 0.60 | nan |
| TD-Dijkstra (V_base) | 1401.10 | 2.43 | nan |
| TD-A* (chim bay) = plan-once | 881.40 | 1.43 | nan |
| TD-A* + heuristic LB | 499.10 | 0.82 | 3.70 |
| TD-A* + ALT | 690.90 | 1.12 | 1.32 |
| TD-A* (Numba) | 881.40 | 0.30 | 0.00 |
| Oracle TD-A* | 1126.00 | 2.29 | nan |
| Plan-once + cảm biến | 917.00 | 1.96 | nan |
| Plan-once + GCN | 882.50 | 37.01 | nan |
| Định tuyến P90 | 881.40 | 105.62 | nan |
| Replan TD-A* | 14184.30 | 29.08 | nan |
| Replan lười | 7044.50 | 16.89 | nan |
| Replan D* Lite | 1217.20 | 24.24 | nan |
| Replan + GCN | 14214.80 | 85.60 | nan |
| Tham lam cục bộ | 102.50 | 29.46 | nan |
| RL cục bộ | 124.70 | 78.79 | nan |
| RL + ctg (quan sát) | 118.90 | 81.49 | nan |
| RL + định hình ctg | 124.70 | 87.05 | nan |
| RL cây-GNN | 124.70 | 151.13 | nan |
| RL LSTM | 124.70 | 105.96 | nan |
| RL cây-GNN + BC | 124.70 | 149.29 | nan |
| RL lai (β + TD-A*) | 36.80 | 96.97 | nan |


### So sánh ghép cặp

| Cặp (A − B) | n | Hiệu TB (phút) | CI thấp | CI cao | p |
|---|---|---|---|---|---|
| Plan-once TD-A* − Dijkstra (ngắn nhất) | 10 | -0.473 | -1.177 | 0.416 | 0.438 |
| Replan TD-A* − Plan-once TD-A* | 10 | -0.125 | -0.305 | 0.015 | 0.375 |
| Oracle TD-A* − Replan TD-A* | 10 | -0.268 | -0.752 | 0.000 | 0.500 |
| Replan lười − Replan TD-A* | 10 | 0.000 | 0.000 | 0.000 | 1.000 |
| Replan D* Lite − Replan TD-A* | 10 | 0.006 | 0.000 | 0.016 | 0.500 |
| Replan + GCN − Replan TD-A* | 10 | 0.005 | 0.000 | 0.015 | 1.000 |
| Plan-once + cảm biến − Plan-once TD-A* | 10 | -0.233 | -0.767 | 0.139 | 0.625 |
| Plan-once + GCN − Plan-once TD-A* | 10 | 0.000 | 0.000 | 0.000 | 1.000 |
| Plan-once + GCN − Plan-once + cảm biến | 10 | 0.233 | -0.139 | 0.767 | 0.625 |
| Định tuyến P90 − Plan-once TD-A* | 10 | 0.123 | -0.012 | 0.314 | 0.625 |
| RL lai (β + TD-A*) − Replan TD-A* | 10 | -0.190 | -0.570 | 0.000 | 1.000 |


### Dự báo GCN

| Mô hình | RMSE chưa quan sát | RMSE mọi cạnh |
|---|---|---|
| Dự báo nền V_base (log-tỉ số = 0) | 0.086 | 0.086 |
| GCN chưa huấn luyện | 0.162 | 0.162 |
| GCN đã huấn luyện | 0.086 | 0.086 |


### Định tuyến P90

| Tuyến | TB | P90 | Std |
|---|---|---|---|
| Plan-once TD-A* | 15.92 | 17.16 | 1.07 |
| Định tuyến P90 | 16.07 | 17.19 | 0.98 |


### Nhiều xe

| Số xe | Không tương tác | Naive | Phối hợp |
|---|---|---|---|
| 400.00 | 27.76 | 28.24 | 28.24 |
| 800.00 | 28.13 | 29.16 | 28.96 |


### Chọn giờ xuất phát

| w | Chính sách | Chi phí TB | Tỉ lệ chờ | Regret |
|---|---|---|---|---|
| 0.250 | Đi ngay | 12.629 | 0.000 | 0.636 |
| 0.250 | Giờ tối ưu (vét cạn) | 11.992 | 0.100 | 0.000 |
| 0.250 | Q-learning | 12.475 | 0.200 | 0.483 |
| 0.500 | Đi ngay | 12.629 | 0.000 | 0.000 |
| 0.500 | Giờ tối ưu (vét cạn) | 12.629 | 0.000 | 0.000 |
| 0.500 | Q-learning | 13.130 | 0.200 | 0.502 |
| 1.000 | Đi ngay | 12.629 | 0.000 | 0.000 |
| 1.000 | Giờ tối ưu (vét cạn) | 12.629 | 0.000 | 0.000 |
| 1.000 | Q-learning | 12.629 | 0.000 | 0.000 |


Hình: `reports/figures/*.png|pdf`; bảng LaTeX: `reports/tables/*.tex`; báo cáo: `reports/paper/main.pdf`.
