import numpy as np

def normalize_obs_optimized(obs):
    # Ép kiểu về numpy array float32 ngay từ đầu để tính toán ma trận siêu tốc
    state = np.array(obs, copy=True, dtype=np.float32)
    
    n_c = int(state[0, 0])
    if n_c == 0:
        return state
        
    n_t = int(state[0, 1])
    n_o = int(state[0, 2])
    
    # 1. Tối ưu hóa Mapping (Thay thế Dictionary bằng Masking ma trận)
    vals = state[:n_c, 3].astype(int)
    mapping_arr = np.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]], dtype=np.float32)
    encoded = np.zeros((n_c, 4), dtype=np.float32)
    valid_mask = (vals >= 0) & (vals <= 3)
    encoded[valid_mask] = mapping_arr[vals[valid_mask]]
    state[:n_c, 0:4] = encoded
    
    # Lấy original_radius để dùng cho các phép chia sau (Shape: [n_c])
    orig_rad = state[:n_c, 19].copy()
    
    # 2. Xử lý các chỉ số cơ bản của camera (Không dùng vòng lặp)
    state[:n_c, 4:16] /= 1000.0
    state[:n_c, 16:18] /= orig_rad[:, None] # Broadcasting chia cho radius của từng camera
    state[:n_c, 18] /= 180.0
    state[:n_c, 19] = 1.0
    state[:n_c, 20:22] /= 180.0
    
    # 3. Xử lý Targets (Sử dụng Reshape để tạo View - Không tốn bộ nhớ copy)
    if n_t > 0:
        t_start = 22
        t_end = t_start + 5 * n_t
        # Đưa thành ma trận 3D (camera, target, thuộc tính) và xử lý cùng 1 lúc
        targets_view = state[:n_c, t_start:t_end].reshape(n_c, n_t, 5)
        targets_view[:, :, :3] /= 1000.0
        
    # 4. Xử lý Obstacles
    if n_o > 0:
        o_start = 22 + 5 * n_t
        o_end = o_start + 4 * n_o
        obs_view = state[:n_c, o_start:o_end].reshape(n_c, n_o, 4)
        obs_view[:, :, :3] /= 1000.0
        
    # 5. Xử lý Teammates
    if n_c > 0:
        tm_start = 22 + 5 * n_t + 4 * n_o
        tm_end = tm_start + 7 * n_c
        tm_view = state[:n_c, tm_start:tm_end].reshape(n_c, n_c, 7)
        tm_view[:, :, :3] /= 1000.0
        # orig_rad[:, None, None] giúp tự động dãn ma trận để chia đúng radius cho từng camera
        tm_view[:, :, 3:5] /= orig_rad[:, None, None]
        tm_view[:, :, 5] /= 180.0
        
    return state

import numpy as np

def normalize_state_optimized(s, num_cameras, num_targets, num_obstacles):
    # 1. Loại bỏ hoàn toàn copy.deepcopy()
    # Việc tạo numpy array với copy=True sẽ copy vùng nhớ C trực tiếp, nhanh hơn deepcopy hàng chục lần
    state = np.array(s, copy=True, dtype=np.float32)
    
    # 2. Xử lý phần Camera (Mỗi camera chiếm 9 phần tử)
    if num_cameras > 0:
        # Cắt lấy mảng camera và "gập" thành ma trận 2D (num_cameras, 9)
        cam_view = state[:9 * num_cameras].reshape(num_cameras, 9)
        
        cam_view[:, 0:3] /= 1000.0
        
        # Lưu lại cột index 6 (tương ứng state[start_idx+6]) để làm mẫu số
        orig_rad = cam_view[:, 6].copy()
        
        # orig_rad[:, None] dùng để broadcast (dóng hàng) chia đúng radius cho từng camera
        cam_view[:, 3:5] /= orig_rad[:, None]
        
        cam_view[:, 5] /= 180.0
        cam_view[:, 6] = 1.0
        cam_view[:, 7:9] /= 180.0

    # 3. Xử lý phần Target (Mỗi target chiếm 14 phần tử)
    if num_targets > 0:
        t_start = 9 * num_cameras
        t_end = t_start + 14 * num_targets
        # "Gập" thành ma trận 2D (num_targets, 14)
        tgt_view = state[t_start:t_end].reshape(num_targets, 14)
        
        tgt_view[:, 0:3] /= 1000.0
        tgt_view[:, 4] /= 1000.0

    # 4. Xử lý phần Obstacles (Mỗi obstacle chiếm 3 phần tử)
    if num_obstacles > 0:
        o_start = 9 * num_cameras + 14 * num_targets
        o_end = o_start + 3 * num_obstacles
        # "Gập" thành ma trận 2D (num_obstacles, 3)
        obs_view = state[o_start:o_end].reshape(num_obstacles, 3)
        
        # Vì chỉ lấy từ 0:3 (toàn bộ obstacle), ta chia thẳng luôn
        obs_view[:, :] /= 1000.0
        
    return state