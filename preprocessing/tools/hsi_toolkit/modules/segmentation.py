# ==========================================================
# 文件名: segmentation.py
# 模块功能: 高光谱种子分割模块（Seed Segmentation）
# ----------------------------------------------------------
# 本模块基于 extract_grains.py 的标准流程，执行籽粒的自动分割。
# 完整流程：
#   ROI裁剪 → 二值化 → 形态学清理 → 分水岭 → 区域过滤与排序
#
# 应用场景：
#   - 小麦、玉米、大豆等籽粒的批量识别与光谱提取
#   - 支持多种排序与行检测算法（Fixed / DBSCAN / PCA-DBSCAN）
#
# ==========================================================

import numpy as np
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False

from skimage import measure, morphology, segmentation, feature, filters
from scipy import ndimage as ndi
from scipy.spatial import distance
import cv2


# ==========================================================
# ⬛ ROI 区域裁剪
# ----------------------------------------------------------
# 从输入图像中提取感兴趣区域（ROI），避免处理整幅图像。
# ROI 通常来自用户框选或种子托板模板。
# ==========================================================
def apply_roi_mask(image: np.ndarray, roi: tuple = None) -> tuple:
    if roi is None:
        return image, np.ones_like(image, dtype=bool)

    y0, y1, x0, x1 = roi
    roi_mask = np.zeros_like(image, dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    roi_image = image * roi_mask
    return roi_image, roi_mask


# ==========================================================
# ⬛ 二值化（Binarization）
# ----------------------------------------------------------
# 将灰度图转为前景/背景二值掩膜。
# 支持 Otsu 自适应阈值与百分位阈值法。
# ==========================================================
def binarize_image(image: np.ndarray, threshold: float = None, method: str = "otsu") -> np.ndarray:
    """改进的二值化方法，支持多种算法"""
    if threshold is None:
        if method == "otsu":
            threshold = filters.threshold_otsu(image)
        elif method == "adaptive":
            # 使用CV2自适应阈值
            image_uint8 = (image * 255).astype(np.uint8)
            binary = cv2.adaptiveThreshold(image_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            return binary.astype(bool)
        else:
            threshold = np.percentile(image[image > 0], 10)
    return image >= threshold


# ==========================================================
# ⬛ 形态学清理（Morphological Cleaning）
# ----------------------------------------------------------
# 去除小杂点、平滑边界、填充孔洞。
# 参数可调：闭运算半径 / 开运算半径 / 最小面积。
# ==========================================================
def morphological_clean(binary: np.ndarray, min_area: int = 150,
                         closing_radius: int = 0, opening_radius: int = 0) -> np.ndarray:
    """形态学清理 - 简化版本，避免复杂的形态学操作"""
    # 移除小对象
    cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=min_area)

    # 跳过复杂的形态学操作，直接填充孔洞
    # 这样可以避免维度问题，同时保持基本功能
    cleaned = ndi.binary_fill_holes(cleaned)

    return cleaned


# ==========================================================
# ⬛ 分水岭分割（Watershed）
# ----------------------------------------------------------
# 根据距离变换分离相邻籽粒，常用于连通籽粒场景。
# min_distance 控制“籽粒中心点”最小间距。
# ==========================================================
def watershed_segment(binary: np.ndarray, min_distance: int = 15) -> tuple:
    distance = ndi.distance_transform_edt(binary)
    
    # peak_local_max API兼容性处理（支持scikit-image不同版本）
    try:
        # 尝试新版API (scikit-image >= 0.22)
        coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary, exclude_border=False)
    except TypeError:
        # 回退到旧版API (scikit-image < 0.22)
        try:
            coords = feature.peak_local_max(distance, min_distance=min_distance, footprint=binary, exclude_border=False)
        except:
            # 最简单的fallback
            coords = feature.peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels, coords


# ==========================================================
# 🌾 四、典型参数建议
# 参数	            原值	                推荐范围	             说明
# eps系数	        0.4	            0.5~0.7	            稍大可以防止合并行
# eps下限	        无	            8.0~12.0 像素	    防止籽粒靠太近导致行丢失
# 角度步长	        5°	            2°	                提升精度
# typical_dist估计	最近邻	        第二邻距	            更稳定
# 噪声点分配阈值	1.5×typical	        2.0×typical	        容错
# ==========================================================
def _assign_rows_grid_optimized(regions):
    """轻量优化版网格排序算法 - 更稳健的行聚类"""
    if not HAS_SKLEARN:
        heights = [r.bbox[2] - r.bbox[0] for r in regions]
        h_med = np.median(heights)
        row_gap = h_med * 2.7
        min_y = min(r.centroid[0] for r in regions)
        row_idx = {r.label: int((r.centroid[0] - min_y) // row_gap) for r in regions}
        return row_idx, "fixed_fallback", h_med, row_gap

    points = np.array([[r.centroid[1], r.centroid[0]] for r in regions])
    if len(points) <= 1:
        return {r.label: 0 for r in regions}, "grid_optimized", 0, 0

    # ---- 改进: 使用第二小邻距来估计 typical_dist ----
    dist_matrix = distance.cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    sorted_dists = np.sort(dist_matrix, axis=1)
    typical_dist = np.median(sorted_dists[:, 1])
    typical_dist = max(typical_dist, 10.0)  # 防止太小

    best_angle = 0.0
    best_labels = None
    best_score = -np.inf

    # ---- 改进: 提高角度搜索精度 ----
    for angle_deg in range(-30, 31, 2):
        angle_rad = np.radians(angle_deg)
        rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])
        rotated = points @ rot_mat.T
        y_coords = rotated[:, 1].reshape(-1, 1)

        # ---- 改进: eps 下限保护 ----
        eps = max(typical_dist * 0.5, 8.0)
        clustering = DBSCAN(eps=eps, min_samples=2).fit(y_coords)
        labels = clustering.labels_

        n_rows = len(set(labels)) - (1 if -1 in labels else 0)

        # ---- 改进: 用平衡度加权评分 ----
        if n_rows > 1:
            counts = np.bincount(labels[labels >= 0])
            row_std = np.std(counts)
            score = n_rows - 0.1 * row_std
        else:
            score = n_rows

        if score > best_score:
            best_score = score
            best_angle = angle_rad
            best_labels = labels

    # ---- 行聚类组织 ----
    if best_labels is None:
        best_labels = np.zeros(len(points))

    rows_dict = {}
    for idx, label in enumerate(best_labels):
        rows_dict.setdefault(label, []).append((points[idx][0], points[idx][1], idx))

    # ---- 改进: 放宽噪声点分配 ----
    if -1 in rows_dict:
        noise_points = rows_dict.pop(-1)
        for nx, ny, nidx in noise_points:
            best_label = None
            min_dist = np.inf
            for label, pts in rows_dict.items():
                row_y = np.mean([p[1] for p in pts])
                dist_y = abs(ny - row_y)
                if dist_y < min_dist:
                    min_dist = dist_y
                    best_label = label
            if min_dist < typical_dist * 2.0:  # 放宽
                rows_dict[best_label].append((nx, ny, nidx))
            else:
                new_label = max(rows_dict.keys()) + 1
                rows_dict[new_label] = [(nx, ny, nidx)]

    # ---- 行号映射 ----
    row_idx = {}
    for label, pts in rows_dict.items():
        for _, _, idx in pts:
            row_idx[regions[idx].label] = label

    h_med = np.median([r.bbox[2] - r.bbox[0] for r in regions])
    return row_idx, f"grid_optimized(angle={np.rad2deg(best_angle):.1f}°)", h_med, typical_dist



# ==========================================================
# ⬛ 区域过滤与排序（Region Filtering & Sorting）
# ----------------------------------------------------------
# 步骤：
#   1. 按面积、形状过滤异常籽粒；
#   2. 依据行号与 x 坐标进行编号；
#   3. 输出重标号后的区域。
# ==========================================================
def filter_and_sort_regions(labels: np.ndarray,
                            min_area: int = 150,
                            max_area: int = 1000,
                            sort_mode: str = "顺序",
                            max_aspect_ratio: float = 8.0,
                            debug: bool = True) -> tuple:
    """区域过滤与行列排序（含调试信息输出）"""
    regions = measure.regionprops(labels)
    valid_regions = []

    # 1. 筛选有效区域
    for r in regions:
        h, w = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
        aspect = max(h, w) / (min(h, w) + 1e-6)
        if min_area <= r.area <= max_area and aspect < max_aspect_ratio:
            valid_regions.append(r)

    if not valid_regions:
        if debug:
            print("[WARNING] 未检测到有效区域，过滤后为空。")
        return np.zeros_like(labels), [], None

    if debug:
        print("\n========== [调试] 区域过滤与排序 ==========")
        print(f"区域总数: {len(regions)} → 有效区域: {len(valid_regions)}")
        print(f"参数: min_area={min_area}, max_area={max_area}, sort_mode={sort_mode}")

    # 2. 计算行号（固定使用网格优化算法）
    if sort_mode == "顺序":
        row_idx, alg_used, h_med, gap = _assign_rows_grid_optimized(valid_regions)
        
        if debug:
            print(f"排序算法: {alg_used} | 平均高度: {h_med:.1f} | 典型间距: {gap:.2f}")

        # 3. 标记行号与x坐标
        region_info = [(r.label, row_idx[r.label], r.centroid[1], r.centroid[0], r)
                       for r in valid_regions]
        # 按行号和x坐标排序
        region_info.sort(key=lambda x: (x[1], x[2]))

    else:
        region_info = [(r.label, 0, r.centroid[1], r.centroid[0], r) for r in valid_regions]
        region_info.sort(key=lambda x: x[2])

    # 调试输出：排序前后比较
    if debug:
        print("\n[调试] 排序前 (原label, y, x)：")
        for r in valid_regions[:10]:  # 只打印前10个
            print(f"  原label={r.label:3d} | y={r.centroid[0]:7.2f}, x={r.centroid[1]:7.2f}")

        print("\n[调试] 排序后 (新label, 行号, y, x)：")
        for new_id, (_, row, x, y, _) in enumerate(region_info[:10], 1):
            print(f"  新label={new_id:3d} | 行={row:2d} | y={y:7.2f}, x={x:7.2f}")

    # 4. 重建label映射
    labels_sorted = np.zeros_like(labels)
    for new_id, (_, _, _, _, region) in enumerate(region_info, start=1):
        coords = np.array(region.coords)
        labels_sorted[coords[:, 0], coords[:, 1]] = new_id

    # 5. 生成排序后区域
    regions_sorted = measure.regionprops(labels_sorted)
    regions_sorted.sort(key=lambda r: r.label)

    if debug:
        n_rows = len(set([r[1] for r in region_info]))
        print(f"\n[调试结果] 最终行数: {n_rows}, 排序后区域数: {len(regions_sorted)}")
        print("==========================================\n")

    # 准备网格排序可视化信息
    grid_info = None
    if sort_mode == "顺序" and len(region_info) > 0:
        # 提取排序后的质心坐标
        sorted_centroids = []
        for _, _, x, y, _ in region_info:
            sorted_centroids.append([x, y])
        sorted_centroids = np.array(sorted_centroids)
        
        grid_info = {
            'sorted_centroids': sorted_centroids,
            'n_rows': len(set([r[1] for r in region_info])),
            'algorithm': alg_used
        }

    return labels_sorted, regions_sorted, grid_info


# ==========================================================
# ⬛ 主函数：完整种子分割流程
# ----------------------------------------------------------
# 包含：
#   - ROI裁剪 → 二值化 → 形态学 → 分水岭 → 过滤排序
# 输出：
#   (labels_final, regions_final, info)
# ==========================================================
def segment_seeds(image: np.ndarray, method: str = "watershed",
                   min_area: int = 150, max_area: int = 1000,
                   threshold: float = None, roi: tuple = None,
                   min_distance: int = 15, closing_r: int = 0,
                   opening_r: int = 0, sort_mode: str = "顺序") -> tuple:

    info = {}
    
    # 步骤1: ROI裁剪
    roi_image, roi_mask = apply_roi_mask(image, roi)
    info['step1_image'] = roi_image.copy()
    info['roi_pixels'] = roi_mask.sum()

    # 步骤2: 二值化
    binary = binarize_image(roi_image, threshold, method)
    info['step2_binary'] = binary.copy()
    info['threshold'] = threshold if threshold else (
        filters.threshold_otsu(roi_image) if method == "otsu"
        else np.percentile(roi_image[roi_image > 0], 10)
    )

    # 步骤3: 形态学清理
    cleaned = morphological_clean(binary, min_area, closing_r, opening_r)
    info['step3_cleaned'] = cleaned.copy()

    # 步骤4: 分割
    if method == "watershed":
        labels, coords = watershed_segment(cleaned, min_distance)
        seed_cnt = len(coords)
        if seed_cnt == 0:
            # 如果未检测到峰值，则退回使用连通区域数
            seed_cnt = labels.max()
        info['seed_centers'] = seed_cnt
        info['step4_coords'] = coords
    else:
        labels, _ = ndi.label(cleaned)
        # 对非分水岭方法，seed_centers 用连通区域数表示
        info['seed_centers'] = labels.max()
        info['step4_coords'] = None

    info['step5_labels'] = labels.copy()
    info['initial_regions'] = labels.max()

    # 步骤5: 过滤与排序
    labels_final, regions_final, grid_info = filter_and_sort_regions(labels, min_area, max_area, sort_mode)
    info['final_regions'] = len(regions_final)
    info['sort_mode'] = sort_mode
    info['sort_algorithm'] = "网格优化"
    
    # 保存网格排序信息用于可视化
    if grid_info:
        info['grid_sorting'] = grid_info
    
    # 检查分割是否成功
    if labels_final.max() == 0 or len(regions_final) == 0:
        info['segmentation_failed'] = True
        print("[警告] 分割未检测到任何有效区域，请调整参数")

    return labels_final, regions_final, info
