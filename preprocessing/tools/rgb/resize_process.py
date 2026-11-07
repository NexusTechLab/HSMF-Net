import cv2
import numpy as np
from pathlib import Path
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage import filters, morphology, measure, segmentation, feature
from scipy import ndimage as ndi
from tqdm.auto import tqdm


def imread_unicode(path: Path) -> np.ndarray | None:
	"""通过 np.fromfile + cv2.imdecode 解决 Windows 非 ASCII 路径读取问题"""
	try:
		data = np.fromfile(str(path), dtype=np.uint8)
		if data.size == 0:
			return None
		return cv2.imdecode(data, cv2.IMREAD_COLOR)
	except Exception:
		return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
	"""通过 cv2.imencode + tofile 写入非 ASCII 路径"""
	ext = path.suffix if path.suffix else ".png"
	ok, buf = cv2.imencode(ext, image)
	if not ok:
		return False
	try:
		buf.tofile(str(path))
		return True
	except Exception:
		return False


def crop_to_foreground(bgr_img: np.ndarray, black_threshold: int = 15, padding: int = 5) -> np.ndarray:
	"""
	裁剪到前景区域，去除黑色背景，保留小麦主体。
	通过亮度阈值找到非黑像素的包围盒并裁剪。
	"""
	gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
	mask = (gray > black_threshold).astype(np.uint8)
	ys, xs = np.where(mask > 0)
	if ys.size == 0 or xs.size == 0:
		return bgr_img
	y1, y2 = int(ys.min()), int(ys.max())
	x1, x2 = int(xs.min()), int(xs.max())
	h, w = bgr_img.shape[:2]
	x1 = max(0, x1 - padding)
	y1 = max(0, y1 - padding)
	x2 = min(w, x2 + 1 + padding)
	y2 = min(h, y2 + 1 + padding)
	return bgr_img[y1:y2, x1:x2].copy()


def remove_background_with_steps(bgr_img: np.ndarray, bg_color: tuple = (0, 0, 0), save_steps: bool = False) -> tuple:
	"""
	小麦背景去除流程（Alpha Matting 版）
	三区域 Trimap + 亮度Alpha，完整保留芒/刷毛等细小特征
	
	返回: (final_result, steps_dict) 
	"""
	h, w = bgr_img.shape[:2]
	steps = {}
	
	# ========== 步骤1: 原始图像 ==========
	steps['01_original'] = bgr_img.copy()
	
	# ========== 步骤2: 多通道特征图 ==========
	hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
	_, S, V = cv2.split(hsv)
	gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
	
	v_norm = V.astype(np.float32) / 255.0
	gray_norm = gray.astype(np.float32) / 255.0
	s_norm = S.astype(np.float32) / 255.0
	
	feature_map = 0.5 * v_norm + 0.3 * gray_norm + 0.2 * s_norm
	
	try:
		otsu_thresh = filters.threshold_otsu(feature_map)
	except Exception:
		otsu_thresh = 0.3
	
	high_thresh = otsu_thresh * 0.8
	low_thresh = otsu_thresh * 0.2
	
	binary_mask = (feature_map > high_thresh).astype(np.uint8) * 255
	steps['02_binary_mask'] = binary_mask
	
	# ========== 步骤3: 极轻量形态学（仅填孔，不磨边）==========
	min_area = int(h * w * 0.0003)
	cleaned = morphology.remove_small_objects(binary_mask > 0, min_size=min_area)
	cleaned_u8 = cleaned.astype(np.uint8) * 255
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	closed = cv2.morphologyEx(cleaned_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
	filled = ndi.binary_fill_holes(closed > 0).astype(np.uint8) * 255
	
	contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if contours:
		largest = max(contours, key=cv2.contourArea)
		coarse_mask = np.zeros((h, w), dtype=np.uint8)
		cv2.drawContours(coarse_mask, [largest], -1, 255, -1)
	else:
		coarse_mask = filled
	
	steps['03_coarse_mask'] = coarse_mask
	
	# ========== 步骤4: 扩展搜索区 + 低阈值捕获细小特征 ==========
	kern_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	search_region = cv2.dilate(coarse_mask, kern_d, iterations=4)
	
	fine_mask = ((feature_map > low_thresh) & (search_region > 0)).astype(np.uint8) * 255
	combined = np.maximum(coarse_mask, fine_mask)
	
	min_fine = max(int(h * w * 0.00005), 10)
	combined_clean = morphology.remove_small_objects(combined > 0, min_size=min_fine)
	combined = ndi.binary_fill_holes(combined_clean).astype(np.uint8) * 255
	
	steps['04_fine_mask'] = combined
	
	# ========== 步骤5: Trimap Alpha 遮罩 ==========
	kern_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	sure_fg = cv2.erode(coarse_mask, kern_e, iterations=2)
	if np.count_nonzero(sure_fg) < np.count_nonzero(coarse_mask) * 0.15:
		sure_fg = cv2.erode(coarse_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
	
	edge_zone = (combined > 0) & (sure_fg == 0)
	
	alpha = np.zeros((h, w), dtype=np.float32)
	alpha[sure_fg > 0] = 1.0
	
	if np.any(edge_zone):
		alpha_vals = (feature_map - low_thresh) / max(high_thresh - low_thresh, 0.01)
		alpha_vals = np.clip(alpha_vals, 0.0, 1.0)
		alpha[edge_zone] = alpha_vals[edge_zone]
	
	# 仅在 combined mask 内部做平滑，防止 alpha 扩散到纯背景
	mask_region = combined > 0
	alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0.5)
	alpha = np.where(mask_region, alpha_blurred, 0.0).astype(np.float32)
	
	# 硬截断：极低 alpha 归零，消除黄膜
	alpha[alpha < 0.1] = 0.0
	alpha = np.clip(alpha, 0.0, 1.0)
	
	steps['05_alpha'] = (alpha * 255).astype(np.uint8)
	
	# ========== 步骤6: Alpha 合成 ==========
	fg_f = bgr_img.astype(np.float32)
	bg_f = np.full_like(fg_f, bg_color, dtype=np.float32)
	a3 = np.stack([alpha, alpha, alpha], axis=-1)
	
	final_result = (fg_f * a3 + bg_f * (1.0 - a3)).astype(np.uint8)
	
	steps['06_final'] = final_result.copy()
	
	return final_result, steps if save_steps else None


def process_one(args: tuple) -> tuple:
	"""工作进程函数：处理一张图片并写入结果/过程图。

	返回: (success: bool, img_name: str, img_time: float)
	"""
	img_path, output_dir, steps_dir, background_color, save_steps, crop_params = args
	start_t = time.time()

	img = imread_unicode(img_path)
	if img is None:
		return False, img_path.name, 0.0

	result, steps = remove_background_with_steps(
		img,
		bg_color=background_color,
		save_steps=save_steps,
	)
	# 裁剪到小麦主体，去除黑色背景，便于模型训练
	threshold, padding = crop_params
	result = crop_to_foreground(result, black_threshold=threshold, padding=padding)
	img_time = time.time() - start_t

	out_path = output_dir / f"{img_path.stem}.png"
	if not imwrite_unicode(out_path, result):
		return False, img_path.name, img_time

	if save_steps and steps_dir and steps:
		img_steps_dir = steps_dir / img_path.stem
		img_steps_dir.mkdir(exist_ok=True)
		for step_name, step_img in steps.items():
			step_path = img_steps_dir / f"{step_name}.png"
			imwrite_unicode(step_path, step_img)

	return True, img_path.name, img_time


def select_folder():
	"""弹出文件夹选择对话框，返回路径字符串；取消则返回空串"""
	try:
		import tkinter as tk
		from tkinter import filedialog
		root = tk.Tk(); root.withdraw()
		folder = filedialog.askdirectory(title="选择待去背景的图片文件夹")
		root.destroy()
		return folder
	except Exception:
		return ""


def main():
	# ========== 参数解析 ==========
	import argparse, sys

	parser = argparse.ArgumentParser(description="批量去除小麦图片背景并裁剪到主体，便于模型训练")
	parser.add_argument('-d', '--dir', metavar='DIR', help='待处理图片文件夹；若省略则弹出对话框')
	parser.add_argument('--bg', metavar='B,G,R', help='背景颜色，如 0,0,0 代表黑色', default='0,0,0')
	parser.add_argument('--save-steps', action='store_true', help='保存过程图（默认否）')
	parser.add_argument('--workers', type=int, default=os.cpu_count() or 1, help='并行进程数，默认等于 CPU 核心数')
	parser.add_argument('--crop-threshold', type=int, default=15, help='裁剪时黑色判定阈值(0-255)，低于此值视为背景')
	parser.add_argument('--crop-padding', type=int, default=5, help='裁剪后边缘留白像素数')

	args = parser.parse_args()

	input_dir = Path(args.dir or select_folder())
	if not input_dir:
		print('未指定文件夹，已退出'); sys.exit(0)

	bgr_vals = tuple(int(x) for x in args.bg.split(',')) if args.bg else (0,0,0)
	SAVE_PROCESS_STEPS = args.save_steps

	# 输出文件夹（去背景 + 裁剪）
	output_dir = Path(str(input_dir) + '_裁剪')
	steps_dir = output_dir.parent / (input_dir.name + '_过程图') if SAVE_PROCESS_STEPS else None

	# 背景颜色设置 (B, G, R)
	BACKGROUND_COLOR = bgr_vals
	CROP_PARAMS = (args.crop_threshold, args.crop_padding)
	
	output_dir.mkdir(parents=True, exist_ok=True)
	if SAVE_PROCESS_STEPS and steps_dir:
		steps_dir.mkdir(parents=True, exist_ok=True)
	
	# 兼容大小写后缀（含 jpg/jpeg/png）
	image_paths = sorted(
		list(input_dir.glob("*.jpg")) +
		list(input_dir.glob("*.JPG")) +
		list(input_dir.glob("*.jpeg")) +
		list(input_dir.glob("*.JPEG")) +
		list(input_dir.glob("*.png")) +
		list(input_dir.glob("*.PNG"))
	)
	if not image_paths:
		print(f"未找到待处理图片：{input_dir}\\*.jpg|*.png")
		return

	total = len(image_paths)
	bg_name = "黑色" if BACKGROUND_COLOR == (0,0,0) else ("白色" if BACKGROUND_COLOR == (255,255,255) else f"RGB{BACKGROUND_COLOR}")
	print(f"找到 {total} 张图片，开始处理...")
	print(f"背景颜色: {bg_name}")
	print(f"裁剪: 阈值={args.crop_threshold}, 留白={args.crop_padding}px")
	print(f"保存过程图: {'是' if SAVE_PROCESS_STEPS else '否'}")
	print("=" * 60)
	
	count = 0
	start_time = time.time()
	workers = max(1, args.workers or 1)

	# 为每一张图片准备参数
	tasks = [
		(
			img_path,
			output_dir,
			steps_dir,
			BACKGROUND_COLOR,
			SAVE_PROCESS_STEPS,
			CROP_PARAMS,
		)
		for img_path in image_paths
	]

	with ProcessPoolExecutor(max_workers=workers) as executor, tqdm(total=total, desc="去背景处理中", unit="图") as pbar:
		futures = {executor.submit(process_one, task): idx for idx, task in enumerate(tasks, 1)}
		for future in as_completed(futures):
			idx = futures[future]
			try:
				success, name, img_time = future.result()
			except Exception as e:
				pbar.write(f"[{idx}/{total}] X 失败: 任务异常 {e}")
				pbar.update(1)
				continue

			if not success:
				pbar.write(f"[{idx}/{total}] X 跳过: {name} (读取或写入失败)")
				pbar.update(1)
				continue

			count += 1
			elapsed = time.time() - start_time
			avg_time = elapsed / count
			remaining = (total - idx) * avg_time
			pbar.update(1)
			pbar.set_postfix_str(f"{name} {img_time:.2f}s, 已完成 {count}/{total}, 预计剩余 {remaining:.1f}s")

	total_time = time.time() - start_time
	print("=" * 60)
	print(f">> 批量去背景+裁剪完成: {count}/{total} 张")
	print(f"   输出目录: {output_dir}")
	if SAVE_PROCESS_STEPS:
		print(f"   过程图: {steps_dir}")
	print(f"   总耗时: {total_time:.1f}s (平均 {total_time/count:.2f}s/张)")



# # 默认裁剪

# # 调整裁剪参数
# python resize_process.py -d "你的图片文件夹" --crop-threshold 20 --crop-padding 10

if __name__ == "__main__":
	main()
