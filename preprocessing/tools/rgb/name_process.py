import os
import re
import argparse
from pathlib import Path


def parse_filename(filename):
    """
    解析文件名，提取前缀、中间数字、后缀数字和扩展名
    
    格式: CX186_XX_YYYY.jpg
    返回: (prefix, middle_num, suffix_num, extension)
    """
        # 支持两种格式：
    # 1. prefix_XX_YYYY.jpg  (原规则)
    # 2. prefix_YYYY.jpg     (省略中间两位，等价于 XX == '00')
    pattern_full = r'^(.+?)_(\d{2})_(\d{4})(\.\w+)$'
    pattern_short = r'^(.+?)_(\d{4})(\.\w+)$'

    match = re.match(pattern_full, filename)
    if match:
        prefix = match.group(1)  # CX186
        middle_num = match.group(2)  # 00, 01, 02, etc.
        suffix_num = match.group(3)  # 0001, 0100, etc.
        extension = match.group(4)  # .jpg
        return prefix, middle_num, suffix_num, extension
    else:
        # 尝试匹配省略中间两位的简写格式
        match = re.match(pattern_short, filename)
        if match:
            prefix = match.group(1)
            middle_num = '00'          # 省略时视为00
            suffix_num = match.group(2)
            extension = match.group(3)
            return prefix, middle_num, suffix_num, extension
    return None


def generate_new_filename(prefix, middle_num, suffix_num, extension):
    """
    根据重命名规则生成新文件名
    
    规则:
    1. 如果中间数字是00，直接去掉中间部分: CX186_00_0001.jpg -> CX186_0001.jpg
    2. 如果中间数字不是00:
       - 计算: 中间数字 + 后缀数字的前两位 = 新前缀数字
       - 新文件名 = prefix_ + 新前缀数字(2位) + 后缀数字的后两位(2位) + extension
    """
    if middle_num == '00':
        # 中间是00，直接去掉中间部分，保留下划线
        new_filename = f"{prefix}_{suffix_num}{extension}"
    else:
        # 中间不是00，需要计算
        middle_int = int(middle_num)
        suffix_prefix = int(suffix_num[:2])  # 后缀数字的前两位
        suffix_suffix = suffix_num[2:]  # 后缀数字的后两位
        
        # 计算新的前缀数字
        new_prefix_num = middle_int + suffix_prefix
        
        # 生成新文件名，新前缀数字保持2位，后缀保持2位，保留下划线
        new_filename = f"{prefix}_{new_prefix_num:02d}{suffix_suffix}{extension}"
    
    return new_filename


def select_folder():
    """
    打开文件夹选择对话框，让用户选择需要重命名的文件夹。
    返回用户选择的文件夹路径；如果用户取消则返回空字符串。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="选择需要重命名的文件夹")
        root.destroy()
        return folder
    except Exception as e:
        print(f"无法打开文件夹选择对话框: {e}")
        return ""


def rename_files_in_folder(folder_path, preview=True, old_prefix=None, new_prefix=None):
    """
    重命名指定文件夹下的所有符合格式的文件
    
    参数:
    folder_path: 文件夹路径
    preview: 如果为True，只预览不实际重命名；如果为False，执行实际重命名
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return
    
    # 获取所有jpg文件
    jpg_files = list(folder.glob('*.jpg'))
    
    if not jpg_files:
        print(f"在文件夹 '{folder_path}' 中未找到jpg文件")
        return
    
    # 解析并准备重命名
    rename_operations = []
    
    for file_path in sorted(jpg_files):
        filename = file_path.name
        parsed = parse_filename(filename)
        
        if parsed is None:
            print(f"跳过 (格式不匹配): {filename}")
            continue
        
        prefix, middle_num, suffix_num, extension = parsed
        new_filename = generate_new_filename(prefix, middle_num, suffix_num, extension)
        new_file_path = file_path.parent / new_filename
        
        # 检查新文件名是否已存在
        if new_file_path.exists() and new_file_path != file_path:
            print(f"警告: 新文件名已存在，跳过: {filename} -> {new_filename}")
            continue
        
        # 处理前缀替换逻辑（如果指定）
        if new_prefix:
            # 若指定旧前缀，仅当匹配时替换；否则全部替换
            if old_prefix is None or prefix == old_prefix:
                new_filename = new_filename.replace(prefix, new_prefix, 1)
                new_file_path = file_path.parent / new_filename
        
        rename_operations.append((file_path, new_file_path, filename, new_filename))
    
    # 显示预览或执行重命名
    if preview:
        print(f"\n预览模式 - 将重命名 {len(rename_operations)} 个文件:\n")
        for old_path, new_path, old_name, new_name in rename_operations:
            print(f"  {old_name} -> {new_name}")
        print(f"\n如需执行重命名，请将 preview=False")
    else:
        print(f"\n开始重命名 {len(rename_operations)} 个文件...\n")
        success_count = 0
        fail_count = 0
        
        for old_path, new_path, old_name, new_name in rename_operations:
            try:
                old_path.rename(new_path)
                print(f"✓ {old_name} -> {new_name}")
                success_count += 1
            except Exception as e:
                print(f"✗ 失败: {old_name} -> {new_name} (错误: {e})")
                fail_count += 1
        
        print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")



# python name_process.py -d "F:\HSI_RGB小麦数据集\RGB_春晓186" --new-prefix CX186 --old-prefix ZM379 --execute
# python name_process.py -d "F:\HSI_RGB小麦数据集\RGB小麦图像_原图像\RGB_许科129" --new-prefix XK129 --execute

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="批量重命名符合 CX186_XX_YYYY.jpg 规则的文件"
    )
    parser.add_argument(
        "-d",
        "--dir",
        metavar="DIR",
        help="目标文件夹路径。如果不提供，则弹出文件夹选择对话框",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行重命名（默认仅预览）",
    )

    parser.add_argument(
        "--old-prefix",
        metavar="OLD",
        help="仅当文件名前缀匹配 OLD 时才替换，默认替换所有前缀",
    )
    parser.add_argument(
        "--new-prefix",
        metavar="NEW",
        help="新的前缀；如果指定则启用前缀替换功能",
    )

    args = parser.parse_args()

    folder_path = args.dir or select_folder()
    if not folder_path:
        print("未指定文件夹，已退出。")
        sys.exit(0)

    preview_mode = not args.execute

    print("=" * 60)
    print("文件重命名工具 - 预览模式" if preview_mode else "文件重命名工具 - 执行模式")
    print("=" * 60)
    rename_files_in_folder(
        folder_path,
        preview=preview_mode,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
    )

