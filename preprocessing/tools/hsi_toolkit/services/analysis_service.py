"""
分析服务 - 籽粒健康等级分析和数据质量诊断
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib

from config import Config
from utils import get_logger, handle_errors, FileHelper

logger = get_logger(__name__)


class AnalysisService:
    """数据分析服务 - 健康等级分析和数据质量诊断"""
    
    # 健康等级判定标准（基于关键波长的反射率阈值）
    HEALTH_CRITERIA = {
        900: {'Grade 0': 0.30, 'Grade 1': 0.28, 'Grade 2': 0.26, 'Grade 3': 0.23},
        970: {'Grade 0': 0.06, 'Grade 1': 0.055, 'Grade 2': 0.050, 'Grade 3': 0.045},
        1100: {'Grade 0': 0.32, 'Grade 1': 0.30, 'Grade 2': 0.27, 'Grade 3': 0.24},
        1200: {'Grade 0': 0.31, 'Grade 1': 0.29, 'Grade 2': 0.27, 'Grade 3': 0.23},
        1400: {'Grade 0': 0.25, 'Grade 1': 0.22, 'Grade 2': 0.20, 'Grade 3': 0.18},
        1450: {'Grade 0': 0.12, 'Grade 1': 0.10, 'Grade 2': 0.09, 'Grade 3': 0.07},
        1600: {'Grade 0': 0.10, 'Grade 1': 0.09, 'Grade 2': 0.08, 'Grade 3': 0.07},
        1700: {'Grade 0': 0.08, 'Grade 1': 0.075, 'Grade 2': 0.070, 'Grade 3': 0.065}
    }
    
    GRADE_DESC = {
        'Grade 0': '健康 (Healthy)',
        'Grade 1': '轻微霉变 (Mild Mold)',
        'Grade 2': '中度霉变 (Moderate Mold)',
        'Grade 3': '重度霉变 (Severe Mold)'
    }
    
    @staticmethod
    @handle_errors(title="籽粒健康等级分析")
    def analyze_grain_health(csv_path: Path, output_dir: Path = None) -> Tuple[Path, Dict]:
        """
        分析籽粒健康等级
        
        Args:
            csv_path: 光谱CSV文件路径
            output_dir: 输出目录，默认与输入文件同目录
        
        Returns:
            tuple: (result_csv_path, statistics_dict)
        """
        logger.info(f"开始分析籽粒健康等级: {csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        if 'Wavelength(nm)' not in df.columns:
            raise ValueError("CSV文件缺少 'Wavelength(nm)' 列")
        
        wavelengths = df['Wavelength(nm)'].values
        grain_columns = [col for col in df.columns if col != 'Wavelength(nm)']
        
        if len(grain_columns) == 0:
            raise ValueError("CSV文件中没有籽粒数据")
        
        # 分析每个籽粒
        results = []
        for grain_name in grain_columns:
            spectrum = df[grain_name].values
            grade, confidence, scores = AnalysisService._classify_grain(
                wavelengths, spectrum
            )
            
            results.append({
                '籽粒名称': grain_name,
                '健康等级': grade,
                '等级描述': AnalysisService.GRADE_DESC[grade],
                '置信度(%)': f"{confidence:.1f}"
            })
        
        # 保存结果
        if output_dir is None:
            output_dir = csv_path.parent
        
        output_path = Path(output_dir) / "grain_health_analysis.csv"
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 统计
        grade_counts = result_df['健康等级'].value_counts().to_dict()
        stats = {
            'total': len(results),
            'grade_counts': grade_counts,
            'output_path': str(output_path)
        }
        
        logger.info(f"健康等级分析完成: {len(results)}个籽粒, 输出: {output_path}")
        return output_path, stats
    
    @staticmethod
    def _classify_grain(wavelengths: np.ndarray, 
                       spectrum: np.ndarray) -> Tuple[str, float, Dict]:
        """
        对单个籽粒进行健康等级分类
        
        Returns:
            tuple: (grade, confidence, grade_scores)
        """
        grade_scores = {'Grade 0': 0, 'Grade 1': 0, 'Grade 2': 0, 'Grade 3': 0}
        
        # 对每个关键波长进行评分
        for target_wl, thresholds in AnalysisService.HEALTH_CRITERIA.items():
            idx = np.argmin(np.abs(wavelengths - target_wl))
            reflectance = spectrum[idx]
            
            # 根据反射率判定等级
            if reflectance >= thresholds['Grade 0']:
                grade_scores['Grade 0'] += 1
            elif reflectance >= thresholds['Grade 1']:
                grade_scores['Grade 1'] += 1
            elif reflectance >= thresholds['Grade 2']:
                grade_scores['Grade 2'] += 1
            else:
                grade_scores['Grade 3'] += 1
        
        # 选择得分最高的等级
        final_grade = max(grade_scores, key=grade_scores.get)
        confidence = grade_scores[final_grade] / len(AnalysisService.HEALTH_CRITERIA) * 100
        
        return final_grade, confidence, grade_scores
    
    @staticmethod
    @handle_errors(title="数据质量诊断")
    def diagnose_data_quality(csv_path: Path, 
                              output_dir: Path = None) -> Tuple[Path, Path, Path, Dict]:
        """
        数据质量诊断 - 检查光谱数据是否存在干扰
        
        Args:
            csv_path: 光谱CSV文件路径
            output_dir: 输出目录，默认与输入文件同目录
        
        Returns:
            tuple: (diagnosis_csv, report_txt, chart_png, statistics)
        """
        logger.info(f"开始数据质量诊断: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        if 'Wavelength(nm)' not in df.columns:
            raise ValueError("CSV文件缺少 'Wavelength(nm)' 列")
        
        wavelengths = df['Wavelength(nm)'].values
        grain_columns = [col for col in df.columns if col != 'Wavelength(nm)']
        
        if len(grain_columns) == 0:
            raise ValueError("CSV文件中没有籽粒数据")
        
        # 健康籽粒的标准阈值（Grade 0）
        healthy_standards = {wl: thresholds['Grade 0'] 
                           for wl, thresholds in AnalysisService.HEALTH_CRITERIA.items()}
        
        # 分析结果
        diagnosis_results = []
        abnormal_grains = []
        
        for grain_name in grain_columns:
            spectrum = df[grain_name].values
            issues = []
            abnormal_wavelengths = []
            
            # 检查关键波长
            for target_wl, expected_min in healthy_standards.items():
                idx = np.argmin(np.abs(wavelengths - target_wl))
                actual_wl = wavelengths[idx]
                reflectance = spectrum[idx]
                
                if reflectance < expected_min:
                    deviation = (expected_min - reflectance) / expected_min * 100
                    abnormal_wavelengths.append({
                        'wavelength': int(actual_wl),
                        'expected': expected_min,
                        'actual': reflectance,
                        'deviation': deviation
                    })
            
            # 检查其他异常
            mean_ref = np.nanmean(spectrum)
            std_ref = np.nanstd(spectrum)
            
            if np.any(spectrum < 0):
                issues.append("存在负值")
            if np.any(spectrum > 1.0):
                issues.append("存在超过1.0的异常高值")
            if mean_ref < 0.05:
                issues.append("整体反射率过低")
            if std_ref > 0.15:
                issues.append("光谱波动过大")
            
            diagnosis_results.append({
                '籽粒名称': grain_name,
                '异常波长数': len(abnormal_wavelengths),
                '平均反射率': f"{mean_ref:.4f}",
                '标准差': f"{std_ref:.4f}",
                '问题描述': '; '.join(issues) if issues else '正常'
            })
            
            if len(abnormal_wavelengths) > 0 or len(issues) > 0:
                abnormal_grains.append({
                    'name': grain_name,
                    'abnormal_wls': abnormal_wavelengths,
                    'issues': issues
                })
        
        # 设置输出目录
        if output_dir is None:
            output_dir = csv_path.parent
        output_dir = Path(output_dir)
        
        # 保存诊断CSV
        diagnosis_csv = output_dir / "data_quality_diagnosis.csv"
        result_df = pd.DataFrame(diagnosis_results)
        result_df.to_csv(diagnosis_csv, index=False, encoding='utf-8-sig')
        
        # 生成详细报告
        report_txt = output_dir / "diagnosis_report.txt"
        AnalysisService._save_diagnosis_report(
            report_txt, csv_path, grain_columns, abnormal_grains, healthy_standards
        )
        
        # 生成可视化图表
        chart_png = output_dir / "diagnosis_charts.png"
        AnalysisService._plot_diagnosis_charts(
            df, wavelengths, grain_columns, abnormal_grains, healthy_standards, chart_png
        )
        
        # 统计信息
        stats = {
            'total': len(grain_columns),
            'abnormal': len(abnormal_grains),
            'normal': len(grain_columns) - len(abnormal_grains),
            'health_rate': (1 - len(abnormal_grains)/len(grain_columns))*100
        }
        
        logger.info(f"数据质量诊断完成: {len(grain_columns)}个籽粒, 异常{len(abnormal_grains)}个")
        return diagnosis_csv, report_txt, chart_png, stats
    
    @staticmethod
    def _save_diagnosis_report(report_path: Path, 
                               csv_path: Path,
                               grain_columns: List,
                               abnormal_grains: List,
                               healthy_standards: Dict):
        """保存详细诊断报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("高光谱数据质量诊断报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据文件: {csv_path.name}\n")
            f.write(f"总籽粒数: {len(grain_columns)}\n")
            f.write(f"异常籽粒数: {len(abnormal_grains)}\n")
            f.write(f"健康率: {(1 - len(abnormal_grains)/len(grain_columns))*100:.1f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("健康籽粒标准（Grade 0）:\n")
            f.write("-" * 80 + "\n")
            for wl, threshold in healthy_standards.items():
                f.write(f"  {wl} nm: ≥ {threshold:.3f}\n")
            f.write("\n")
            
            if len(abnormal_grains) > 0:
                f.write("-" * 80 + "\n")
                f.write("异常籽粒详细信息:\n")
                f.write("-" * 80 + "\n\n")
                
                for grain in abnormal_grains:
                    f.write(f"【{grain['name']}】\n")
                    
                    if grain['abnormal_wls']:
                        f.write("  低于健康标准的波长:\n")
                        for wl_info in grain['abnormal_wls']:
                            f.write(f"    {wl_info['wavelength']} nm: "
                                   f"实际={wl_info['actual']:.4f}, "
                                   f"标准={wl_info['expected']:.3f}, "
                                   f"偏差={wl_info['deviation']:.1f}%\n")
                    
                    if grain['issues']:
                        f.write("  其他问题:\n")
                        for issue in grain['issues']:
                            f.write(f"    - {issue}\n")
                    
                    f.write("\n")
            else:
                f.write("-" * 80 + "\n")
                f.write("✅ 所有籽粒数据质量良好，符合健康标准！\n")
                f.write("-" * 80 + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("可能的干扰因素分析:\n")
            f.write("=" * 80 + "\n")
            f.write("1. 背景干扰: 分割掩膜可能包含背景像素\n")
            f.write("2. 光照不均: 成像时光照条件不一致\n")
            f.write("3. 暗电流噪声: 相机暗电流未正确校正\n")
            f.write("4. 白板校正: 反射率校正不准确\n")
            f.write("5. 边缘效应: 籽粒边缘像素混入背景\n")
            f.write("6. 数据归一化: 归一化方法不当\n")
    
    @staticmethod
    def _plot_diagnosis_charts(df: pd.DataFrame,
                               wavelengths: np.ndarray,
                               grain_columns: List,
                               abnormal_grains: List,
                               healthy_standards: Dict,
                               output_path: Path):
        """绘制诊断图表"""
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(16, 10))
        
        # 子图1: 所有光谱曲线
        ax1 = plt.subplot(2, 2, 1)
        abnormal_names = [g['name'] for g in abnormal_grains]
        for grain_name in grain_columns:
            spectrum = df[grain_name].values
            if grain_name in abnormal_names:
                ax1.plot(wavelengths, spectrum, 'r-', alpha=0.3, linewidth=1)
            else:
                ax1.plot(wavelengths, spectrum, 'g-', alpha=0.3, linewidth=1)
        
        # 标注健康标准线
        for wl, threshold in healthy_standards.items():
            idx = np.argmin(np.abs(wavelengths - wl))
            ax1.plot(wavelengths[idx], threshold, 'b*', markersize=10)
        
        ax1.set_xlabel('波长 (nm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('反射率', fontsize=12, fontweight='bold')
        ax1.set_title('所有籽粒光谱 (红色=异常, 绿色=正常, 蓝星=标准)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 平均光谱 vs 标准
        ax2 = plt.subplot(2, 2, 2)
        mean_spectrum = df[grain_columns].mean(axis=1).values
        ax2.plot(wavelengths, mean_spectrum, 'b-', linewidth=2, label='平均光谱')
        
        std_wls = []
        std_vals = []
        for wl, threshold in healthy_standards.items():
            idx = np.argmin(np.abs(wavelengths - wl))
            std_wls.append(wavelengths[idx])
            std_vals.append(threshold)
        ax2.plot(std_wls, std_vals, 'ro', markersize=8, label='健康标准')
        
        ax2.set_xlabel('波长 (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('反射率', fontsize=12, fontweight='bold')
        ax2.set_title('平均光谱 vs 健康标准', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 关键波长反射率分布
        ax3 = plt.subplot(2, 2, 3)
        key_wavelengths = list(healthy_standards.keys())
        box_data = []
        for wl in key_wavelengths:
            idx = np.argmin(np.abs(wavelengths - wl))
            values = [df[grain][idx] for grain in grain_columns]
            box_data.append(values)
        
        bp = ax3.boxplot(box_data, labels=[str(w) for w in key_wavelengths])
        
        # 标注标准线
        for i, (wl, threshold) in enumerate(healthy_standards.items(), 1):
            ax3.axhline(y=threshold, xmin=(i-0.4)/len(key_wavelengths),
                       xmax=(i+0.4)/len(key_wavelengths),
                       color='r', linestyle='--', linewidth=2)
        
        ax3.set_xlabel('波长 (nm)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('反射率', fontsize=12, fontweight='bold')
        ax3.set_title('关键波长反射率分布 (红线=标准)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 子图4: 异常统计
        ax4 = plt.subplot(2, 2, 4)
        normal_count = len(grain_columns) - len(abnormal_grains)
        abnormal_count = len(abnormal_grains)
        
        colors = ['#2ecc71', '#e74c3c']
        labels = [f'正常\n({normal_count}颗)', f'异常\n({abnormal_count}颗)']
        sizes = [normal_count, abnormal_count]
        
        if abnormal_count > 0:
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        else:
            ax4.pie([100], labels=['全部正常\n(100%)'], colors=['#2ecc71'],
                   startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        ax4.set_title('数据质量统计', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

