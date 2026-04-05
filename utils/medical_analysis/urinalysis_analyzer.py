"""
尿常规分析模块

功能：
- 解析尿常规报告文本
- 实现定性指标解析（尿蛋白、尿糖、尿潜血等）
- 建立异常指标与临床意义的映射关系
- 生成结构化分析报告
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .medical_reference import (
    medical_db,
    RiskLevel,
    QualitativeReference,
    ReferenceRange
)

logger = logging.getLogger(__name__)


@dataclass
class UrinalysisIndicator:
    """尿常规指标数据类"""
    name: str
    name_en: str
    value: str
    status: str = "正常"
    risk_level: RiskLevel = RiskLevel.LOW
    reference: Optional[QualitativeReference] = None
    clinical_significance: str = ""


@dataclass
class UrinalysisAnalysisResult:
    """尿常规分析结果数据类"""
    indicators: List[UrinalysisIndicator] = field(default_factory=list)
    abnormal_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    diagnosis_hints: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""


class UrinalysisAnalyzer:
    """尿常规分析器"""
    
    def __init__(self):
        self.db = medical_db
        self.indicator_patterns = self._build_indicator_patterns()
    
    def _build_indicator_patterns(self) -> Dict[str, str]:
        """
        构建指标识别的正则表达式模式。
        
        Returns:
            Dict[str, str]: 指标名称到正则模式的映射
        """
        return {
            "PRO": r"尿蛋白[：:]\s*([阴性阳性\-\+]+)",
            "GLU": r"尿糖[：:]\s*([阴性阳性\-\+]+)",
            "BLD": r"尿潜血[：:]\s*([阴性阳性\-\+]+)",
            "LEU": r"尿白细胞[：:]\s*([阴性阳性\-\+]+)",
            "KET": r"尿酮体[：:]\s*([阴性阳性\-\+]+)",
            "NIT": r"尿亚硝酸盐[：:]\s*([阴性阳性\-\+]+)",
            "BIL": r"尿胆红素[：:]\s*([阴性阳性\-\+]+)",
            "UBG": r"尿胆原[：:]\s*([阴性阳性\-\+]+)",
            "PH": r"尿酸碱度|尿pH[：:]\s*([\d.]+)",
            "SG": r"尿比重[：:]\s*([\d.]+)",
        }
    
    def normalize_qualitative_value(self, raw_value: str) -> str:
        """
        标准化定性指标的值。
        
        Args:
            raw_value: 原始值
        
        Returns:
            str: 标准化后的值
        """
        raw_value = raw_value.strip()
        
        if "阴性" in raw_value or raw_value == "-":
            return "阴性（-）"
        elif "阳性" in raw_value or "+" in raw_value:
            plus_count = raw_value.count("+")
            if plus_count == 1:
                return "阳性（+）"
            elif plus_count == 2:
                return "阳性（++）"
            elif plus_count == 3:
                return "阳性（+++）"
            elif plus_count >= 4:
                return "阳性（++++）"
        
        return raw_value
    
    def parse_urinalysis_report(self, report_text: str) -> Dict[str, str]:
        """
        解析尿常规报告文本，提取指标数值。
        
        Args:
            report_text: 尿常规报告文本
        
        Returns:
            Dict[str, str]: 指标名称到数值的映射
        
        Raises:
            ValueError: 如果报告文本为空
        """
        if not report_text or not report_text.strip():
            raise ValueError("报告文本不能为空")
        
        results = {}
        
        for indicator, pattern in self.indicator_patterns.items():
            match = re.search(pattern, report_text, re.IGNORECASE)
            if match:
                value = match.group(1)
                
                if indicator in ["PH", "SG"]:
                    try:
                        results[indicator] = float(value)
                        logger.debug(f"解析到指标 {indicator}: {value}")
                    except ValueError as e:
                        logger.warning(f"解析指标 {indicator} 失败: {e}")
                else:
                    normalized_value = self.normalize_qualitative_value(value)
                    results[indicator] = normalized_value
                    logger.debug(f"解析到指标 {indicator}: {normalized_value}")
        
        logger.info(f"成功解析 {len(results)} 个尿常规指标")
        return results
    
    def analyze_qualitative_indicator(
        self,
        indicator_name: str,
        value: str
    ) -> UrinalysisIndicator:
        """
        分析定性指标。
        
        Args:
            indicator_name: 指标名称
            value: 指标值
        
        Returns:
            UrinalysisIndicator: 指标分析结果
        """
        reference = self.db.urinalysis_references.get(indicator_name)
        
        if not reference:
            logger.warning(f"未找到指标 {indicator_name} 的参考范围")
            return UrinalysisIndicator(
                name=indicator_name,
                name_en=indicator_name,
                value=value,
                status="未知"
            )
        
        status, risk_level = self.db.evaluate_qualitative(value, reference)
        
        clinical_significance = ""
        if status != "正常":
            clinical_significance = reference.clinical_significance.get("阳性", "")
        
        return UrinalysisIndicator(
            name=reference.name,
            name_en=reference.name_en,
            value=value,
            status=status,
            risk_level=risk_level,
            reference=reference,
            clinical_significance=clinical_significance
        )
    
    def analyze_quantitative_indicator(
        self,
        indicator_name: str,
        value: float
    ) -> UrinalysisIndicator:
        """
        分析定量指标。
        
        Args:
            indicator_name: 指标名称
            value: 指标数值
        
        Returns:
            UrinalysisIndicator: 指标分析结果
        """
        reference = self.db.urinalysis_references.get(indicator_name)
        
        if not reference or not isinstance(reference, ReferenceRange):
            logger.warning(f"未找到指标 {indicator_name} 的参考范围")
            return UrinalysisIndicator(
                name=indicator_name,
                name_en=indicator_name,
                value=str(value),
                status="未知"
            )
        
        status, risk_level = self.db.evaluate_value(value, reference)
        
        clinical_significance = ""
        if status != "正常":
            if value < reference.normal_min:
                clinical_significance = reference.clinical_significance.get("降低", "")
            else:
                clinical_significance = reference.clinical_significance.get("升高", "")
        
        return UrinalysisIndicator(
            name=reference.name,
            name_en=reference.name_en,
            value=str(value),
            status=status,
            risk_level=risk_level,
            reference=reference,
            clinical_significance=clinical_significance
        )
    
    def generate_diagnosis_hints(
        self,
        indicators: List[UrinalysisIndicator]
    ) -> List[str]:
        """
        基于异常指标生成诊断提示。
        
        Args:
            indicators: 指标列表
        
        Returns:
            List[str]: 诊断提示列表
        """
        hints = []
        
        pro = next((i for i in indicators if i.name_en == "Urine Protein"), None)
        glu = next((i for i in indicators if i.name_en == "Urine Glucose"), None)
        bld = next((i for i in indicators if i.name_en == "Urine Blood"), None)
        leu = next((i for i in indicators if i.name_en == "Urine Leukocyte"), None)
        ket = next((i for i in indicators if i.name_en == "Urine Ketone"), None)
        
        if pro and pro.status != "正常":
            hints.append("尿蛋白阳性 → 提示肾病可能，建议肾内科就诊")
        
        if glu and glu.status != "正常":
            hints.append("尿糖阳性 → 提示糖代谢异常，建议内分泌科就诊")
        
        if bld and bld.status != "正常":
            hints.append("尿潜血阳性 → 提示泌尿系统问题，建议泌尿外科或肾内科就诊")
        
        if leu and leu.status != "正常":
            hints.append("尿白细胞阳性 → 提示泌尿系统感染可能")
        
        if ket and ket.status != "正常":
            hints.append("尿酮体阳性 → 提示糖尿病酮症酸中毒或饥饿状态")
        
        if pro and bld and leu:
            if pro.status != "正常" and bld.status != "正常" and leu.status != "正常":
                hints.append("尿蛋白 + 尿潜血 + 尿白细胞阳性 → 提示肾炎可能，建议肾内科就诊")
        
        return hints
    
    def generate_recommendations(
        self,
        indicators: List[UrinalysisIndicator]
    ) -> List[str]:
        """
        基于分析结果生成建议。
        
        Args:
            indicators: 指标列表
        
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        critical_indicators = [i for i in indicators if i.risk_level == RiskLevel.CRITICAL]
        if critical_indicators:
            recommendations.append("⚠️ 检测到危急值，建议立即就医！")
        
        high_risk_indicators = [i for i in indicators if i.risk_level == RiskLevel.HIGH]
        if high_risk_indicators:
            recommendations.append("检测到显著异常指标，建议尽快就医检查")
        
        pro = next((i for i in indicators if i.name_en == "Urine Protein"), None)
        if pro and pro.status != "正常":
            recommendations.append("尿蛋白异常建议：低盐低蛋白饮食，避免剧烈运动，定期复查尿常规")
        
        glu = next((i for i in indicators if i.name_en == "Urine Glucose"), None)
        if glu and glu.status != "正常":
            recommendations.append("尿糖异常建议：控制碳水化合物摄入，监测血糖，内分泌科就诊")
        
        bld = next((i for i in indicators if i.name_en == "Urine Blood"), None)
        if bld and bld.status != "正常":
            recommendations.append("尿潜血异常建议：多饮水，避免憋尿，泌尿外科就诊")
        
        if not recommendations:
            recommendations.append("尿常规各项指标正常，建议保持健康生活方式")
        
        return recommendations
    
    def analyze(self, report_text: str) -> UrinalysisAnalysisResult:
        """
        综合分析尿常规报告。
        
        Args:
            report_text: 尿常规报告文本
        
        Returns:
            UrinalysisAnalysisResult: 分析结果
        
        Raises:
            ValueError: 如果报告文本为空或无法解析
        """
        logger.info("开始分析尿常规报告")
        
        parsed_data = self.parse_urinalysis_report(report_text)
        
        if not parsed_data:
            raise ValueError("无法从报告文本中解析出任何尿常规指标")
        
        indicators = []
        for indicator_name, value in parsed_data.items():
            if indicator_name in ["PH", "SG"]:
                indicator = self.analyze_quantitative_indicator(indicator_name, value)
            else:
                indicator = self.analyze_qualitative_indicator(indicator_name, value)
            indicators.append(indicator)
        
        abnormal_count = sum(1 for i in indicators if i.status != "正常")
        
        risk_levels = [i.risk_level for i in indicators]
        if RiskLevel.CRITICAL in risk_levels:
            overall_risk = RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            overall_risk = RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        diagnosis_hints = self.generate_diagnosis_hints(indicators)
        recommendations = self.generate_recommendations(indicators)
        
        summary = self._generate_summary(indicators, abnormal_count, overall_risk)
        
        result = UrinalysisAnalysisResult(
            indicators=indicators,
            abnormal_count=abnormal_count,
            risk_level=overall_risk,
            diagnosis_hints=diagnosis_hints,
            recommendations=recommendations,
            summary=summary
        )
        
        logger.info(f"尿常规分析完成: {abnormal_count} 个异常指标，风险等级: {overall_risk.value}")
        return result
    
    def _generate_summary(
        self,
        indicators: List[UrinalysisIndicator],
        abnormal_count: int,
        risk_level: RiskLevel
    ) -> str:
        """
        生成分析摘要。
        
        Args:
            indicators: 指标列表
            abnormal_count: 异常指标数量
            risk_level: 风险等级
        
        Returns:
            str: 分析摘要
        """
        if abnormal_count == 0:
            return "尿常规各项指标均在正常范围内，未发现明显异常。"
        
        abnormal_indicators = [i for i in indicators if i.status != "正常"]
        abnormal_names = [i.name for i in abnormal_indicators]
        
        summary = f"尿常规检查发现 {abnormal_count} 项异常指标：{', '.join(abnormal_names)}。"
        summary += f"整体风险等级：{risk_level.value}。"
        
        if risk_level == RiskLevel.CRITICAL:
            summary += "请立即就医！"
        elif risk_level == RiskLevel.HIGH:
            summary += "建议尽快就医检查。"
        elif risk_level == RiskLevel.MEDIUM:
            summary += "建议定期复查，关注身体变化。"
        
        return summary


urinalysis_analyzer = UrinalysisAnalyzer()
