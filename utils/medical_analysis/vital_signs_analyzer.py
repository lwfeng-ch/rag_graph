"""
生命体征整合分析模块

功能：
- 接收体温、血压、心率等生命体征数据
- 建立生命体征与检验指标的关联分析
- 生成综合健康评估
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .medical_reference import (
    medical_db,
    RiskLevel,
    ReferenceRange
)

logger = logging.getLogger(__name__)


@dataclass
class VitalSignIndicator:
    """生命体征指标数据类"""
    name: str
    name_en: str
    value: float
    unit: str
    status: str = "正常"
    risk_level: RiskLevel = RiskLevel.LOW
    reference: Optional[ReferenceRange] = None
    clinical_significance: str = ""


@dataclass
class VitalSignsAnalysisResult:
    """生命体征分析结果数据类"""
    indicators: List[VitalSignIndicator] = field(default_factory=list)
    abnormal_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    diagnosis_hints: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    summary: str = ""


class VitalSignsAnalyzer:
    """生命体征分析器"""
    
    def __init__(self):
        self.db = medical_db
    
    def analyze_indicator(self, indicator_name: str, value: float) -> VitalSignIndicator:
        """
        分析单个生命体征指标。
        
        Args:
            indicator_name: 指标名称
            value: 指标数值
        
        Returns:
            VitalSignIndicator: 指标分析结果
        """
        reference = self.db.vital_signs_references.get(indicator_name)
        
        if not reference:
            logger.warning(f"未找到指标 {indicator_name} 的参考范围")
            return VitalSignIndicator(
                name=indicator_name,
                name_en=indicator_name,
                value=value,
                unit="",
                status="未知"
            )
        
        status, risk_level = self.db.evaluate_value(value, reference)
        
        clinical_significance = ""
        if status != "正常":
            if value < reference.normal_min:
                clinical_significance = reference.clinical_significance.get("降低", "")
            else:
                clinical_significance = reference.clinical_significance.get("升高", "")
        
        return VitalSignIndicator(
            name=reference.name,
            name_en=reference.name_en,
            value=value,
            unit=reference.unit,
            status=status,
            risk_level=risk_level,
            reference=reference,
            clinical_significance=clinical_significance
        )
    
    def analyze_vital_signs(
        self,
        temperature: Optional[float] = None,
        heart_rate: Optional[float] = None,
        systolic_bp: Optional[float] = None,
        diastolic_bp: Optional[float] = None,
        respiratory_rate: Optional[float] = None,
        spo2: Optional[float] = None
    ) -> VitalSignsAnalysisResult:
        """
        综合分析生命体征。
        
        Args:
            temperature: 体温（℃）
            heart_rate: 心率（次/分）
            systolic_bp: 收缩压（mmHg）
            diastolic_bp: 舒张压（mmHg）
            respiratory_rate: 呼吸频率（次/分）
            spo2: 血氧饱和度（%）
        
        Returns:
            VitalSignsAnalysisResult: 分析结果
        """
        logger.info("开始分析生命体征")
        
        indicators = []
        
        if temperature is not None:
            indicators.append(self.analyze_indicator("TEMP", temperature))
        
        if heart_rate is not None:
            indicators.append(self.analyze_indicator("HR", heart_rate))
        
        if systolic_bp is not None:
            indicators.append(self.analyze_indicator("SBP", systolic_bp))
        
        if diastolic_bp is not None:
            indicators.append(self.analyze_indicator("DBP", diastolic_bp))
        
        if respiratory_rate is not None:
            indicators.append(self.analyze_indicator("RR", respiratory_rate))
        
        if spo2 is not None:
            indicators.append(self.analyze_indicator("SPO2", spo2))
        
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
        
        result = VitalSignsAnalysisResult(
            indicators=indicators,
            abnormal_count=abnormal_count,
            risk_level=overall_risk,
            diagnosis_hints=diagnosis_hints,
            recommendations=recommendations,
            summary=summary
        )
        
        logger.info(f"生命体征分析完成: {abnormal_count} 个异常指标，风险等级: {overall_risk.value}")
        return result
    
    def generate_diagnosis_hints(
        self,
        indicators: List[VitalSignIndicator]
    ) -> List[str]:
        """
        基于异常指标生成诊断提示。
        
        Args:
            indicators: 指标列表
        
        Returns:
            List[str]: 诊断提示列表
        """
        hints = []
        
        temp = next((i for i in indicators if i.name_en == "Temperature"), None)
        hr = next((i for i in indicators if i.name_en == "Heart Rate"), None)
        sbp = next((i for i in indicators if i.name_en == "Systolic Blood Pressure"), None)
        dbp = next((i for i in indicators if i.name_en == "Diastolic Blood Pressure"), None)
        spo2 = next((i for i in indicators if i.name_en == "Blood Oxygen Saturation"), None)
        
        if temp and temp.status != "正常":
            if temp.value > 37.3:
                if temp.value >= 39.0:
                    hints.append("高热（≥39.0℃）→ 提示严重感染或炎症，建议立即就医")
                else:
                    hints.append("发热（>37.3℃）→ 提示可能存在感染，建议监测体温变化")
        
        if hr and hr.status != "正常":
            if hr.value > 100:
                hints.append("心动过速（>100次/分）→ 提示发热、贫血、甲亢或心功能不全")
            else:
                hints.append("心动过缓（<60次/分）→ 提示甲减、运动员状态或心脏传导阻滞")
        
        if sbp and dbp:
            if sbp.status != "正常" or dbp.status != "正常":
                if sbp.value >= 140 or dbp.value >= 90:
                    hints.append("高血压（≥140/90 mmHg）→ 提示心血管风险，建议心内科就诊")
                elif sbp.value < 90 or dbp.value < 60:
                    hints.append("低血压（<90/60 mmHg）→ 提示休克、脱水或心功能不全")
        
        if spo2 and spo2.status != "正常":
            if spo2.value < 90:
                hints.append("低血氧（<90%）→ 提示呼吸衰竭，建议立即就医")
            elif spo2.value < 95:
                hints.append("血氧偏低（<95%）→ 提示呼吸功能异常，建议监测")
        
        return hints
    
    def generate_recommendations(
        self,
        indicators: List[VitalSignIndicator]
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
        
        temp = next((i for i in indicators if i.name_en == "Temperature"), None)
        if temp and temp.status != "正常" and temp.value > 37.3:
            recommendations.append("发热建议：多饮水，注意休息，物理降温，如持续高热请就医")
        
        sbp = next((i for i in indicators if i.name_en == "Systolic Blood Pressure"), None)
        dbp = next((i for i in indicators if i.name_en == "Diastolic Blood Pressure"), None)
        if sbp and dbp and (sbp.status != "正常" or dbp.status != "正常"):
            if sbp.value >= 140 or dbp.value >= 90:
                recommendations.append("高血压建议：低盐饮食，规律运动，定期监测血压，心内科就诊")
        
        if not recommendations:
            recommendations.append("生命体征各项指标正常，建议保持健康生活方式")
        
        return recommendations
    
    def _generate_summary(
        self,
        indicators: List[VitalSignIndicator],
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
            return "生命体征各项指标均在正常范围内，未发现明显异常。"
        
        abnormal_indicators = [i for i in indicators if i.status != "正常"]
        abnormal_names = [i.name for i in abnormal_indicators]
        
        summary = f"生命体征检查发现 {abnormal_count} 项异常指标：{', '.join(abnormal_names)}。"
        summary += f"整体风险等级：{risk_level.value}。"
        
        if risk_level == RiskLevel.CRITICAL:
            summary += "请立即就医！"
        elif risk_level == RiskLevel.HIGH:
            summary += "建议尽快就医检查。"
        elif risk_level == RiskLevel.MEDIUM:
            summary += "建议定期复查，关注身体变化。"
        
        return summary


vital_signs_analyzer = VitalSignsAnalyzer()
