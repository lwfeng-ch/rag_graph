"""
血常规（CBC）分析模块

功能：
- 解析血常规报告文本
- 识别异常指标（↑、↓标记）
- 建立初步诊断逻辑
- 生成结构化分析报告
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .medical_reference import medical_db, Gender, RiskLevel, ReferenceRange

logger = logging.getLogger(__name__)


@dataclass
class CBCIndicator:
    """血常规指标数据类"""

    name: str
    name_en: str
    value: float
    unit: str
    status: str = "正常"
    risk_level: RiskLevel = RiskLevel.LOW
    reference: Optional[ReferenceRange] = None
    clinical_significance: str = ""


@dataclass
class CBCAnalysisResult:
    """血常规分析结果数据类"""

    indicators: List[CBCIndicator] = field(default_factory=list)
    abnormal_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    diagnosis_hints: List[str] = field(default_factory=list)
    summary: str = ""


class CBCAnalyzer:
    """血常规分析器"""

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
            "WBC": r"白细胞[计数]*[：:]\s*([\d.]+)\s*[×xX]?\s*10[⁹9]/L",
            "HGB": r"血红蛋白[：:]\s*([\d.]+)\s*g/L",
            "PLT": r"血小板[计数]*[：:]\s*([\d.]+)\s*[×xX]?\s*10[⁹9]/L",
            "RBC": r"红细胞[计数]*[：:]\s*([\d.]+)\s*[×xX]?\s*10[¹¹12]/L",
            "HCT": r"红细胞压积[：:]\s*([\d.]+)\s*L/L",
            "MCV": r"平均红细胞体积[：:]\s*([\d.]+)\s*fL",
            "MCH": r"平均血红蛋白量[：:]\s*([\d.]+)\s*pg",
            "MCHC": r"平均血红蛋白浓度[：:]\s*([\d.]+)\s*g/L",
            "NEUT%": r"中性粒细胞[百分比]*[：:]\s*([\d.]+)\s*%",
            "LYMPH%": r"淋巴细胞[百分比]*[：:]\s*([\d.]+)\s*%",
        }

    def parse_cbc_report(
        self, report_text: str, gender: Gender = Gender.UNKNOWN
    ) -> Dict[str, float]:
        """
        解析血常规报告文本，提取指标数值。

        Args:
            report_text: 血常规报告文本
            gender: 性别（用于性别相关指标）

        Returns:
            Dict[str, float]: 指标名称到数值的映射

        Raises:
            ValueError: 如果报告文本为空
        """
        if not report_text or not report_text.strip():
            raise ValueError("报告文本不能为空")

        results = {}

        for indicator, pattern in self.indicator_patterns.items():
            match = re.search(pattern, report_text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    results[indicator] = value
                    logger.debug(f"解析到指标 {indicator}: {value}")
                except ValueError as e:
                    logger.warning(f"解析指标 {indicator} 失败: {e}")

        logger.info(f"成功解析 {len(results)} 个血常规指标")
        return results

    def detect_abnormal_markers(self, report_text: str) -> Dict[str, str]:
        """
        检测报告中的异常标记（↑、↓）。

        Args:
            report_text: 血常规报告文本

        Returns:
            Dict[str, str]: 指标名称到异常标记的映射
        """
        abnormal_markers = {}

        patterns = {
            "WBC": r"白细胞[计数]*[：:]\s*([\d.]+)\s*[×xX]?\s*10[⁹9]/L\s*([↑↓])",
            "HGB": r"血红蛋白[：:]\s*([\d.]+)\s*g/L\s*([↑↓])",
            "PLT": r"血小板[计数]*[：:]\s*([\d.]+)\s*[×xX]?\s*10[⁹9]/L\s*([↑↓])",
        }

        for indicator, pattern in patterns.items():
            match = re.search(pattern, report_text)
            if match:
                marker = match.group(2)
                abnormal_markers[indicator] = marker
                logger.debug(f"检测到异常标记 {indicator}: {marker}")

        return abnormal_markers

    def analyze_indicator(
        self, indicator_name: str, value: float, gender: Gender = Gender.UNKNOWN
    ) -> CBCIndicator:
        """
        分析单个血常规指标。

        Args:
            indicator_name: 指标名称
            value: 指标数值
            gender: 性别

        Returns:
            CBCIndicator: 指标分析结果
        """
        reference = self.db.get_reference("cbc", indicator_name, gender)

        if not reference:
            logger.warning(f"未找到指标 {indicator_name} 的参考范围")
            return CBCIndicator(
                name=indicator_name,
                name_en=indicator_name,
                value=value,
                unit="",
                status="未知",
            )

        status, risk_level = self.db.evaluate_value(value, reference)

        clinical_significance = ""
        if status != "正常":
            if value < reference.normal_min:
                clinical_significance = reference.clinical_significance.get("降低", "")
            else:
                clinical_significance = reference.clinical_significance.get("升高", "")

        return CBCIndicator(
            name=reference.name,
            name_en=reference.name_en,
            value=value,
            unit=reference.unit,
            status=status,
            risk_level=risk_level,
            reference=reference,
            clinical_significance=clinical_significance,
        )

    def generate_diagnosis_hints(self, indicators: List[CBCIndicator]) -> List[str]:
        """
        基于异常指标生成诊断提示。

        Args:
            indicators: 指标列表

        Returns:
            List[str]: 诊断提示列表
        """
        hints = []

        wbc = next(
            (i for i in indicators if i.name_en == "White Blood Cell Count"), None
        )
        hgb = next((i for i in indicators if "Hemoglobin" in i.name_en), None)
        plt = next((i for i in indicators if i.name_en == "Platelet Count"), None)

        if wbc and wbc.status != "正常":
            if wbc.value > wbc.reference.normal_max:
                hints.append("白细胞升高 → 提示可能存在感染、炎症或应激状态")
            else:
                hints.append("白细胞降低 → 提示可能存在病毒感染或骨髓抑制")

        if hgb and hgb.status != "正常":
            if hgb.value < hgb.reference.normal_min:
                hints.append("血红蛋白降低 → 提示可能存在贫血，建议进一步检查贫血类型")
            else:
                hints.append("血红蛋白升高 → 提示可能存在脱水或慢性缺氧")

        if plt and plt.status != "正常":
            if plt.value < plt.reference.normal_min:
                hints.append("血小板降低 → 提示可能存在出血风险，建议避免外伤")
            else:
                hints.append("血小板升高 → 提示可能存在炎症或骨髓增殖性疾病")

        if wbc and hgb and plt:
            if wbc.status != "正常" and hgb.status != "正常" and plt.status != "正常":
                hints.append("多系血细胞异常 → 建议血液科就诊，排除血液系统疾病")

        return hints

    def generate_recommendations(self, indicators: List[CBCIndicator]) -> List[str]:
        """
        基于分析结果生成建议。

        Args:
            indicators: 指标列表

        Returns:
            List[str]: 建议列表
        """
        recommendations = []

        critical_indicators = [
            i for i in indicators if i.risk_level == RiskLevel.CRITICAL
        ]
        if critical_indicators:
            recommendations.append("⚠️ 检测到危急值，建议立即就医！")

        high_risk_indicators = [i for i in indicators if i.risk_level == RiskLevel.HIGH]
        if high_risk_indicators:
            recommendations.append("检测到显著异常指标，建议尽快就医检查")

        medium_risk_indicators = [
            i for i in indicators if i.risk_level == RiskLevel.MEDIUM
        ]
        if medium_risk_indicators:
            recommendations.append("部分指标轻度异常，建议定期复查")

        hgb = next((i for i in indicators if "Hemoglobin" in i.name_en), None)
        if hgb and hgb.status != "正常" and hgb.value < hgb.reference.normal_min:
            recommendations.append("贫血患者建议：增加红肉、动物肝脏、菠菜等富铁食物")

        wbc = next(
            (i for i in indicators if i.name_en == "White Blood Cell Count"), None
        )
        if wbc and wbc.status != "正常":
            if wbc.value > wbc.reference.normal_max:
                recommendations.append(
                    "白细胞升高建议：注意休息，多饮水，如有发热症状及时就医"
                )

        if not recommendations:
            recommendations.append("血常规各项指标正常，建议保持健康生活方式")

        return recommendations

    def analyze(
        self, report_text: str, gender: Gender = Gender.UNKNOWN
    ) -> CBCAnalysisResult:
        """
        综合分析血常规报告。

        Args:
            report_text: 血常规报告文本
            gender: 性别

        Returns:
            CBCAnalysisResult: 分析结果

        Raises:
            ValueError: 如果报告文本为空或无法解析
        """
        logger.info("开始分析血常规报告")

        parsed_data = self.parse_cbc_report(report_text, gender)

        if not parsed_data:
            raise ValueError("无法从报告文本中解析出任何血常规指标")

        indicators = []
        for indicator_name, value in parsed_data.items():
            indicator = self.analyze_indicator(indicator_name, value, gender)
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

        summary = self._generate_summary(indicators, abnormal_count, overall_risk)

        result = CBCAnalysisResult(
            indicators=indicators,
            abnormal_count=abnormal_count,
            risk_level=overall_risk,
            diagnosis_hints=diagnosis_hints,
            summary=summary,
        )

        logger.info(
            f"血常规分析完成: {abnormal_count} 个异常指标，风险等级: {overall_risk.value}"
        )
        return result

    def _generate_summary(
        self, indicators: List[CBCIndicator], abnormal_count: int, risk_level: RiskLevel
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
            return "血常规各项指标均在正常范围内，未发现明显异常。"

        abnormal_indicators = [i for i in indicators if i.status != "正常"]
        abnormal_names = [i.name for i in abnormal_indicators]

        summary = (
            f"血常规检查发现 {abnormal_count} 项异常指标：{', '.join(abnormal_names)}。"
        )
        summary += f"整体风险等级：{risk_level.value}。"

        if risk_level == RiskLevel.CRITICAL:
            summary += "请立即就医！"
        elif risk_level == RiskLevel.HIGH:
            summary += "建议尽快就医检查。"
        elif risk_level == RiskLevel.MEDIUM:
            summary += "建议定期复查，关注身体变化。"

        return summary


cbc_analyzer = CBCAnalyzer()
