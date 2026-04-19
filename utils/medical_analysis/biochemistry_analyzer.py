"""
血生化分析模块

功能：
- 解析血生化报告文本
- 聚焦血糖、肌酐、ALT/AST 关键指标
- 建立指标间关联性分析
- 生成结构化分析报告
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .medical_reference import medical_db, Gender, RiskLevel, ReferenceRange

logger = logging.getLogger(__name__)


@dataclass
class BiochemistryIndicator:
    """血生化指标数据类"""

    name: str
    name_en: str
    value: float
    unit: str
    status: str = "正常"
    risk_level: RiskLevel = RiskLevel.LOW
    reference: Optional[ReferenceRange] = None
    clinical_significance: str = ""


@dataclass
class BiochemistryAnalysisResult:
    """血生化分析结果数据类"""

    indicators: List[BiochemistryIndicator] = field(default_factory=list)
    abnormal_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    diagnosis_hints: List[str] = field(default_factory=list)
    correlation_analysis: List[str] = field(default_factory=list)
    summary: str = ""


class BiochemistryAnalyzer:
    """血生化分析器"""

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
            "GLU_fasting": r"空腹血糖[：:]\s*([\d.]+)\s*mmol/L",
            "GLU_postprandial": r"餐后.*?血糖[：:]\s*([\d.]+)\s*mmol/L",
            "HbA1c": r"糖化血红蛋白[：:]\s*([\d.]+)\s*%",
            "Cr": r"肌酐[：:]\s*([\d.]+)\s*μmol/L",
            "BUN": r"尿素氮[：:]\s*([\d.]+)\s*mmol/L",
            "ALT": r"谷丙转氨酶|ALT[：:]\s*([\d.]+)\s*U/L",
            "AST": r"谷草转氨酶|AST[：:]\s*([\d.]+)\s*U/L",
            "TBIL": r"总胆红素[：:]\s*([\d.]+)\s*μmol/L",
            "DBIL": r"直接胆红素[：:]\s*([\d.]+)\s*μmol/L",
            "ALB": r"白蛋白[：:]\s*([\d.]+)\s*g/L",
        }

    def parse_biochemistry_report(
        self, report_text: str, gender: Gender = Gender.UNKNOWN
    ) -> Dict[str, float]:
        """
        解析血生化报告文本，提取指标数值。

        Args:
            report_text: 血生化报告文本
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

        logger.info(f"成功解析 {len(results)} 个血生化指标")
        return results

    def analyze_indicator(
        self, indicator_name: str, value: float, gender: Gender = Gender.UNKNOWN
    ) -> BiochemistryIndicator:
        """
        分析单个血生化指标。

        Args:
            indicator_name: 指标名称
            value: 指标数值
            gender: 性别

        Returns:
            BiochemistryIndicator: 指标分析结果
        """
        reference = self.db.get_reference("biochemistry", indicator_name, gender)

        if not reference:
            logger.warning(f"未找到指标 {indicator_name} 的参考范围")
            return BiochemistryIndicator(
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

        return BiochemistryIndicator(
            name=reference.name,
            name_en=reference.name_en,
            value=value,
            unit=reference.unit,
            status=status,
            risk_level=risk_level,
            reference=reference,
            clinical_significance=clinical_significance,
        )

    def analyze_correlations(
        self, indicators: List[BiochemistryIndicator]
    ) -> List[str]:
        """
        分析指标间的关联性。

        Args:
            indicators: 指标列表

        Returns:
            List[str]: 关联性分析结果列表
        """
        correlations = []

        glu = next((i for i in indicators if "Glucose" in i.name_en), None)
        cr = next((i for i in indicators if "Creatinine" in i.name_en), None)
        alt = next(
            (i for i in indicators if "ALT" in i.name_en or "谷丙转氨酶" in i.name),
            None,
        )
        ast = next(
            (i for i in indicators if "AST" in i.name_en or "谷草转氨酶" in i.name),
            None,
        )

        if glu and cr:
            if glu.status != "正常" and cr.status != "正常":
                if (
                    glu.value > glu.reference.normal_max
                    and cr.value > cr.reference.normal_max
                ):
                    correlations.append(
                        "血糖升高 + 肌酐升高 → 提示糖尿病肾病可能，建议内分泌科和肾内科就诊"
                    )

        if alt and ast:
            if alt.status != "正常" or ast.status != "正常":
                if (
                    alt.value > alt.reference.normal_max
                    and ast.value > ast.reference.normal_max
                ):
                    ast_alt_ratio = ast.value / alt.value if alt.value > 0 else 0
                    if ast_alt_ratio > 2:
                        correlations.append(
                            "AST/ALT > 2 → 提示可能存在酒精性肝病或肝硬化"
                        )
                    elif ast_alt_ratio < 1:
                        correlations.append("ALT > AST → 提示可能存在急性肝炎或脂肪肝")

        if glu and alt:
            if glu.status != "正常" and alt.status != "正常":
                if (
                    glu.value > glu.reference.normal_max
                    and alt.value > alt.reference.normal_max
                ):
                    correlations.append(
                        "血糖升高 + 肝功能异常 → 提示代谢综合征可能，建议综合评估"
                    )

        return correlations

    def generate_diagnosis_hints(
        self, indicators: List[BiochemistryIndicator]
    ) -> List[str]:
        """
        基于异常指标生成诊断提示。

        Args:
            indicators: 指标列表

        Returns:
            List[str]: 诊断提示列表
        """
        hints = []

        glu = next((i for i in indicators if "Glucose" in i.name_en), None)
        cr = next((i for i in indicators if "Creatinine" in i.name_en), None)
        alt = next(
            (i for i in indicators if "ALT" in i.name_en or "谷丙转氨酶" in i.name),
            None,
        )
        ast = next(
            (i for i in indicators if "AST" in i.name_en or "谷草转氨酶" in i.name),
            None,
        )
        hba1c = next(
            (i for i in indicators if "Glycated Hemoglobin" in i.name_en), None
        )

        if glu and glu.status != "正常":
            if glu.value > glu.reference.normal_max:
                if glu.value >= 7.0:
                    hints.append(
                        "空腹血糖 ≥ 7.0 mmol/L → 提示糖尿病可能，建议内分泌科就诊"
                    )
                elif glu.value >= 6.1:
                    hints.append(
                        "空腹血糖 6.1-7.0 mmol/L → 提示糖尿病前期，建议控制饮食、加强运动"
                    )

        if hba1c and hba1c.status != "正常":
            if hba1c.value > 7.0:
                hints.append(
                    "糖化血红蛋白 > 7% → 提示近期血糖控制不佳，建议调整治疗方案"
                )

        if cr and cr.status != "正常":
            if cr.value > cr.reference.normal_max:
                hints.append("肌酐升高 → 提示肾功能异常，建议肾内科就诊")

        if alt and alt.status != "正常":
            if alt.value > alt.reference.normal_max:
                hints.append("ALT 升高 → 提示肝损伤可能，建议消化内科或肝病科就诊")

        if ast and ast.status != "正常":
            if ast.value > ast.reference.normal_max:
                hints.append("AST 升高 → 提示肝损伤或心肌损伤可能")

        return hints

    def generate_recommendations(
        self, indicators: List[BiochemistryIndicator]
    ) -> List[str]:
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

        glu = next((i for i in indicators if "Glucose" in i.name_en), None)
        if glu and glu.status != "正常" and glu.value > glu.reference.normal_max:
            recommendations.append(
                "血糖异常建议：控制碳水化合物摄入，增加运动，定期监测血糖"
            )

        cr = next((i for i in indicators if "Creatinine" in i.name_en), None)
        if cr and cr.status != "正常" and cr.value > cr.reference.normal_max:
            recommendations.append(
                "肾功能异常建议：低盐低蛋白饮食，避免使用肾毒性药物，定期复查肾功能"
            )

        alt = next(
            (i for i in indicators if "ALT" in i.name_en or "谷丙转氨酶" in i.name),
            None,
        )
        if alt and alt.status != "正常" and alt.value > alt.reference.normal_max:
            recommendations.append(
                "肝功能异常建议：戒酒，避免高脂饮食，慎用肝毒性药物，定期复查肝功能"
            )

        if not recommendations:
            recommendations.append("血生化各项指标正常，建议保持健康生活方式")

        return recommendations

    def analyze(
        self, report_text: str, gender: Gender = Gender.UNKNOWN
    ) -> BiochemistryAnalysisResult:
        """
        综合分析血生化报告。

        Args:
            report_text: 血生化报告文本
            gender: 性别

        Returns:
            BiochemistryAnalysisResult: 分析结果

        Raises:
            ValueError: 如果报告文本为空或无法解析
        """
        logger.info("开始分析血生化报告")

        parsed_data = self.parse_biochemistry_report(report_text, gender)

        if not parsed_data:
            raise ValueError("无法从报告文本中解析出任何血生化指标")

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
        correlation_analysis = self.analyze_correlations(indicators)

        summary = self._generate_summary(indicators, abnormal_count, overall_risk)

        result = BiochemistryAnalysisResult(
            indicators=indicators,
            abnormal_count=abnormal_count,
            risk_level=overall_risk,
            diagnosis_hints=diagnosis_hints,
            correlation_analysis=correlation_analysis,
            summary=summary,
        )

        logger.info(
            f"血生化分析完成: {abnormal_count} 个异常指标，风险等级: {overall_risk.value}"
        )
        return result

    def _generate_summary(
        self,
        indicators: List[BiochemistryIndicator],
        abnormal_count: int,
        risk_level: RiskLevel,
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
            return "血生化各项指标均在正常范围内，未发现明显异常。"

        abnormal_indicators = [i for i in indicators if i.status != "正常"]
        abnormal_names = [i.name for i in abnormal_indicators]

        summary = (
            f"血生化检查发现 {abnormal_count} 项异常指标：{', '.join(abnormal_names)}。"
        )
        summary += f"整体风险等级：{risk_level.value}。"

        if risk_level == RiskLevel.CRITICAL:
            summary += "请立即就医！"
        elif risk_level == RiskLevel.HIGH:
            summary += "建议尽快就医检查。"
        elif risk_level == RiskLevel.MEDIUM:
            summary += "建议定期复查，关注身体变化。"

        return summary


biochemistry_analyzer = BiochemistryAnalyzer()
