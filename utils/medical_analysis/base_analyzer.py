"""
医疗分析器抽象基类模块

功能：
- 定义医疗分析器的统一接口
- 提供通用的分析方法
- 支持多态调用
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .medical_reference import RiskLevel


class AnalysisType(Enum):
    """分析类型枚举"""

    CBC = "血常规"
    BIOCHEMISTRY = "血生化"
    URINALYSIS = "尿常规"
    VITAL_SIGNS = "生命体征"
    SYMPTOM = "症状分析"


@dataclass
class BaseAnalysisResult:
    """分析结果基类"""

    analysis_type: AnalysisType
    abnormal_count: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    diagnosis_hints: List[str] = field(default_factory=list)
    summary: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BaseMedicalAnalyzer(ABC):
    """
    医疗分析器抽象基类。

    所有医疗分析器（血常规、血生化、尿常规等）都应继承此类。

    Attributes:
        db: 医疗参考数据库
        indicator_patterns: 指标识别模式

    Example:
        >>> class CBCAnalyzer(BaseMedicalAnalyzer):
        ...     def analyze(self, report_text: str) -> BaseAnalysisResult:
        ...         # 实现具体分析逻辑
        ...         pass
    """

    def __init__(self):
        """初始化分析器。"""
        self.db = None
        self.indicator_patterns: Dict[str, str] = {}

    @abstractmethod
    def analyze(self, report_text: str, **kwargs) -> BaseAnalysisResult:
        """
        分析医疗报告文本。

        Args:
            report_text: 医疗报告文本
            **kwargs: 额外参数（如性别、年龄等）

        Returns:
            BaseAnalysisResult: 分析结果
        """
        pass

    @abstractmethod
    def parse_report(self, report_text: str, **kwargs) -> Dict[str, Any]:
        """
        解析医疗报告文本，提取指标数值。

        Args:
            report_text: 医疗报告文本
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 解析后的指标字典
        """
        pass

    @abstractmethod
    def get_analysis_type(self) -> AnalysisType:
        """
        获取分析类型。

        Returns:
            AnalysisType: 分析类型枚举值
        """
        pass

    def validate_input(self, report_text: str) -> bool:
        """
        验证输入是否有效。

        Args:
            report_text: 医疗报告文本

        Returns:
            bool: 输入是否有效
        """
        if not report_text or not isinstance(report_text, str):
            return False
        return len(report_text.strip()) > 0

    def calculate_risk_level(self, abnormal_count: int) -> RiskLevel:
        """
        根据异常指标数量计算风险等级。

        Args:
            abnormal_count: 异常指标数量

        Returns:
            RiskLevel: 风险等级
        """
        if abnormal_count == 0:
            return RiskLevel.LOW
        elif abnormal_count <= 2:
            return RiskLevel.MEDIUM
        elif abnormal_count <= 4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def generate_summary(
        self, analysis_type: AnalysisType, abnormal_count: int, risk_level: RiskLevel
    ) -> str:
        """
        生成分析摘要。

        Args:
            analysis_type: 分析类型
            abnormal_count: 异常指标数量
            risk_level: 风险等级

        Returns:
            str: 分析摘要
        """
        risk_desc = {
            RiskLevel.LOW: "低风险",
            RiskLevel.MEDIUM: "中等风险",
            RiskLevel.HIGH: "高风险",
            RiskLevel.CRITICAL: "危急",
        }

        if abnormal_count == 0:
            return f"{analysis_type.value}检查结果正常，无明显异常指标。"
        else:
            return (
                f"{analysis_type.value}检查发现 {abnormal_count} 项异常指标，"
                f"整体评估为{risk_desc[risk_level]}，建议进一步检查或咨询医生。"
            )
