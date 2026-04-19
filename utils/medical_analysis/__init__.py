"""
医疗分析模块

包含：
- 医疗指标参考数据库
- 各类检验报告分析器
- 分诊引擎
- 医疗工具封装
- 抽象基类（支持扩展）
"""

from .medical_reference import Gender, RiskLevel, MedicalReferenceDatabase
from .base_analyzer import (
    BaseMedicalAnalyzer,
    BaseAnalysisResult,
    AnalysisType,
)
from .cbc_analyzer import cbc_analyzer, CBCAnalyzer
from .biochemistry_analyzer import biochemistry_analyzer, BiochemistryAnalyzer
from .urinalysis_analyzer import urinalysis_analyzer, UrinalysisAnalyzer
from .vital_signs_analyzer import vital_signs_analyzer, VitalSignsAnalyzer
from .symptom_analyzer import symptom_analyzer, SymptomAnalyzer

from .medical_tools import (
    get_medical_tools,
    analyze_cbc_report,
    analyze_biochemistry_report,
    analyze_urinalysis_report,
    analyze_vital_signs,
    analyze_symptoms,
)

__all__ = [
    # 医疗参考
    "Gender",
    "RiskLevel",
    "MedicalReferenceDatabase",
    # 抽象基类
    "BaseMedicalAnalyzer",
    "BaseAnalysisResult",
    "AnalysisType",
    # 分析器
    "cbc_analyzer",
    "CBCAnalyzer",
    "biochemistry_analyzer",
    "BiochemistryAnalyzer",
    "urinalysis_analyzer",
    "UrinalysisAnalyzer",
    "vital_signs_analyzer",
    "VitalSignsAnalyzer",
    "symptom_analyzer",
    "SymptomAnalyzer",

    # 医疗工具
    "get_medical_tools",
    "analyze_cbc_report",
    "analyze_biochemistry_report",
    "analyze_urinalysis_report",
    "analyze_vital_signs",
    "analyze_symptoms",
]
