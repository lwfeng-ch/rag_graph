"""
医疗工具兼容层

为了保持向后兼容性，此文件重导出 medical_analysis 模块的公共接口。
推荐使用：from utils.medical_analysis import ...
"""

from utils.medical_analysis import (
    # 医疗参考
    Gender,
    RiskLevel,
    MedicalReferenceDatabase,
    # 分析器
    cbc_analyzer,
    CBCAnalyzer,
    biochemistry_analyzer,
    BiochemistryAnalyzer,
    urinalysis_analyzer,
    UrinalysisAnalyzer,
    vital_signs_analyzer,
    VitalSignsAnalyzer,
    symptom_analyzer,
    SymptomAnalyzer,
    # 医疗工具
    get_medical_tools,
    analyze_cbc_report,
    analyze_biochemistry_report,
    analyze_urinalysis_report,
    analyze_vital_signs,
    analyze_symptoms,
)

__all__ = [
    "Gender",
    "RiskLevel",
    "MedicalReferenceDatabase",
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
    "get_medical_tools",
    "analyze_cbc_report",
    "analyze_biochemistry_report",
    "analyze_urinalysis_report",
    "analyze_vital_signs",
    "analyze_symptoms",
]
