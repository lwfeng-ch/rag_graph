"""
医疗分析工具模块

功能：
- 将医疗报告分析功能封装为 LangChain Tool
- 提供结构化输入输出
- 支持 Agent 调用和状态传递

遵循 LangChain Tool 最佳实践：
- 使用 Pydantic BaseModel 定义参数
- 清晰的工具名称和描述
- 统一的返回值格式
- 完善的错误处理
"""

import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from enum import Enum

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .medical_reference import Gender, RiskLevel
from .cbc_analyzer import cbc_analyzer
from .biochemistry_analyzer import biochemistry_analyzer
from .urinalysis_analyzer import urinalysis_analyzer
from .vital_signs_analyzer import vital_signs_analyzer
from .symptom_analyzer import symptom_analyzer

logger = logging.getLogger(__name__)


def _gender_from_str(gender_str: str) -> Gender:
    """
    将字符串转换为 Gender 枚举。

    Args:
        gender_str: 性别字符串（male/female/unknown）

    Returns:
        Gender: 性别枚举值
    """
    gender_map = {
        "male": Gender.MALE,
        "female": Gender.FEMALE,
        "unknown": Gender.UNKNOWN,
        "男": Gender.MALE,
        "女": Gender.FEMALE,
    }
    return gender_map.get(gender_str.lower(), Gender.UNKNOWN)


def _convert_value(obj: Any) -> Any:
    """
    递归转换对象为 JSON 可序列化格式。

    处理以下类型：
    - dataclass: 递归转换为字典
    - Enum: 转换为其值
    - set: 转换为列表
    - list/dict: 递归处理元素
    - None: 保持不变

    Args:
        obj: 待转换的对象

    Returns:
        Any: JSON 可序列化的对象
    """
    if obj is None:
        return None
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, set):
        return [_convert_value(item) for item in obj]
    elif isinstance(obj, list):
        return [_convert_value(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _convert_value(v) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        return {k: _convert_value(v) for k, v in asdict(obj).items()}
    else:
        return obj


def _result_to_json(result: Any, analysis_type: str) -> str:
    """
    将分析结果转换为 JSON 字符串。

    Args:
        result: 分析结果对象（dataclass）
        analysis_type: 分析类型（cbc/biochemistry/urinalysis/vital_signs/symptom/triage）

    Returns:
        str: JSON 格式的结果字符串
    """
    try:
        result_dict = _convert_value(result)
        
        output = {
            "success": True,
            "analysis_type": analysis_type,
            **result_dict
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"转换结果为 JSON 失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": analysis_type
        }, ensure_ascii=False)


class CBCReportInput(BaseModel):
    """血常规报告分析输入参数"""
    report_text: str = Field(
        description="血常规检查报告文本，包含白细胞、血红蛋白、血小板等指标"
    )
    gender: str = Field(
        default="unknown",
        description="患者性别：male（男）、female（女）或 unknown（未知）"
    )


class BiochemistryReportInput(BaseModel):
    """血生化报告分析输入参数"""
    report_text: str = Field(
        description="血生化检查报告文本，包含血糖、肌酐、ALT/AST等指标"
    )
    gender: str = Field(
        default="unknown",
        description="患者性别：male（男）、female（女）或 unknown（未知）"
    )


class UrinalysisReportInput(BaseModel):
    """尿常规报告分析输入参数"""
    report_text: str = Field(
        description="尿常规检查报告文本，包含尿蛋白、尿糖、尿潜血等指标"
    )


class VitalSignsInput(BaseModel):
    """生命体征分析输入参数"""
    temperature: Optional[float] = Field(
        default=None,
        description="体温（摄氏度），正常范围 36.0-37.3"
    )
    heart_rate: Optional[float] = Field(
        default=None,
        description="心率（次/分），正常范围 60-100"
    )
    systolic_bp: Optional[float] = Field(
        default=None,
        description="收缩压（mmHg），正常范围 90-140"
    )
    diastolic_bp: Optional[float] = Field(
        default=None,
        description="舒张压（mmHg），正常范围 60-90"
    )
    respiratory_rate: Optional[float] = Field(
        default=None,
        description="呼吸频率（次/分），正常范围 12-20"
    )
    spo2: Optional[float] = Field(
        default=None,
        description="血氧饱和度（%），正常范围 95-100"
    )


class SymptomInput(BaseModel):
    """症状分析输入参数"""
    symptom_text: str = Field(
        description="患者症状描述文本，如'我最近发烧、咳嗽、头晕'"
    )


@tool("analyze_cbc_report", args_schema=CBCReportInput)
def analyze_cbc_report(report_text: str, gender: str = "unknown") -> str:
    """
    分析血常规（CBC）报告，识别异常指标并评估风险等级。
    
    适用场景：用户提供了血常规检查报告，需要评估感染、贫血、血液系统疾病风险。
    
    Returns:
        JSON 格式结果，包含：异常指标列表、诊断提示、风险等级（low/medium/high/critical）
    """
    logger.info(f"开始分析血常规报告，性别: {gender}")
    
    try:
        gender_enum = _gender_from_str(gender)
        result = cbc_analyzer.analyze(report_text, gender_enum)
        return _result_to_json(result, "cbc")
    except Exception as e:
        logger.error(f"血常规分析失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": "cbc"
        }, ensure_ascii=False)


@tool("analyze_biochemistry_report", args_schema=BiochemistryReportInput)
def analyze_biochemistry_report(report_text: str, gender: str = "unknown") -> str:
    """
    分析血生化报告，评估糖尿病、肾功能、肝功能风险。
    
    适用场景：用户提供了血生化检查报告，需要评估糖尿病、肾病、肝病风险。
    
    Returns:
        JSON 格式结果，包含：异常指标、关联性分析、风险等级
    """
    logger.info(f"开始分析血生化报告，性别: {gender}")
    
    try:
        gender_enum = _gender_from_str(gender)
        result = biochemistry_analyzer.analyze(report_text, gender_enum)
        return _result_to_json(result, "biochemistry")
    except Exception as e:
        logger.error(f"血生化分析失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": "biochemistry"
        }, ensure_ascii=False)


@tool("analyze_urinalysis_report", args_schema=UrinalysisReportInput)
def analyze_urinalysis_report(report_text: str) -> str:
    """
    分析尿常规报告，评估肾病、泌尿系统疾病风险。
    
    适用场景：用户提供了尿常规检查报告，需要评估肾病、泌尿系统感染风险。
    
    Returns:
        JSON 格式结果，包含：异常指标、临床意义、风险等级
    """
    logger.info("开始分析尿常规报告")
    
    try:
        result = urinalysis_analyzer.analyze(report_text)
        return _result_to_json(result, "urinalysis")
    except Exception as e:
        logger.error(f"尿常规分析失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": "urinalysis"
        }, ensure_ascii=False)


@tool("analyze_vital_signs", args_schema=VitalSignsInput)
def analyze_vital_signs(
    temperature: Optional[float] = None,
    heart_rate: Optional[float] = None,
    systolic_bp: Optional[float] = None,
    diastolic_bp: Optional[float] = None,
    respiratory_rate: Optional[float] = None,
    spo2: Optional[float] = None
) -> str:
    """
    分析生命体征数据，评估发热、高血压、低血氧等风险。
    
    适用场景：用户提供了体温、血压等生命体征数据，需要评估各项指标是否正常。
    
    Returns:
        JSON 格式结果，包含：异常指标、诊断提示、风险等级
    """
    logger.info("开始分析生命体征")
    
    try:
        result = vital_signs_analyzer.analyze_vital_signs(
            temperature=temperature,
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            respiratory_rate=respiratory_rate,
            spo2=spo2
        )
        return _result_to_json(result, "vital_signs")
    except Exception as e:
        logger.error(f"生命体征分析失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": "vital_signs"
        }, ensure_ascii=False)


@tool("analyze_symptoms", args_schema=SymptomInput)
def analyze_symptoms(symptom_text: str) -> str:
    """
    分析症状文本，识别关键症状并评估紧急程度。
    
    适用场景：用户描述了自己的症状，需要提取关键症状并评估紧急程度。
    
    Returns:
        JSON 格式结果，包含：检测到的症状、紧急程度、相关指标
    """
    logger.info("开始分析症状文本")
    
    try:
        result = symptom_analyzer.analyze_symptoms(symptom_text)
        return _result_to_json(result, "symptom")
    except Exception as e:
        logger.error(f"症状分析失败: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": "symptom"
        }, ensure_ascii=False)


def get_medical_tools() -> List:
    """
    获取所有医疗分析工具列表。

    Returns:
        List: 医疗工具列表
    """
    return [
        analyze_cbc_report,
        analyze_biochemistry_report,
        analyze_urinalysis_report,
        analyze_vital_signs,
        analyze_symptoms,
    ]
