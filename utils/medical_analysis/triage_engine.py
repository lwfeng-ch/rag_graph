"""
分诊逻辑模块

功能：
- 基于血常规（80%权重）、血生化、尿常规、生命体征和症状文本构建综合分诊模型
- 实现风险等级评估功能
- 生成结构化的分析建议报告
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .medical_reference import Gender, RiskLevel
from .cbc_analyzer import cbc_analyzer, CBCAnalysisResult
from .biochemistry_analyzer import biochemistry_analyzer, BiochemistryAnalysisResult
from .urinalysis_analyzer import urinalysis_analyzer, UrinalysisAnalysisResult
from .vital_signs_analyzer import vital_signs_analyzer, VitalSignsAnalysisResult
from .symptom_analyzer import symptom_analyzer, SymptomAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """分诊结果数据类"""
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    cbc_result: Optional[CBCAnalysisResult] = None
    biochemistry_result: Optional[BiochemistryAnalysisResult] = None
    urinalysis_result: Optional[UrinalysisAnalysisResult] = None
    vital_signs_result: Optional[VitalSignsAnalysisResult] = None
    symptom_result: Optional[SymptomAnalysisResult] = None
    department_recommendations: List[str] = field(default_factory=list)
    urgent_actions: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    comprehensive_summary: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class TriageEngine:
    """分诊引擎"""
    
    def __init__(self):
        self.weights = {
            "cbc": 0.40,
            "biochemistry": 0.25,
            "urinalysis": 0.15,
            "vital_signs": 0.15,
            "symptoms": 0.05
        }
    
    def calculate_risk_score(
        self,
        cbc_result: Optional[CBCAnalysisResult] = None,
        biochemistry_result: Optional[BiochemistryAnalysisResult] = None,
        urinalysis_result: Optional[UrinalysisAnalysisResult] = None,
        vital_signs_result: Optional[VitalSignsAnalysisResult] = None,
        symptom_result: Optional[SymptomAnalysisResult] = None
    ) -> float:
        """
        计算综合风险评分（0-100分）。
        
        Args:
            cbc_result: 血常规分析结果
            biochemistry_result: 血生化分析结果
            urinalysis_result: 尿常规分析结果
            vital_signs_result: 生命体征分析结果
            symptom_result: 症状分析结果
        
        Returns:
            float: 综合风险评分
        """
        risk_score = 0.0
        
        if cbc_result:
            cbc_score = self._risk_level_to_score(cbc_result.risk_level)
            risk_score += cbc_score * self.weights["cbc"]
        
        if biochemistry_result:
            biochemistry_score = self._risk_level_to_score(biochemistry_result.risk_level)
            risk_score += biochemistry_score * self.weights["biochemistry"]
        
        if urinalysis_result:
            urinalysis_score = self._risk_level_to_score(urinalysis_result.risk_level)
            risk_score += urinalysis_score * self.weights["urinalysis"]
        
        if vital_signs_result:
            vital_signs_score = self._risk_level_to_score(vital_signs_result.risk_level)
            risk_score += vital_signs_score * self.weights["vital_signs"]
        
        if symptom_result:
            symptom_score = self._urgency_to_score(symptom_result.urgency_level)
            risk_score += symptom_score * self.weights["symptoms"]
        
        return min(risk_score, 100.0)
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """
        将风险等级转换为评分。
        
        Args:
            risk_level: 风险等级
        
        Returns:
            float: 评分（0-100）
        """
        score_map = {
            RiskLevel.LOW: 10.0,
            RiskLevel.MEDIUM: 50.0,
            RiskLevel.HIGH: 80.0,
            RiskLevel.CRITICAL: 100.0
        }
        return score_map.get(risk_level, 10.0)
    
    def _urgency_to_score(self, urgency_level: str) -> float:
        """
        将紧急程度转换为评分。
        
        Args:
            urgency_level: 紧急程度
        
        Returns:
            float: 评分（0-100）
        """
        score_map = {
            "低": 10.0,
            "中": 50.0,
            "高": 100.0
        }
        return score_map.get(urgency_level, 10.0)
    
    def determine_overall_risk(self, risk_score: float) -> RiskLevel:
        """
        根据风险评分确定整体风险等级。
        
        Args:
            risk_score: 风险评分
        
        Returns:
            RiskLevel: 整体风险等级
        """
        if risk_score >= 80:
            return RiskLevel.CRITICAL
        elif risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def generate_department_recommendations(
        self,
        cbc_result: Optional[CBCAnalysisResult] = None,
        biochemistry_result: Optional[BiochemistryAnalysisResult] = None,
        urinalysis_result: Optional[UrinalysisAnalysisResult] = None,
        vital_signs_result: Optional[VitalSignsAnalysisResult] = None,
        symptom_result: Optional[SymptomAnalysisResult] = None
    ) -> List[str]:
        """
        生成科室推荐。
        
        Args:
            cbc_result: 血常规分析结果
            biochemistry_result: 血生化分析结果
            urinalysis_result: 尿常规分析结果
            vital_signs_result: 生命体征分析结果
            symptom_result: 症状分析结果
        
        Returns:
            List[str]: 科室推荐列表
        """
        departments = set()
        
        if cbc_result:
            for indicator in cbc_result.indicators:
                if indicator.status != "正常":
                    if "Hemoglobin" in indicator.name_en:
                        departments.add("血液科")
                    if "White Blood Cell" in indicator.name_en:
                        departments.add("感染科")
        
        if biochemistry_result:
            for indicator in biochemistry_result.indicators:
                if indicator.status != "正常":
                    if "Glucose" in indicator.name_en:
                        departments.add("内分泌科")
                    if "Creatinine" in indicator.name_en:
                        departments.add("肾内科")
                    if "ALT" in indicator.name_en or "AST" in indicator.name_en:
                        departments.add("消化内科")
        
        if urinalysis_result:
            for indicator in urinalysis_result.indicators:
                if indicator.status != "正常":
                    if indicator.name_en in ["Urine Protein", "Urine Blood"]:
                        departments.add("肾内科")
                    if indicator.name_en in ["Urine Leukocyte", "Urine Nitrite"]:
                        departments.add("泌尿外科")
        
        if vital_signs_result:
            for indicator in vital_signs_result.indicators:
                if indicator.status != "正常":
                    if indicator.name_en in ["Systolic Blood Pressure", "Diastolic Blood Pressure"]:
                        departments.add("心内科")
                    if indicator.name_en == "Blood Oxygen Saturation":
                        departments.add("呼吸内科")
        
        if symptom_result:
            for symptom in symptom_result.detected_symptoms:
                if symptom.category.value == "心血管系统":
                    departments.add("心内科")
                elif symptom.category.value == "呼吸系统":
                    departments.add("呼吸内科")
                elif symptom.category.value == "消化系统":
                    departments.add("消化内科")
                elif symptom.category.value == "泌尿系统":
                    departments.add("肾内科")
        
        return list(departments) if departments else ["全科门诊"]
    
    def generate_urgent_actions(
        self,
        overall_risk: RiskLevel,
        cbc_result: Optional[CBCAnalysisResult] = None,
        biochemistry_result: Optional[BiochemistryAnalysisResult] = None,
        vital_signs_result: Optional[VitalSignsAnalysisResult] = None
    ) -> List[str]:
        """
        生成紧急行动建议。
        
        Args:
            overall_risk: 整体风险等级
            cbc_result: 血常规分析结果
            biochemistry_result: 血生化分析结果
            vital_signs_result: 生命体征分析结果
        
        Returns:
            List[str]: 紧急行动建议列表
        """
        actions = []
        
        if overall_risk == RiskLevel.CRITICAL:
            actions.append("⚠️ 危急值警报！请立即前往急诊科就诊！")
            actions.append("建议拨打 120 急救电话")
        
        if cbc_result:
            critical_indicators = [i for i in cbc_result.indicators if i.risk_level == RiskLevel.CRITICAL]
            if critical_indicators:
                actions.append(f"血常规危急值：{', '.join([i.name for i in critical_indicators])}，需立即处理")
        
        if biochemistry_result:
            critical_indicators = [i for i in biochemistry_result.indicators if i.risk_level == RiskLevel.CRITICAL]
            if critical_indicators:
                actions.append(f"血生化危急值：{', '.join([i.name for i in critical_indicators])}，需立即处理")
        
        if vital_signs_result:
            critical_indicators = [i for i in vital_signs_result.indicators if i.risk_level == RiskLevel.CRITICAL]
            if critical_indicators:
                actions.append(f"生命体征危急值：{', '.join([i.name for i in critical_indicators])}，需立即处理")
        
        return actions
    
    def generate_follow_up_suggestions(
        self,
        overall_risk: RiskLevel,
        cbc_result: Optional[CBCAnalysisResult] = None,
        biochemistry_result: Optional[BiochemistryAnalysisResult] = None,
        urinalysis_result: Optional[UrinalysisAnalysisResult] = None
    ) -> List[str]:
        """
        生成随访建议。
        
        Args:
            overall_risk: 整体风险等级
            cbc_result: 血常规分析结果
            biochemistry_result: 血生化分析结果
            urinalysis_result: 尿常规分析结果
        
        Returns:
            List[str]: 随访建议列表
        """
        suggestions = []
        
        if overall_risk == RiskLevel.LOW:
            suggestions.append("建议每年进行一次常规体检")
        elif overall_risk == RiskLevel.MEDIUM:
            suggestions.append("建议 3-6 个月后复查异常指标")
        elif overall_risk == RiskLevel.HIGH:
            suggestions.append("建议 1-2 周后复查，并遵医嘱治疗")
        elif overall_risk == RiskLevel.CRITICAL:
            suggestions.append("请按医嘱进行治疗，出院后定期复查")
        
        if cbc_result and cbc_result.abnormal_count > 0:
            suggestions.append("血常规异常指标建议 2-4 周后复查")
        
        if biochemistry_result and biochemistry_result.abnormal_count > 0:
            suggestions.append("血生化异常指标建议 1-3 个月后复查")
        
        if urinalysis_result and urinalysis_result.abnormal_count > 0:
            suggestions.append("尿常规异常指标建议 1-2 周后复查")
        
        return suggestions
    
    def generate_comprehensive_summary(
        self,
        overall_risk: RiskLevel,
        risk_score: float,
        cbc_result: Optional[CBCAnalysisResult] = None,
        biochemistry_result: Optional[BiochemistryAnalysisResult] = None,
        urinalysis_result: Optional[UrinalysisAnalysisResult] = None,
        vital_signs_result: Optional[VitalSignsAnalysisResult] = None,
        symptom_result: Optional[SymptomAnalysisResult] = None
    ) -> str:
        """
        生成综合摘要。
        
        Args:
            overall_risk: 整体风险等级
            risk_score: 风险评分
            cbc_result: 血常规分析结果
            biochemistry_result: 血生化分析结果
            urinalysis_result: 尿常规分析结果
            vital_signs_result: 生命体征分析结果
            symptom_result: 症状分析结果
        
        Returns:
            str: 综合摘要
        """
        summary_parts = []
        
        summary_parts.append(f"综合健康评估：风险等级 {overall_risk.value}，风险评分 {risk_score:.1f} 分。")
        
        abnormal_counts = []
        if cbc_result and cbc_result.abnormal_count > 0:
            abnormal_counts.append(f"血常规 {cbc_result.abnormal_count} 项异常")
        if biochemistry_result and biochemistry_result.abnormal_count > 0:
            abnormal_counts.append(f"血生化 {biochemistry_result.abnormal_count} 项异常")
        if urinalysis_result and urinalysis_result.abnormal_count > 0:
            abnormal_counts.append(f"尿常规 {urinalysis_result.abnormal_count} 项异常")
        if vital_signs_result and vital_signs_result.abnormal_count > 0:
            abnormal_counts.append(f"生命体征 {vital_signs_result.abnormal_count} 项异常")
        if symptom_result and symptom_result.symptom_count > 0:
            abnormal_counts.append(f"症状 {symptom_result.symptom_count} 个")
        
        if abnormal_counts:
            summary_parts.append("检测发现：" + "、".join(abnormal_counts) + "。")
        else:
            summary_parts.append("各项检查指标均在正常范围内。")
        
        if overall_risk == RiskLevel.CRITICAL:
            summary_parts.append("⚠️ 存在危急值，请立即就医！")
        elif overall_risk == RiskLevel.HIGH:
            summary_parts.append("建议尽快就医检查，明确诊断。")
        elif overall_risk == RiskLevel.MEDIUM:
            summary_parts.append("建议定期复查，关注身体变化。")
        else:
            summary_parts.append("建议保持健康生活方式，定期体检。")
        
        return "".join(summary_parts)
    
    def perform_triage(
        self,
        cbc_report: Optional[str] = None,
        biochemistry_report: Optional[str] = None,
        urinalysis_report: Optional[str] = None,
        temperature: Optional[float] = None,
        heart_rate: Optional[float] = None,
        systolic_bp: Optional[float] = None,
        diastolic_bp: Optional[float] = None,
        respiratory_rate: Optional[float] = None,
        spo2: Optional[float] = None,
        symptom_text: Optional[str] = None,
        gender: Gender = Gender.UNKNOWN
    ) -> TriageResult:
        """
        执行综合分诊。
        
        Args:
            cbc_report: 血常规报告文本
            biochemistry_report: 血生化报告文本
            urinalysis_report: 尿常规报告文本
            temperature: 体温（℃）
            heart_rate: 心率（次/分）
            systolic_bp: 收缩压（mmHg）
            diastolic_bp: 舒张压（mmHg）
            respiratory_rate: 呼吸频率（次/分）
            spo2: 血氧饱和度（%）
            symptom_text: 症状文本
            gender: 性别
        
        Returns:
            TriageResult: 分诊结果
        """
        logger.info("开始执行综合分诊")
        
        cbc_result = None
        if cbc_report:
            try:
                cbc_result = cbc_analyzer.analyze(cbc_report, gender)
            except Exception as e:
                logger.error(f"血常规分析失败: {e}")
        
        biochemistry_result = None
        if biochemistry_report:
            try:
                biochemistry_result = biochemistry_analyzer.analyze(biochemistry_report, gender)
            except Exception as e:
                logger.error(f"血生化分析失败: {e}")
        
        urinalysis_result = None
        if urinalysis_report:
            try:
                urinalysis_result = urinalysis_analyzer.analyze(urinalysis_report)
            except Exception as e:
                logger.error(f"尿常规分析失败: {e}")
        
        vital_signs_result = None
        if any([temperature, heart_rate, systolic_bp, diastolic_bp, respiratory_rate, spo2]):
            try:
                vital_signs_result = vital_signs_analyzer.analyze_vital_signs(
                    temperature=temperature,
                    heart_rate=heart_rate,
                    systolic_bp=systolic_bp,
                    diastolic_bp=diastolic_bp,
                    respiratory_rate=respiratory_rate,
                    spo2=spo2
                )
            except Exception as e:
                logger.error(f"生命体征分析失败: {e}")
        
        symptom_result = None
        if symptom_text:
            try:
                symptom_result = symptom_analyzer.analyze_symptoms(symptom_text)
            except Exception as e:
                logger.error(f"症状分析失败: {e}")
        
        risk_score = self.calculate_risk_score(
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            urinalysis_result=urinalysis_result,
            vital_signs_result=vital_signs_result,
            symptom_result=symptom_result
        )
        
        overall_risk = self.determine_overall_risk(risk_score)
        
        department_recommendations = self.generate_department_recommendations(
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            urinalysis_result=urinalysis_result,
            vital_signs_result=vital_signs_result,
            symptom_result=symptom_result
        )
        
        urgent_actions = self.generate_urgent_actions(
            overall_risk=overall_risk,
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            vital_signs_result=vital_signs_result
        )
        
        follow_up_suggestions = self.generate_follow_up_suggestions(
            overall_risk=overall_risk,
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            urinalysis_result=urinalysis_result
        )
        
        comprehensive_summary = self.generate_comprehensive_summary(
            overall_risk=overall_risk,
            risk_score=risk_score,
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            urinalysis_result=urinalysis_result,
            vital_signs_result=vital_signs_result,
            symptom_result=symptom_result
        )
        
        result = TriageResult(
            overall_risk_level=overall_risk,
            risk_score=risk_score,
            cbc_result=cbc_result,
            biochemistry_result=biochemistry_result,
            urinalysis_result=urinalysis_result,
            vital_signs_result=vital_signs_result,
            symptom_result=symptom_result,
            department_recommendations=department_recommendations,
            urgent_actions=urgent_actions,
            follow_up_suggestions=follow_up_suggestions,
            comprehensive_summary=comprehensive_summary
        )
        
        logger.info(f"分诊完成: 风险等级 {overall_risk.value}，评分 {risk_score:.1f}")
        return result


triage_engine = TriageEngine()
