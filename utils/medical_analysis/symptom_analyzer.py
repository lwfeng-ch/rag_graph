"""
症状文本分析模块

功能：
- 从用户输入文本中识别关键症状
- 建立症状与检验指标的关联性分析模型
- 提升分诊准确性
"""

import re
import logging
from typing import Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SymptomCategory(Enum):
    """症状分类枚举"""
    GENERAL = "全身症状"
    RESPIRATORY = "呼吸系统"
    CARDIOVASCULAR = "心血管系统"
    DIGESTIVE = "消化系统"
    URINARY = "泌尿系统"
    NEUROLOGICAL = "神经系统"
    MUSCULOSKELETAL = "肌肉骨骼系统"
    SKIN = "皮肤"


@dataclass
class Symptom:
    """症状数据类"""
    name: str
    category: SymptomCategory
    keywords: List[str]
    related_indicators: List[str] = field(default_factory=list)
    severity: str = "轻度"


@dataclass
class SymptomAnalysisResult:
    """症状分析结果数据类"""
    detected_symptoms: List[Symptom] = field(default_factory=list)
    symptom_count: int = 0
    categories: Set[SymptomCategory] = field(default_factory=set)
    related_indicators: List[str] = field(default_factory=list)
    urgency_level: str = "低"
    summary: str = ""


class SymptomAnalyzer:
    """症状分析器"""
    
    def __init__(self):
        self.symptom_database = self._init_symptom_database()
    
    def _init_symptom_database(self) -> Dict[str, Symptom]:
        """
        初始化症状数据库。
        
        Returns:
            Dict[str, Symptom]: 症状名称到症状对象的映射
        """
        return {
            "发烧": Symptom(
                name="发烧",
                category=SymptomCategory.GENERAL,
                keywords=["发烧", "发热", "体温高", "发烫", "烧"],
                related_indicators=["WBC", "TEMP"],
                severity="中度"
            ),
            "咳嗽": Symptom(
                name="咳嗽",
                category=SymptomCategory.RESPIRATORY,
                keywords=["咳嗽", "咳", "咳痰", "干咳"],
                related_indicators=["WBC", "NEUT%"],
                severity="轻度"
            ),
            "胸痛": Symptom(
                name="胸痛",
                category=SymptomCategory.CARDIOVASCULAR,
                keywords=["胸痛", "胸口疼", "胸闷", "心前区疼痛"],
                related_indicators=["HR", "SBP", "DBP", "AST"],
                severity="重度"
            ),
            "乏力": Symptom(
                name="乏力",
                category=SymptomCategory.GENERAL,
                keywords=["乏力", "疲劳", "没劲", "无力", "累"],
                related_indicators=["HGB", "GLU_fasting"],
                severity="轻度"
            ),
            "头晕": Symptom(
                name="头晕",
                category=SymptomCategory.NEUROLOGICAL,
                keywords=["头晕", "眩晕", "头昏", "晕"],
                related_indicators=["HGB", "SBP", "DBP", "GLU_fasting"],
                severity="中度"
            ),
            "头痛": Symptom(
                name="头痛",
                category=SymptomCategory.NEUROLOGICAL,
                keywords=["头痛", "头疼", "偏头痛", "脑袋疼"],
                related_indicators=["SBP", "DBP"],
                severity="中度"
            ),
            "恶心": Symptom(
                name="恶心",
                category=SymptomCategory.DIGESTIVE,
                keywords=["恶心", "想吐", "反胃"],
                related_indicators=["ALT", "AST", "GLU_fasting"],
                severity="轻度"
            ),
            "呕吐": Symptom(
                name="呕吐",
                category=SymptomCategory.DIGESTIVE,
                keywords=["呕吐", "吐", "吐了"],
                related_indicators=["ALT", "AST", "KET"],
                severity="中度"
            ),
            "腹痛": Symptom(
                name="腹痛",
                category=SymptomCategory.DIGESTIVE,
                keywords=["腹痛", "肚子疼", "腹部疼痛", "肚子痛"],
                related_indicators=["WBC", "ALT", "AST"],
                severity="中度"
            ),
            "腹泻": Symptom(
                name="腹泻",
                category=SymptomCategory.DIGESTIVE,
                keywords=["腹泻", "拉肚子", "拉稀", "腹泻"],
                related_indicators=["WBC", "HGB"],
                severity="中度"
            ),
            "尿频": Symptom(
                name="尿频",
                category=SymptomCategory.URINARY,
                keywords=["尿频", "小便多", "尿多"],
                related_indicators=["GLU_fasting", "PRO", "LEU"],
                severity="轻度"
            ),
            "尿急": Symptom(
                name="尿急",
                category=SymptomCategory.URINARY,
                keywords=["尿急", "憋不住尿"],
                related_indicators=["LEU", "NIT"],
                severity="轻度"
            ),
            "尿痛": Symptom(
                name="尿痛",
                category=SymptomCategory.URINARY,
                keywords=["尿痛", "小便疼", "排尿疼痛"],
                related_indicators=["LEU", "BLD", "NIT"],
                severity="中度"
            ),
            "水肿": Symptom(
                name="水肿",
                category=SymptomCategory.URINARY,
                keywords=["水肿", "浮肿", "肿胀"],
                related_indicators=["PRO", "ALB", "Cr"],
                severity="中度"
            ),
            "心悸": Symptom(
                name="心悸",
                category=SymptomCategory.CARDIOVASCULAR,
                keywords=["心悸", "心慌", "心跳快"],
                related_indicators=["HR", "HGB", "GLU_fasting"],
                severity="中度"
            ),
            "气短": Symptom(
                name="气短",
                category=SymptomCategory.RESPIRATORY,
                keywords=["气短", "呼吸困难", "喘", "喘不上气"],
                related_indicators=["HR", "SPO2", "HGB"],
                severity="重度"
            ),
            "皮疹": Symptom(
                name="皮疹",
                category=SymptomCategory.SKIN,
                keywords=["皮疹", "疹子", "红疹", "皮肤红"],
                related_indicators=["WBC", "PLT"],
                severity="轻度"
            ),
            "关节痛": Symptom(
                name="关节痛",
                category=SymptomCategory.MUSCULOSKELETAL,
                keywords=["关节痛", "关节疼", "关节疼痛"],
                related_indicators=["WBC", "NEUT%"],
                severity="中度"
            )
        }
    
    def extract_symptoms(self, text: str) -> List[Symptom]:
        """
        从文本中提取症状。
        
        Args:
            text: 用户输入的文本
        
        Returns:
            List[Symptom]: 检测到的症状列表
        
        Raises:
            ValueError: 如果文本为空
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        detected_symptoms = []
        
        for symptom_name, symptom in self.symptom_database.items():
            for keyword in symptom.keywords:
                if keyword in text:
                    detected_symptoms.append(symptom)
                    logger.debug(f"检测到症状: {symptom_name}（关键词: {keyword}）")
                    break
        
        logger.info(f"从文本中检测到 {len(detected_symptoms)} 个症状")
        return detected_symptoms
    
    def analyze_symptoms(self, text: str) -> SymptomAnalysisResult:
        """
        综合分析症状文本。
        
        Args:
            text: 用户输入的文本
        
        Returns:
            SymptomAnalysisResult: 分析结果
        """
        logger.info("开始分析症状文本")
        
        detected_symptoms = self.extract_symptoms(text)
        
        if not detected_symptoms:
            return SymptomAnalysisResult(
                detected_symptoms=[],
                symptom_count=0,
                categories=set(),
                related_indicators=[],
                urgency_level="低",
                summary="未检测到明显症状描述。"
            )
        
        categories = {s.category for s in detected_symptoms}
        
        related_indicators = []
        for symptom in detected_symptoms:
            related_indicators.extend(symptom.related_indicators)
        related_indicators = list(set(related_indicators))
        
        urgency_level = self._determine_urgency(detected_symptoms)
        
        summary = self._generate_summary(detected_symptoms, categories, urgency_level)
        
        result = SymptomAnalysisResult(
            detected_symptoms=detected_symptoms,
            symptom_count=len(detected_symptoms),
            categories=categories,
            related_indicators=related_indicators,
            urgency_level=urgency_level,
            summary=summary
        )
        
        logger.info(f"症状分析完成: {len(detected_symptoms)} 个症状，紧急程度: {urgency_level}")
        return result
    
    def _determine_urgency(self, symptoms: List[Symptom]) -> str:
        """
        判断症状紧急程度。
        
        Args:
            symptoms: 症状列表
        
        Returns:
            str: 紧急程度（"高", "中", "低"）
        """
        severity_weights = {"重度": 3, "中度": 2, "轻度": 1}
        
        total_severity = sum(severity_weights.get(s.severity, 1) for s in symptoms)
        
        if total_severity >= 6 or any(s.severity == "重度" for s in symptoms):
            return "高"
        elif total_severity >= 3:
            return "中"
        else:
            return "低"
    
    def _generate_summary(
        self,
        symptoms: List[Symptom],
        categories: Set[SymptomCategory],
        urgency_level: str
    ) -> str:
        """
        生成分析摘要。
        
        Args:
            symptoms: 症状列表
            categories: 症状分类集合
            urgency_level: 紧急程度
        
        Returns:
            str: 分析摘要
        """
        symptom_names = [s.name for s in symptoms]
        category_names = [c.value for c in categories]
        
        summary = f"检测到 {len(symptoms)} 个症状：{', '.join(symptom_names)}。"
        summary += f"涉及系统：{', '.join(category_names)}。"
        summary += f"紧急程度：{urgency_level}。"
        
        if urgency_level == "高":
            summary += "建议尽快就医！"
        elif urgency_level == "中":
            summary += "建议及时就医检查。"
        
        return summary


symptom_analyzer = SymptomAnalyzer()
