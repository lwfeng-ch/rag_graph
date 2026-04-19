"""
医疗检验指标参考数据库模块

数据来源：
- 国家卫生健康委员会 WS/T 405—2012 血细胞分析参考区间
- 国家卫生健康委员会 WS/T 779-2021 儿童血细胞分析参考区间
- 三甲医院医学检验中心临床检验标准

包含：
- 血常规（CBC）指标参考范围
- 血生化指标参考范围
- 尿常规指标参考范围
- 生命体征参考范围
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Gender(Enum):
    """性别枚举"""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(Enum):
    """年龄分组枚举"""

    ADULT = "adult"  # 成人（≥18岁）
    CHILD = "child"  # 儿童（6个月-18岁）
    INFANT = "infant"  # 婴幼儿（<6个月）
    NEWBORN = "newborn"  # 新生儿（<28天）


class RiskLevel(Enum):
    """风险等级枚举"""

    LOW = "低风险"
    MEDIUM = "中风险"
    HIGH = "高风险"
    CRITICAL = "危急值"


@dataclass
class ReferenceRange:
    """参考范围数据类"""

    name: str
    name_en: str
    unit: str
    normal_min: float
    normal_max: float
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    description: str = ""
    clinical_significance: Dict[str, str] = None

    def __post_init__(self):
        if self.clinical_significance is None:
            self.clinical_significance = {}


@dataclass
class QualitativeReference:
    """定性指标参考数据类"""

    name: str
    name_en: str
    normal_value: str
    abnormal_values: List[str]
    description: str = ""
    clinical_significance: Dict[str, str] = None

    def __post_init__(self):
        if self.clinical_significance is None:
            self.clinical_significance = {}


class MedicalReferenceDatabase:
    """医疗检验指标参考数据库"""

    def __init__(self):
        self.cbc_references = self._init_cbc_references()
        self.biochemistry_references = self._init_biochemistry_references()
        self.urinalysis_references = self._init_urinalysis_references()
        self.vital_signs_references = self._init_vital_signs_references()

    def _init_cbc_references(self) -> Dict[str, ReferenceRange]:
        """
        初始化血常规（CBC）参考范围。

        数据来源：国家卫生健康委员会 WS/T 405—2012

        Returns:
            Dict[str, ReferenceRange]: 血常规指标参考范围字典
        """
        return {
            "WBC": ReferenceRange(
                name="白细胞计数",
                name_en="White Blood Cell Count",
                unit="×10⁹/L",
                normal_min=4.0,
                normal_max=10.0,
                critical_low=2.0,
                critical_high=30.0,
                description="白细胞是免疫系统的重要组成部分",
                clinical_significance={
                    "升高": "细菌感染、炎症、应激、白血病、组织坏死",
                    "降低": "病毒感染、骨髓抑制、自身免疫病、脾功能亢进",
                },
            ),
            "HGB_male": ReferenceRange(
                name="血红蛋白（男）",
                name_en="Hemoglobin (Male)",
                unit="g/L",
                normal_min=130.0,
                normal_max=175.0,
                critical_low=70.0,
                critical_high=200.0,
                description="血红蛋白是红细胞内运输氧的特殊蛋白质",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症、慢性缺氧",
                    "降低": "贫血、白血病、营养不良",
                },
            ),
            "HGB_female": ReferenceRange(
                name="血红蛋白（女）",
                name_en="Hemoglobin (Female)",
                unit="g/L",
                normal_min=120.0,
                normal_max=150.0,
                critical_low=70.0,
                critical_high=200.0,
                description="血红蛋白是红细胞内运输氧的特殊蛋白质",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症、慢性缺氧",
                    "降低": "贫血、白血病、营养不良",
                },
            ),
            "PLT": ReferenceRange(
                name="血小板计数",
                name_en="Platelet Count",
                unit="×10⁹/L",
                normal_min=125.0,
                normal_max=350.0,
                critical_low=50.0,
                critical_high=500.0,
                description="血小板参与止血和凝血过程",
                clinical_significance={
                    "升高": "炎症、缺铁性贫血、骨髓增殖性疾病",
                    "降低": "血小板减少症、白血病、脾功能亢进",
                },
            ),
            "RBC_male": ReferenceRange(
                name="红细胞计数（男）",
                name_en="Red Blood Cell Count (Male)",
                unit="×10¹²/L",
                normal_min=4.3,
                normal_max=5.8,
                critical_low=3.0,
                critical_high=7.0,
                description="红细胞是血液中数量最多的细胞",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症、慢性缺氧",
                    "降低": "贫血、白血病、营养不良",
                },
            ),
            "RBC_female": ReferenceRange(
                name="红细胞计数（女）",
                name_en="Red Blood Cell Count (Female)",
                unit="×10¹²/L",
                normal_min=3.8,
                normal_max=5.1,
                critical_low=3.0,
                critical_high=7.0,
                description="红细胞是血液中数量最多的细胞",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症、慢性缺氧",
                    "降低": "贫血、白血病、营养不良",
                },
            ),
            "HCT_male": ReferenceRange(
                name="红细胞压积（男）",
                name_en="Hematocrit (Male)",
                unit="L/L",
                normal_min=0.40,
                normal_max=0.54,
                critical_low=0.25,
                critical_high=0.60,
                description="红细胞在全血中所占体积比例",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症",
                    "降低": "贫血、营养不良",
                },
            ),
            "HCT_female": ReferenceRange(
                name="红细胞压积（女）",
                name_en="Hematocrit (Female)",
                unit="L/L",
                normal_min=0.37,
                normal_max=0.47,
                critical_low=0.25,
                critical_high=0.60,
                description="红细胞在全血中所占体积比例",
                clinical_significance={
                    "升高": "脱水、真性红细胞增多症",
                    "降低": "贫血、营养不良",
                },
            ),
            "MCV": ReferenceRange(
                name="平均红细胞体积",
                name_en="Mean Corpuscular Volume",
                unit="fL",
                normal_min=80.0,
                normal_max=100.0,
                critical_low=60.0,
                critical_high=120.0,
                description="红细胞平均体积，用于贫血类型鉴别",
                clinical_significance={
                    "升高": "巨幼细胞性贫血、溶血性贫血",
                    "降低": "缺铁性贫血、地中海贫血",
                },
            ),
            "MCH": ReferenceRange(
                name="平均血红蛋白量",
                name_en="Mean Corpuscular Hemoglobin",
                unit="pg",
                normal_min=27.0,
                normal_max=34.0,
                critical_low=20.0,
                critical_high=40.0,
                description="红细胞平均血红蛋白含量",
                clinical_significance={"升高": "巨幼细胞性贫血", "降低": "缺铁性贫血"},
            ),
            "MCHC": ReferenceRange(
                name="平均血红蛋白浓度",
                name_en="Mean Corpuscular Hemoglobin Concentration",
                unit="g/L",
                normal_min=320.0,
                normal_max=360.0,
                critical_low=280.0,
                critical_high=400.0,
                description="红细胞平均血红蛋白浓度",
                clinical_significance={
                    "升高": "球形红细胞增多症",
                    "降低": "缺铁性贫血",
                },
            ),
            "NEUT%": ReferenceRange(
                name="中性粒细胞百分比",
                name_en="Neutrophil Percentage",
                unit="%",
                normal_min=50.0,
                normal_max=70.0,
                critical_low=20.0,
                critical_high=90.0,
                description="中性粒细胞是白细胞的主要成分",
                clinical_significance={
                    "升高": "细菌感染、炎症、应激",
                    "降低": "病毒感染、骨髓抑制",
                },
            ),
            "LYMPH%": ReferenceRange(
                name="淋巴细胞百分比",
                name_en="Lymphocyte Percentage",
                unit="%",
                normal_min=20.0,
                normal_max=40.0,
                critical_low=10.0,
                critical_high=60.0,
                description="淋巴细胞参与免疫反应",
                clinical_significance={
                    "升高": "病毒感染、结核病、淋巴细胞白血病",
                    "降低": "免疫缺陷、应激状态",
                },
            ),
        }

    def _init_biochemistry_references(self) -> Dict[str, ReferenceRange]:
        """
        初始化血生化参考范围。

        数据来源：三甲医院医学检验中心标准

        Returns:
            Dict[str, ReferenceRange]: 血生化指标参考范围字典
        """
        return {
            "GLU_fasting": ReferenceRange(
                name="空腹血糖",
                name_en="Fasting Blood Glucose",
                unit="mmol/L",
                normal_min=3.9,
                normal_max=6.1,
                critical_low=2.8,
                critical_high=16.7,
                description="空腹状态下的血糖浓度",
                clinical_significance={
                    "升高": "糖尿病、应激状态、胰腺炎",
                    "降低": "低血糖、胰岛素过量、营养不良",
                },
            ),
            "GLU_postprandial": ReferenceRange(
                name="餐后2小时血糖",
                name_en="2-hour Postprandial Blood Glucose",
                unit="mmol/L",
                normal_min=3.9,
                normal_max=7.8,
                critical_low=2.8,
                critical_high=16.7,
                description="餐后2小时血糖浓度",
                clinical_significance={"升高": "糖尿病、糖耐量异常", "降低": "低血糖"},
            ),
            "HbA1c": ReferenceRange(
                name="糖化血红蛋白",
                name_en="Glycated Hemoglobin",
                unit="%",
                normal_min=4.0,
                normal_max=6.0,
                critical_low=4.0,
                critical_high=10.0,
                description="反映过去2-3个月的平均血糖水平",
                clinical_significance={
                    "升高": "糖尿病控制不佳",
                    "降低": "低血糖、溶血性贫血",
                },
            ),
            "Cr_male": ReferenceRange(
                name="肌酐（男）",
                name_en="Creatinine (Male)",
                unit="μmol/L",
                normal_min=53.0,
                normal_max=106.0,
                critical_low=30.0,
                critical_high=200.0,
                description="肌酐是肌肉代谢产物，反映肾功能",
                clinical_significance={
                    "升高": "肾功能不全、脱水、心力衰竭",
                    "降低": "营养不良、肌肉萎缩",
                },
            ),
            "Cr_female": ReferenceRange(
                name="肌酐（女）",
                name_en="Creatinine (Female)",
                unit="μmol/L",
                normal_min=44.0,
                normal_max=97.0,
                critical_low=30.0,
                critical_high=200.0,
                description="肌酐是肌肉代谢产物，反映肾功能",
                clinical_significance={
                    "升高": "肾功能不全、脱水、心力衰竭",
                    "降低": "营养不良、肌肉萎缩",
                },
            ),
            "BUN": ReferenceRange(
                name="尿素氮",
                name_en="Blood Urea Nitrogen",
                unit="mmol/L",
                normal_min=2.9,
                normal_max=8.2,
                critical_low=1.0,
                critical_high=20.0,
                description="尿素氮是蛋白质代谢产物",
                clinical_significance={
                    "升高": "肾功能不全、脱水、消化道出血",
                    "降低": "肝功能不全、营养不良",
                },
            ),
            "ALT": ReferenceRange(
                name="谷丙转氨酶",
                name_en="Alanine Aminotransferase",
                unit="U/L",
                normal_min=0.0,
                normal_max=40.0,
                critical_low=0.0,
                critical_high=200.0,
                description="ALT主要存在于肝细胞中",
                clinical_significance={"升高": "肝损伤、肝炎、脂肪肝、药物性肝损伤"},
            ),
            "AST": ReferenceRange(
                name="谷草转氨酶",
                name_en="Aspartate Aminotransferase",
                unit="U/L",
                normal_min=0.0,
                normal_max=40.0,
                critical_low=0.0,
                critical_high=200.0,
                description="AST存在于肝、心、骨骼肌中",
                clinical_significance={"升高": "肝损伤、心肌梗死、肌肉损伤"},
            ),
            "TBIL": ReferenceRange(
                name="总胆红素",
                name_en="Total Bilirubin",
                unit="μmol/L",
                normal_min=3.4,
                normal_max=17.1,
                critical_low=0.0,
                critical_high=50.0,
                description="胆红素是胆色素的主要成分",
                clinical_significance={"升高": "黄疸、肝胆疾病、溶血"},
            ),
            "DBIL": ReferenceRange(
                name="直接胆红素",
                name_en="Direct Bilirubin",
                unit="μmol/L",
                normal_min=0.0,
                normal_max=6.8,
                critical_low=0.0,
                critical_high=30.0,
                description="直接胆红素是结合胆红素",
                clinical_significance={"升高": "梗阻性黄疸、肝细胞性黄疸"},
            ),
            "ALB": ReferenceRange(
                name="白蛋白",
                name_en="Albumin",
                unit="g/L",
                normal_min=35.0,
                normal_max=55.0,
                critical_low=25.0,
                critical_high=60.0,
                description="白蛋白是血浆中主要的蛋白质",
                clinical_significance={
                    "升高": "脱水",
                    "降低": "肝功能不全、营养不良、肾病综合征",
                },
            ),
        }

    def _init_urinalysis_references(self) -> Dict[str, QualitativeReference]:
        """
        初始化尿常规参考范围。

        数据来源：三甲医院医学检验中心标准

        Returns:
            Dict[str, QualitativeReference]: 尿常规指标参考范围字典
        """
        return {
            "PRO": QualitativeReference(
                name="尿蛋白",
                name_en="Urine Protein",
                normal_value="阴性（-）",
                abnormal_values=[
                    "阳性（+）",
                    "阳性（++）",
                    "阳性（+++）",
                    "阳性（++++）",
                ],
                description="尿液中蛋白质含量",
                clinical_significance={"阳性": "肾病、糖尿病肾病、高血压肾损害、肾炎"},
            ),
            "GLU": QualitativeReference(
                name="尿糖",
                name_en="Urine Glucose",
                normal_value="阴性（-）",
                abnormal_values=[
                    "阳性（+）",
                    "阳性（++）",
                    "阳性（+++）",
                    "阳性（++++）",
                ],
                description="尿液中葡萄糖含量",
                clinical_significance={"阳性": "糖尿病、肾性糖尿、妊娠期糖尿"},
            ),
            "BLD": QualitativeReference(
                name="尿潜血",
                name_en="Urine Blood",
                normal_value="阴性（-）",
                abnormal_values=["阳性（+）", "阳性（++）", "阳性（+++）"],
                description="尿液中红细胞含量",
                clinical_significance={"阳性": "泌尿系统感染、肾结石、肾炎、肿瘤"},
            ),
            "LEU": QualitativeReference(
                name="尿白细胞",
                name_en="Urine Leukocyte",
                normal_value="阴性（-）",
                abnormal_values=["阳性（+）", "阳性（++）", "阳性（+++）"],
                description="尿液中白细胞含量",
                clinical_significance={"阳性": "泌尿系统感染、肾炎"},
            ),
            "KET": QualitativeReference(
                name="尿酮体",
                name_en="Urine Ketone",
                normal_value="阴性（-）",
                abnormal_values=["阳性（+）", "阳性（++）", "阳性（+++）"],
                description="尿液中酮体含量",
                clinical_significance={"阳性": "糖尿病酮症酸中毒、饥饿、妊娠剧吐"},
            ),
            "NIT": QualitativeReference(
                name="尿亚硝酸盐",
                name_en="Urine Nitrite",
                normal_value="阴性（-）",
                abnormal_values=["阳性（+）"],
                description="尿液中亚硝酸盐含量",
                clinical_significance={"阳性": "泌尿系统细菌感染"},
            ),
            "BIL": QualitativeReference(
                name="尿胆红素",
                name_en="Urine Bilirubin",
                normal_value="阴性（-）",
                abnormal_values=["阳性（+）", "阳性（++）"],
                description="尿液中胆红素含量",
                clinical_significance={"阳性": "梗阻性黄疸、肝细胞性黄疸"},
            ),
            "UBG": QualitativeReference(
                name="尿胆原",
                name_en="Urine Urobilinogen",
                normal_value="阴性（-）或弱阳性（±）",
                abnormal_values=["阳性（+）", "阳性（++）"],
                description="尿液中尿胆原含量",
                clinical_significance={"阳性": "溶血性黄疸、肝细胞性黄疸"},
            ),
            "PH": ReferenceRange(
                name="尿酸碱度",
                name_en="Urine pH",
                unit="",
                normal_min=5.0,
                normal_max=8.0,
                critical_low=4.5,
                critical_high=9.0,
                description="尿液的酸碱度",
                clinical_significance={
                    "升高": "泌尿系统感染、代谢性碱中毒",
                    "降低": "代谢性酸中毒、糖尿病酮症酸中毒",
                },
            ),
            "SG": ReferenceRange(
                name="尿比重",
                name_en="Specific Gravity",
                unit="",
                normal_min=1.005,
                normal_max=1.030,
                critical_low=1.001,
                critical_high=1.035,
                description="尿液的浓缩程度",
                clinical_significance={
                    "升高": "脱水、糖尿病、心衰",
                    "降低": "大量饮水、尿崩症、慢性肾衰竭",
                },
            ),
        }

    def _init_vital_signs_references(self) -> Dict[str, ReferenceRange]:
        """
        初始化生命体征参考范围。

        Returns:
            Dict[str, ReferenceRange]: 生命体征参考范围字典
        """
        return {
            "TEMP": ReferenceRange(
                name="体温",
                name_en="Temperature",
                unit="℃",
                normal_min=36.0,
                normal_max=37.3,
                critical_low=35.0,
                critical_high=41.0,
                description="人体核心温度",
                clinical_significance={
                    "升高": "感染、炎症、中暑",
                    "降低": "低体温、休克",
                },
            ),
            "HR": ReferenceRange(
                name="心率",
                name_en="Heart Rate",
                unit="次/分",
                normal_min=60.0,
                normal_max=100.0,
                critical_low=40.0,
                critical_high=150.0,
                description="心脏每分钟跳动的次数",
                clinical_significance={
                    "升高": "发热、贫血、甲亢、心力衰竭",
                    "降低": "甲减、运动员、心脏传导阻滞",
                },
            ),
            "SBP": ReferenceRange(
                name="收缩压",
                name_en="Systolic Blood Pressure",
                unit="mmHg",
                normal_min=90.0,
                normal_max=139.0,
                critical_low=80.0,
                critical_high=180.0,
                description="心脏收缩时的血压",
                clinical_significance={
                    "升高": "高血压、应激状态",
                    "降低": "休克、脱水、心功能不全",
                },
            ),
            "DBP": ReferenceRange(
                name="舒张压",
                name_en="Diastolic Blood Pressure",
                unit="mmHg",
                normal_min=60.0,
                normal_max=89.0,
                critical_low=50.0,
                critical_high=120.0,
                description="心脏舒张时的血压",
                clinical_significance={"升高": "高血压", "降低": "休克、脱水"},
            ),
            "RR": ReferenceRange(
                name="呼吸频率",
                name_en="Respiratory Rate",
                unit="次/分",
                normal_min=16.0,
                normal_max=20.0,
                critical_low=10.0,
                critical_high=30.0,
                description="每分钟呼吸的次数",
                clinical_significance={
                    "升高": "发热、肺炎、心衰",
                    "降低": "镇静剂过量、颅内压增高",
                },
            ),
            "SPO2": ReferenceRange(
                name="血氧饱和度",
                name_en="Blood Oxygen Saturation",
                unit="%",
                normal_min=95.0,
                normal_max=100.0,
                critical_low=90.0,
                critical_high=100.0,
                description="血液中氧气的饱和程度",
                clinical_significance={"降低": "呼吸衰竭、肺炎、心衰"},
            ),
        }

    def get_reference(
        self, test_type: str, indicator: str, gender: Gender = Gender.UNKNOWN
    ) -> Optional[ReferenceRange]:
        """
        获取指定指标的参考范围。

        Args:
            test_type: 检测类型（"cbc", "biochemistry", "urinalysis", "vital_signs"）
            indicator: 指标名称或缩写
            gender: 性别（用于性别相关指标）

        Returns:
            Optional[ReferenceRange]: 参考范围对象，如果未找到则返回 None

        Raises:
            ValueError: 如果 test_type 不在支持的类型列表中
        """
        if test_type == "cbc":
            references = self.cbc_references
        elif test_type == "biochemistry":
            references = self.biochemistry_references
        elif test_type == "urinalysis":
            references = self.urinalysis_references
        elif test_type == "vital_signs":
            references = self.vital_signs_references
        else:
            raise ValueError(f"不支持的检测类型: {test_type}")

        if gender == Gender.MALE and f"{indicator}_male" in references:
            return references[f"{indicator}_male"]
        elif gender == Gender.FEMALE and f"{indicator}_female" in references:
            return references[f"{indicator}_female"]
        elif indicator in references:
            return references[indicator]

        return None

    def evaluate_value(
        self, value: float, reference: ReferenceRange
    ) -> Tuple[str, RiskLevel]:
        """
        评估数值是否异常及风险等级。

        Args:
            value: 待评估的数值
            reference: 参考范围对象

        Returns:
            Tuple[str, RiskLevel]: (异常状态描述, 风险等级)
        """
        if value < reference.normal_min:
            if reference.critical_low and value <= reference.critical_low:
                return "危急值偏低", RiskLevel.CRITICAL
            elif value < reference.normal_min * 0.8:
                return "显著偏低", RiskLevel.HIGH
            else:
                return "偏低", RiskLevel.MEDIUM
        elif value > reference.normal_max:
            if reference.critical_high and value >= reference.critical_high:
                return "危急值偏高", RiskLevel.CRITICAL
            elif value > reference.normal_max * 1.2:
                return "显著偏高", RiskLevel.HIGH
            else:
                return "偏高", RiskLevel.MEDIUM
        else:
            return "正常", RiskLevel.LOW

    def evaluate_qualitative(
        self, value: str, reference: QualitativeReference
    ) -> Tuple[str, RiskLevel]:
        """
        评估定性指标是否异常及风险等级。

        Args:
            value: 待评估的定性值
            reference: 定性指标参考对象

        Returns:
            Tuple[str, RiskLevel]: (异常状态描述, 风险等级)
        """
        if value == reference.normal_value or "阴性" in value:
            return "正常", RiskLevel.LOW
        elif value in reference.abnormal_values or "阳性" in value:
            if "++++" in value or "+++" in value:
                return "显著异常", RiskLevel.HIGH
            elif "++" in value:
                return "中度异常", RiskLevel.MEDIUM
            else:
                return "轻度异常", RiskLevel.LOW
        else:
            return "未知", RiskLevel.LOW


medical_db = MedicalReferenceDatabase()
