# utils/feishu_mcp.py
"""
飞书 MCP（Model Context Protocol）集成模块

功能：
- 飞书多维表格记录管理
- 高危风险记录上报
- 支持可选启用（未配置时降级运行）

使用方式：
    from utils.feishu_mcp import feishu_mcp_manager
    
    if feishu_mcp_manager.is_initialized():
        feishu_mcp_manager.add_critical_risk_record(
            user_id="user123",
            risk_data={...}
        )

注意：
- 飞书 MCP 是可选功能，未配置时不会影响主流程
- 需要配置环境变量：FEISHU_APP_ID, FEISHU_APP_SECRET
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeishuMCPConfig:
    """
    飞书 MCP 配置。

    Attributes:
        app_id: 飞书应用 ID
        app_secret: 飞书应用密钥
        base_id: 多维表格 Base ID
        table_id: 数据表 ID
        enabled: 是否启用飞书 MCP
    """
    app_id: str = ""
    app_secret: str = ""
    base_id: str = ""
    table_id: str = ""
    enabled: bool = False

    def __post_init__(self):
        if not self.app_id:
            self.app_id = os.getenv("FEISHU_APP_ID", "")
        if not self.app_secret:
            self.app_secret = os.getenv("FEISHU_APP_SECRET", "")
        if not self.base_id:
            self.base_id = os.getenv("FEISHU_BASE_ID", "")
        if not self.table_id:
            self.table_id = os.getenv("FEISHU_TABLE_ID", "")
        
        self.enabled = bool(self.app_id and self.app_secret and self.base_id and self.table_id)


class FeishuMCPManager:
    """
    飞书 MCP 管理器。

    负责与飞书多维表格交互，包括：
    - 获取访问令牌
    - 创建记录
    - 查询记录
    """

    def __init__(self, config: Optional[FeishuMCPConfig] = None):
        """
        初始化飞书 MCP 管理器。

        Args:
            config: 飞书 MCP 配置，为 None 时从环境变量读取
        """
        self.config = config or FeishuMCPConfig()
        self._access_token: Optional[str] = None
        self._initialized = False
        
        if self.config.enabled:
            self._initialize()

    def _initialize(self) -> bool:
        """
        初始化飞书 MCP 连接。

        Returns:
            bool: 初始化是否成功
        """
        try:
            import requests
            
            url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            headers = {"Content-Type": "application/json"}
            data = {
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            
            if result.get("code") == 0:
                self._access_token = result.get("tenant_access_token")
                self._initialized = True
                logger.info("飞书 MCP 初始化成功")
                return True
            else:
                logger.warning(f"飞书 MCP 初始化失败: {result.get('msg')}")
                return False
                
        except ImportError:
            logger.warning("requests 库未安装，飞书 MCP 功能不可用")
            return False
        except Exception as e:
            logger.warning(f"飞书 MCP 初始化异常: {e}")
            return False

    def is_initialized(self) -> bool:
        """
        检查飞书 MCP 是否已初始化。

        Returns:
            bool: 是否已初始化
        """
        return self._initialized

    def add_critical_risk_record(
        self,
        user_id: str,
        risk_data: Dict[str, Any],
    ) -> bool:
        """
        添加高危风险记录到飞书多维表格。

        Args:
            user_id: 用户 ID
            risk_data: 风险数据，包含：
                - risk_level: 风险等级
                - risk_warning: 风险警告
                - symptoms: 症状列表
                - recommended_departments: 推荐科室
                - triage_confidence: 分诊置信度

        Returns:
            bool: 是否添加成功

        Example:
            >>> success = manager.add_critical_risk_record(
            ...     user_id="user123",
            ...     risk_data={
            ...         "risk_level": "critical",
            ...         "risk_warning": "出现胸痛症状，建议立即就医",
            ...         "symptoms": ["胸痛", "呼吸困难"],
            ...         "recommended_departments": ["急诊科", "心内科"],
            ...         "triage_confidence": 0.95,
            ...     }
            ... )
        """
        if not self._initialized:
            logger.warning("飞书 MCP 未初始化，跳过风险记录上报")
            return False
        
        try:
            import requests
            from datetime import datetime
            
            url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.config.base_id}/tables/{self.config.table_id}/records"
            
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }
            
            fields = {
                "用户ID": user_id,
                "风险等级": risk_data.get("risk_level", "unknown"),
                "风险警告": risk_data.get("risk_warning", ""),
                "症状列表": ", ".join(risk_data.get("symptoms", [])),
                "推荐科室": ", ".join(risk_data.get("recommended_departments", [])),
                "分诊置信度": risk_data.get("triage_confidence", 0.0),
                "上报时间": datetime.now().isoformat(),
            }
            
            data = {
                "fields": fields,
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            
            if result.get("code") == 0:
                logger.info(f"高危风险记录上报成功: user_id={user_id}")
                return True
            else:
                logger.warning(f"高危风险记录上报失败: {result.get('msg')}")
                return False
                
        except Exception as e:
            logger.error(f"高危风险记录上报异常: {e}", exc_info=True)
            return False

    def refresh_token(self) -> bool:
        """
        刷新访问令牌。

        Returns:
            bool: 是否刷新成功
        """
        if not self.config.enabled:
            return False
        
        return self._initialize()


feishu_mcp_manager = FeishuMCPManager()
