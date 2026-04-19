# utils/auth.py
"""
用户认证模块

功能：
- 支持三种认证方式：API Key、JWT Token、开发模式
- 安全获取 user_id，防止用户伪造
- 统一的认证配置管理

认证方式优先级：
1. API Key（Header: X-API-Key）- 服务间调用
2. JWT Token（Header: Authorization）- 前端用户
3. 开发模式（请求体 userId）- 仅开发环境
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from fastapi import HTTPException

logger = logging.getLogger(__name__)


@dataclass
class AuthConfig:
    """
    认证配置类。

    Attributes:
        dev_mode: 是否启用开发模式（允许请求体 userId）
        api_keys: 有效的 API Key 列表
        jwt_secret: JWT 密钥
        jwt_algorithm: JWT 加密算法
    """

    dev_mode: bool = True
    api_keys: list = None
    jwt_secret: str = None
    jwt_algorithm: str = "HS256"

    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = []
        if self.jwt_secret is None:
            self.jwt_secret = os.getenv(
                "JWT_SECRET", "dev-secret-key-please-change-in-production"
            )


_auth_config: Optional[AuthConfig] = None


def get_auth_config() -> AuthConfig:
    """
    获取全局认证配置实例。

    Returns:
        AuthConfig: 认证配置实例
    """
    global _auth_config
    if _auth_config is None:
        dev_mode = os.getenv("AUTH_DEV_MODE", "true").lower() == "true"
        api_keys_str = os.getenv("AUTH_API_KEYS", "")
        api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

        _auth_config = AuthConfig(
            dev_mode=dev_mode,
            api_keys=api_keys,
        )
    return _auth_config


def _validate_api_key(x_api_key: Optional[str]) -> Optional[str]:
    """
    验证 API Key 并返回对应的 user_id。

    安全策略：
    1. 白名单验证：只在 config.api_keys 列表中的 key 才能通过
    2. 开发模式兼容：未配置 api_keys 时记录警告但允许 sk- 开头的 key（仅开发环境）
    3. 生产模式严格：必须配置白名单，否则拒绝所有请求

    Args:
        x_api_key: API Key 字符串

    Returns:
        Optional[str]: 验证成功返回 user_id，失败返回 None
    """
    if not x_api_key:
        return None

    config = get_auth_config()

    # 方案1：白名单验证（推荐，生产环境必须配置）
    if config.api_keys and x_api_key in config.api_keys:
        logger.info(f"API Key 白名单验证成功")
        return f"api_user_{hash(x_api_key) % 10000}"

    # 方案2：开发模式兼容（仅当未配置白名单时）
    if not config.api_keys and config.dev_mode:
        if x_api_key.startswith("sk-"):
            logger.warning(
                f"[安全警告] 使用开发模式API Key验证（未配置AUTH_API_KEYS环境变量）。"
                f"请在生产环境配置有效的API Key白名单！"
            )
            return f"api_user_dev_{x_api_key[3:11]}"

    # 所有验证均失败
    logger.warning(f"无效的 API Key: {x_api_key[:10]}... (已拒绝)")
    return None


def _validate_jwt_token(authorization: Optional[str]) -> Optional[str]:
    """
    验证 JWT Token 并返回对应的 user_id。

    Args:
        authorization: Authorization Header 值

    Returns:
        Optional[str]: 验证成功返回 user_id，失败返回 None
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        logger.warning("Authorization Header 格式错误，应为 'Bearer <token>'")
        return None

    token = authorization[7:]

    try:
        import jwt

        config = get_auth_config()
        payload = jwt.decode(
            token, config.jwt_secret, algorithms=[config.jwt_algorithm]
        )
        user_id = payload.get("user_id") or payload.get("sub")

        if user_id:
            return str(user_id)

        logger.warning("JWT Token 中未找到 user_id 或 sub 字段")
        return None

    except ImportError:
        logger.warning("JWT 库未安装，跳过 JWT 验证。请运行: pip install PyJWT")
        return None
    except jwt.ExpiredSignatureError:
        logger.warning("JWT Token 已过期")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"JWT Token 验证失败: {e}")
        return None
    except Exception as e:
        logger.error(f"JWT 验证异常: {e}", exc_info=True)
        return None


def _validate_dev_user_id(request_user_id: Optional[str]) -> Optional[str]:
    """
    验证开发模式下的 user_id。

    Args:
        request_user_id: 请求体中的 userId

    Returns:
        Optional[str]: 验证成功返回 user_id，失败返回 None
    """
    if not request_user_id:
        return None

    config = get_auth_config()

    if not config.dev_mode:
        logger.warning("生产环境不允许从请求体获取 user_id")
        return None

    if not isinstance(request_user_id, str) or len(request_user_id) == 0:
        logger.warning(f"无效的 userId 格式: {request_user_id}")
        return None

    return request_user_id


def get_current_user_id(
    x_api_key: Optional[str] = None,
    authorization: Optional[str] = None,
    request_user_id: Optional[str] = None,
) -> str:
    """
    安全获取当前用户的 user_id。

    认证优先级：
    1. API Key（Header: X-API-Key）
    2. JWT Token（Header: Authorization）
    3. 开发模式（请求体 userId）

    Args:
        x_api_key: API Key Header
        authorization: Authorization Header
        request_user_id: 请求体中的 userId（仅开发模式）

    Returns:
        str: 用户 ID

    Raises:
        HTTPException: 认证失败时抛出 401 错误

    Example:
        >>> user_id = get_current_user_id(
        ...     x_api_key="sk-12345678",
        ...     authorization=None,
        ...     request_user_id=None
        ... )
    """
    user_id = _validate_api_key(x_api_key)
    if user_id:
        logger.info(f"API Key 认证成功: {user_id}")
        return user_id

    user_id = _validate_jwt_token(authorization)
    if user_id:
        logger.info(f"JWT Token 认证成功: {user_id}")
        return user_id

    user_id = _validate_dev_user_id(request_user_id)
    if user_id:
        logger.info(f"开发模式认证成功: {user_id}")
        return user_id

    logger.error("所有认证方式均失败")
    raise HTTPException(
        status_code=401,
        detail="未授权：请提供有效的 API Key、JWT Token 或在开发模式下提供 userId",
    )
