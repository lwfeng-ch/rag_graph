"""
数据库连接池管理模块

功能：
- 管理 PostgreSQL 连接池
- 提供连接获取/释放接口
- 支持上下文管理器
- 自动重连和健康检查

依赖：
- psycopg_pool: PostgreSQL 连接池
- 可选依赖，未安装时降级为单连接模式
"""

import logging
import threading
from typing import Optional, Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_pool_instance: Optional[Any] = None
_pool_lock = threading.Lock()


def _check_psycopg_pool() -> bool:
    """
    检查 psycopg_pool 是否可用。

    Returns:
        bool: 是否可用
    """
    try:
        from psycopg_pool import ConnectionPool
        return True
    except ImportError:
        logger.warning(
            "psycopg_pool 未安装，连接池功能将降级为单连接模式。"
            "建议执行: pip install psycopg-pool"
        )
        return False


class DatabaseConnectionPool:
    """
    数据库连接池管理器。

    支持：
    - 连接池管理
    - 自动重连
    - 健康检查
    - 上下文管理器

    Example:
        >>> pool = DatabaseConnectionPool.get_instance()
        >>> with pool.connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT 1")
    """

    def __init__(
        self,
        db_uri: str,
        min_size: int = 2,
        max_size: int = 20,
        open_timeout: float = 30.0,
    ):
        """
        初始化连接池。

        Args:
            db_uri: 数据库连接 URI
            min_size: 最小连接数
            max_size: 最大连接数
            open_timeout: 打开超时时间（秒）
        """
        self.db_uri = db_uri
        self.min_size = min_size
        self.max_size = max_size
        self.open_timeout = open_timeout
        self._pool: Optional[Any] = None
        self._use_pool = _check_psycopg_pool()

        if self._use_pool:
            self._init_pool()

    def _init_pool(self) -> None:
        """初始化连接池。"""
        try:
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                self.db_uri,
                min_size=self.min_size,
                max_size=self.max_size,
                open_timeout=self.open_timeout,
                open=True,
            )
            logger.info(
                f"数据库连接池初始化成功: min={self.min_size}, max={self.max_size}"
            )
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            self._use_pool = False
            self._pool = None

    @contextmanager
    def connection(self):
        """
        获取数据库连接（上下文管理器）。

        Yields:
            数据库连接对象

        Raises:
            RuntimeError: 连接获取失败
        """
        if self._use_pool and self._pool:
            try:
                with self._pool.connection() as conn:
                    yield conn
            except Exception as e:
                logger.error(f"从连接池获取连接失败: {e}")
                raise RuntimeError(f"数据库连接获取失败: {e}")
        else:
            conn = self._get_fallback_connection()
            try:
                yield conn
            finally:
                if conn:
                    conn.close()

    def _get_fallback_connection(self):
        """
        获取备用连接（单连接模式）。

        Returns:
            数据库连接对象
        """
        try:
            import psycopg

            conn = psycopg.connect(self.db_uri)
            logger.debug("使用单连接模式获取数据库连接")
            return conn
        except ImportError:
            logger.error("psycopg 未安装，无法连接数据库")
            raise RuntimeError("psycopg 未安装，请执行: pip install psycopg")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise RuntimeError(f"数据库连接失败: {e}")

    def check_health(self) -> bool:
        """
        检查连接池健康状态。

        Returns:
            bool: 连接池是否健康
        """
        if not self._use_pool or not self._pool:
            return False

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"连接池健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接池。"""
        if self._pool:
            try:
                self._pool.close()
                logger.info("数据库连接池已关闭")
            except Exception as e:
                logger.warning(f"关闭连接池时出错: {e}")
            finally:
                self._pool = None

    def get_stats(self) -> dict:
        """
        获取连接池统计信息。

        Returns:
            dict: 统计信息
        """
        if not self._use_pool or not self._pool:
            return {
                "mode": "fallback",
                "pool_enabled": False,
            }

        try:
            return {
                "mode": "pool",
                "pool_enabled": True,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "health": self.check_health(),
            }
        except Exception as e:
            return {
                "mode": "pool",
                "pool_enabled": True,
                "error": str(e),
            }

    @classmethod
    def get_instance(
        cls,
        db_uri: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 20,
    ) -> "DatabaseConnectionPool":
        """
        获取连接池单例实例。

        Args:
            db_uri: 数据库连接 URI（首次调用时必需）
            min_size: 最小连接数
            max_size: 最大连接数

        Returns:
            DatabaseConnectionPool: 连接池实例
        """
        global _pool_instance

        with _pool_lock:
            if _pool_instance is None:
                if db_uri is None:
                    from utils.config import Config
                    db_uri = Config.DB_URI

                _pool_instance = cls(
                    db_uri=db_uri,
                    min_size=min_size,
                    max_size=max_size,
                )
            return _pool_instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置连接池单例实例。"""
        global _pool_instance

        with _pool_lock:
            if _pool_instance is not None:
                _pool_instance.close()
                _pool_instance = None


def get_db_connection():
    """
    获取数据库连接的便捷函数。

    Returns:
        数据库连接上下文管理器
    """
    pool = DatabaseConnectionPool.get_instance()
    return pool.connection()


def check_db_health() -> bool:
    """
    检查数据库健康状态。

    Returns:
        bool: 数据库是否健康
    """
    pool = DatabaseConnectionPool.get_instance()
    return pool.check_health()
