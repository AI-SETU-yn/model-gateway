"""Compatibility dependency exports for legacy imports."""

from app.core.config import get_chat_service, get_litellm_service, get_settings

__all__ = ['get_settings', 'get_litellm_service', 'get_chat_service']
