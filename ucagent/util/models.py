# -*- coding: utf-8 -*-
"""Model utilities for UCAgent chat models."""

from typing import Any, Optional
from .config import Config
from langchain_core.rate_limiters import InMemoryRateLimiter
from ucagent.util.log import echo_g


def get_chat_model_openai(cfg: Config, callbacks, rate_limiter, streaming: Optional[bool] = None) -> Any:
    """Get OpenAI chat model instance.

    Args:
        cfg: Configuration object containing OpenAI settings.

    Returns:
        ChatOpenAI instance.

    Raises:
        ImportError: If langchain_openai is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "Please install langchain_openai to use OpenAI chat model. "
            "You can install it with: pip3 install langchain_openai"
        )
    kw = cfg.openai.as_dict()
    model_name = kw.pop("model_name")
    if model_name:
        kw["model"] = model_name
    if "seed" not in kw:
        kw["seed"] = cfg.seed
    if callbacks:
        kw.update({"callbacks": callbacks})
    if streaming is not None:
        kw.update({"streaming": streaming})
    if rate_limiter:
        kw.update({"rate_limiter": rate_limiter})
    return ChatOpenAI(**kw)


def get_chat_model_anthropic(cfg: Config, callbacks, rate_limiter, streaming: Optional[bool] = None) -> Any:
    """Get Anthropic chat model instance.

    Args:
        cfg: Configuration object containing Anthropic settings.

    Returns:
        ChatAnthropic instance.

    Raises:
        ImportError: If langchain_anthropic is not installed.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "Please install langchain_anthropic to use Anthropic chat model. "
            "You can install it with: pip3 install langchain_anthropic"
        )
    kw = cfg.anthropic.as_dict()
    if rate_limiter:
        kw.update({"rate_limiter": rate_limiter})
    if callbacks:
        kw.update(
            {
                "callbacks": callbacks,
            }
        )
    if streaming is not None:
        kw.update({"streaming": streaming})
    llm = ChatAnthropic(**kw)
    return llm


def get_chat_model_google_genai(cfg: Config, callbacks, rate_limiter, streaming: Optional[bool] = None) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "Please install langchain_google_genai to use Google GenAI chat model. "
            "You can install it with: pip3 install langchain_google_genai"
        )
    kw = cfg.google_genai.as_dict()
    model_name = kw.pop("model_name")
    if model_name:
        kw["model"] = model_name
    if callbacks:
        kw.update(
            {
                "callbacks": callbacks,
            }
        )
    if streaming is not None:
        kw.update({"streaming": streaming})
    if rate_limiter:
        kw.update({"rate_limiter": rate_limiter})
    return ChatGoogleGenerativeAI(**kw)


def _merge_section_with_root(root_cfg: Config, section_name: Optional[str]) -> Config:
    """Build an effective model config from a root config and an optional section."""
    if not section_name:
        return root_cfg
    if not root_cfg.has_attr(section_name):
        raise AttributeError(f"Configuration does not have attribute '{section_name}'")
    section_cfg = getattr(root_cfg, section_name)
    if hasattr(section_cfg, "as_dict"):
        section_dict = section_cfg.as_dict()
    elif isinstance(section_cfg, dict):
        section_dict = dict(section_cfg)
    else:
        raise TypeError(f"Section '{section_name}' must be Config or dict, got {type(section_cfg)}")

    merged = root_cfg.as_dict()
    merged["model_type"] = section_dict.get("model_type", merged.get("model_type", "openai"))
    merged["seed"] = section_dict.get("seed", merged.get("seed"))

    for provider in ("openai", "anthropic", "google_genai", "rate_limiter"):
        base_provider = merged.get(provider, {})
        section_provider = section_dict.get(provider, {})
        if not isinstance(base_provider, dict):
            base_provider = {}
        if not isinstance(section_provider, dict):
            section_provider = {}
        merged[provider] = {**base_provider, **section_provider}

    for key, value in section_dict.items():
        if key in {"openai", "anthropic", "google_genai", "rate_limiter"}:
            continue
        merged[key] = value
    return Config(merged)


def get_chat_model_from_section(
    cfg: Config,
    section_name: Optional[str],
    callbacks: Any = None,
    streaming: Optional[bool] = None,
) -> Any:
    effective_cfg = _merge_section_with_root(cfg, section_name)
    if not effective_cfg.rate_limiter.enabled:
        rate_limiter = None
    else:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=effective_cfg.rate_limiter.requests_per_second,
            check_every_n_seconds=effective_cfg.rate_limiter.check_every_n_seconds,
            max_bucket_size=effective_cfg.rate_limiter.max_bucket_size,
        )
        echo_g(
            "Rate limiter enabled with %d requests per minute (RPM)."
            % (effective_cfg.rate_limiter.requests_per_second * 60)
        )
    model_type = effective_cfg.get_value("model_type", "openai")
    func = "get_chat_model_%s" % model_type
    if section_name:
        echo_g(f"Using model type: {model_type} in get_chat_model_from_section('{section_name}').")
    else:
        echo_g(f"Using model type: {model_type} in get_chat_model.")
    if func in globals():
        return globals()[func](effective_cfg, callbacks, rate_limiter, streaming=streaming)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types are: "
            f"{', '.join([ f.removeprefix('get_chat_model_') for f in globals().keys() if f.startswith('get_chat_model_') ])}."
        )


def get_chat_model(cfg: Config, callbacks: Any = None, streaming: Optional[bool] = None) -> Any:
    return get_chat_model_from_section(cfg, None, callbacks, streaming=streaming)
