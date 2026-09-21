# -*- coding: utf-8 -*-
"""Version information for CO-UCAgent."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("co-ucagent")
except PackageNotFoundError:
    # Source-tree execution before installation.
    __version__ = "0.2.0.source-code"

__author__ = "Jiabao Wang; XS-MLVP"
__email__ = "unitychip@bosc.ac.cn"
__description__ = "Evidence-grounded context engineering for local-LLM chip verification agents"

banner = f"""
\u001b[34m   __  __   ______    ___                           __ \u001b[0m
\u001b[34m  / / / /  / ____/   /   |   ____ _  ___    ____   / /_\u001b[0m
\u001b[34m / / / /  / /       / /| |  / __ `/ / _ \\  / __ \\ / __/\u001b[0m
\u001b[34m/ /_/ /  / /___    / ___ | / /_/ / /  __/ / / / // /_ \u001b[0m
\u001b[34m\\____/   \\____/   /_/  |_| \\__, /  \\___/ /_/ /_/ \\__/\u001b[0m
\u001b[34m                          /____/                       \u001b[0m \u001b[36mv{__version__}\u001b[0m
"""
