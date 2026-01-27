#!/usr/bin/env python3
import subprocess
import sys


TAGS = [
    "v0.3.1",
    "v0.3.0",
    "v0.2.3",
    "v0.2.2",
    "v0.2.1",
    "v0.1.3",
    "v0.1.2",
    "v0.1.1",
    "v0.1.0",
    "v0.0.2",
]


CHECK_SNIPPET = r"""
import inspect
import toffee
ok = True
print("toffee_file:", toffee.__file__)
print("toffee_version:", getattr(toffee, "__version__", "unknown"))
print("has_Dut:", hasattr(toffee, "Dut"))
print("has_Step:", hasattr(toffee, "Step"))
print("has_register_env:", hasattr(toffee, "register_env"))
try:
    print("Signal_sig:", inspect.signature(toffee.Signal))
except Exception as e:
    ok = False
    print("Signal_sig_error:", e)
try:
    _ = toffee.Signal(1)
    print("Signal(1): ok")
except Exception as e:
    ok = False
    print("Signal(1)_error:", e)
try:
    print("Signals_sig:", inspect.signature(toffee.Signals))
except Exception as e:
    ok = False
    print("Signals_sig_error:", e)
try:
    _ = toffee.Signals(32, 32, 32)
    print("Signals(32,32,32): ok")
except Exception as e:
    ok = False
    print("Signals(32,32,32)_error:", e)
print("CHECK_OK:", ok)
"""


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    for tag in TAGS:
        print("\n===", tag, "===")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-U",
                "--force-reinstall",
                f"pytoffee@git+https://github.com/XS-MLVP/toffee@{tag}",
            ]
        )
        run([sys.executable, "-c", CHECK_SNIPPET])


if __name__ == "__main__":
    main()
