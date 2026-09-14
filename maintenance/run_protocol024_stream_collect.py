"""Portable launcher for the immutable Protocol-023 resource guard helper.

Protocol-023's configure() uses psutil.BELOW_NORMAL_PRIORITY_CLASS, a Windows
constant.  The Protocol-024 Actions runner is Linux.  This compatibility shim
is installed before importing/calling the collector and changes no workload,
seed, timeline, label rule, or response-law numeric.
"""
import psutil

if not hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
    # POSIX nice +10 is the closest low-priority intent and is accepted by
    # psutil.Process.nice on an unprivileged GitHub-hosted Linux runner.
    psutil.BELOW_NORMAL_PRIORITY_CLASS = 10

from prepare_ftmoe_protocol024_stream import main


if __name__ == "__main__":
    main()
