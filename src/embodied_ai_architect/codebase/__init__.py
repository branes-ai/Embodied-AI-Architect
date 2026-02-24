"""Codebase analysis package for full-application hardware assessment.

Provides tools to scan, analyze, and convert arbitrary C++/Rust/Python
applications into workload profiles for the existing PPA pipeline.
"""

from embodied_ai_architect.codebase.models import (
    CodebaseAnalysisResult,
    ComputeKernel,
    DataflowLink,
    ScanResult,
    SourceFile,
)
from embodied_ai_architect.codebase.scanner import CodebaseScanner
from embodied_ai_architect.codebase.converter import CodebaseConverter

__all__ = [
    "CodebaseAnalysisResult",
    "CodebaseConverter",
    "CodebaseScanner",
    "ComputeKernel",
    "DataflowLink",
    "ScanResult",
    "SourceFile",
]
