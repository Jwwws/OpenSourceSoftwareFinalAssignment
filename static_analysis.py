import libcst as cst
import libcst.matchers as m
import os
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pysnooper
import sys


@dataclass
class CodeIssue:
    """代码问题数据类"""
    file_path: str
    line: int
    column: int
    issue_type: str
    message: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    suggestion: Optional[str] = None

    def to_dict(self):
        """转换为字典"""
        return asdict(self)


@dataclass
class RefactoringSummary:
    """重构统计汇总"""
    total_files_analyzed: int = 0
    files_with_refactorings: int = 0
    total_refactorings: int = 0
    refactorings_by_type: Dict[str, int] = None
    refactorings_by_file: Dict[str, int] = None
    modified_files: List[str] = None
    refactoring_details: List[Dict] = None

    def __post_init__(self):
        if self.refactorings_by_type is None:
            self.refactorings_by_type = {}
        if self.refactorings_by_file is None:
            self.refactorings_by_file = {}
        if self.modified_files is None:
            self.modified_files = []
        if self.refactoring_details is None:
            self.refactoring_details = []

    def add_refactoring(self, refactoring_type: str, file_path: str, line: int, description: str):
        """添加重构记录"""
        self.total_refactorings += 1
        self.refactorings_by_type[refactoring_type] = self.refactorings_by_type.get(refactoring_type, 0) + 1
        self.refactorings_by_file[file_path] = self.refactorings_by_file.get(file_path, 0) + 1
        self.refactoring_details.append({
            "file": file_path,
            "type": refactoring_type,
            "line": line,
            "description": description
        })

    def print_summary(self):
        """打印重构统计"""
        print("\n" + "=" * 80)
        print("代码重构统计报告")
        print("=" * 80)
        print(f"分析文件总数: {self.total_files_analyzed}")
        print(f"有重构的文件数: {self.files_with_refactorings}")
        print(f"总重构条数: {self.total_refactorings}")

        if self.modified_files:
            print(f"已修改文件数: {len(self.modified_files)}")
            if self.modified_files:
                print("\n已修改文件列表:")
                for i, file in enumerate(self.modified_files[:10], 1):  # 只显示前10个
                    print(f"  {i:2d}. {file}")
                if len(self.modified_files) > 10:
                    print(f"  ... 还有 {len(self.modified_files) - 10} 个文件")

        if self.refactorings_by_type:
            print("\n重构类型统计:")
            for ref_type, count in sorted(self.refactorings_by_type.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_refactorings * 100) if self.total_refactorings > 0 else 0
                print(f"  {ref_type:25s}: {count:4d} 次 ({percentage:5.1f}%)")

        if self.refactoring_details and len(self.refactoring_details) <= 20:
            print("\n重构详情:")
            for detail in self.refactoring_details:
                print(f"  {detail['file']}:{detail['line']:4d} - {detail['type']} - {detail['description']}")

        print("=" * 80)


class PandasPatternVisitor(cst.CSTVisitor):
    """扩展的代码模式访问器"""
    # 必须声明依赖，才能获取位置信息
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.issues: List[CodeIssue] = []

        self.stats = {
            "class_defs": 0,
            "func_defs": 0,
            "unsafe_eval_usage": 0,
            "docstring_missing": 0,
            "import_star_count": 0,
            "magic_methods": 0,
            "decorators": 0,
            "lambda_count": 0,
            "list_comprehensions": 0,
            "dict_comprehensions": 0,
            "set_comprehensions": 0,
            "generator_expressions": 0,
            "async_functions": 0,
            "global_vars": 0,
            "nonlocal_vars": 0,
            "nested_depths": [],
            "long_functions": 0,
            "long_classes": 0,
            "context_managers": 0,
            "ternary_expressions": 0,
            "assert_statements": 0,
            "raise_statements": 0,
            "try_except_blocks": 0,
            "with_statements": 0,
            "yield_expressions": 0,
            "return_statements": 0,
            "isinstance_calls": 0,
            "type_calls": 0,
        }

        self.current_function_depth = 0

    def _get_pos(self, node: cst.CSTNode):
        """安全获取节点行列号的辅助方法"""
        pos = self.get_metadata(cst.metadata.PositionProvider, node)
        return pos.start.line, pos.start.column

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if any(isinstance(item, cst.ImportStar) for item in node.names):
            line, col = self._get_pos(node)
            self.stats["import_star_count"] += 1
            self.issues.append(CodeIssue(
                file_path=self.file_path, line=line, column=col,
                issue_type="ImportStar", message="Avoid using 'from module import *'",
                severity="medium", suggestion="Explicitly import only what you need"
            ))

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.stats["class_defs"] += 1
        if not node.get_docstring():
            line, col = self._get_pos(node.name)
            self.issues.append(CodeIssue(
                file_path=self.file_path, line=line, column=col,
                issue_type="MissingDocstring", message="Class missing docstring",
                severity="low", suggestion="Add a class docstring"
            ))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.stats["func_defs"] += 1
        self.current_function_depth = 0
        if not node.get_docstring():
            self.stats["docstring_missing"] += 1
            line, col = self._get_pos(node.name)
            self.issues.append(CodeIssue(
                file_path=self.file_path, line=line, column=col,
                issue_type="MissingDocstring", message="Function missing docstring",
                severity="low", suggestion="Add a docstring describing the function"
            ))
        if node.asynchronous is not None:
            self.stats["async_functions"] += 1
        if node.decorators:
            self.stats["decorators"] += len(node.decorators)

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            func_name = node.func.value
            line, col = self._get_pos(node.func)

            if func_name in ["eval", "exec"]:
                self.stats["unsafe_eval_usage"] += 1
                self.issues.append(CodeIssue(
                    file_path=self.file_path, line=line, column=col,
                    issue_type="UnsafeEval", message=f"Use of unsafe function '{func_name}'",
                    severity="high", suggestion="Use ast.literal_eval() instead"
                ))
            elif func_name.startswith("__") and func_name.endswith("__"):
                self.stats["magic_methods"] += 1
            elif func_name == "isinstance":
                self.stats["isinstance_calls"] += 1
            elif func_name == "type":
                self.stats["type_calls"] += 1

    def visit_Lambda(self, node: cst.Lambda) -> None:
        self.stats["lambda_count"] += 1

    def visit_ListComp(self, node: cst.ListComp) -> None:
        self.stats["list_comprehensions"] += 1

    def visit_DictComp(self, node: cst.DictComp) -> None:
        self.stats["dict_comprehensions"] += 1

    def visit_SetComp(self, node: cst.SetComp) -> None:
        self.stats["set_comprehensions"] += 1

    def visit_GeneratorExp(self, node: cst.GeneratorExp) -> None:
        self.stats["generator_expressions"] += 1

    def visit_Global(self, node: cst.Global) -> None:
        self.stats["global_vars"] += len(node.names)
        line, col = self._get_pos(node)
        self.issues.append(CodeIssue(
            file_path=self.file_path, line=line, column=col,
            issue_type="GlobalVariable", message="Use of global variables",
            severity="medium", suggestion="Use parameters or class attributes"
        ))

    def visit_Nonlocal(self, node: cst.Nonlocal) -> None:
        self.stats["nonlocal_vars"] += len(node.names)

    def visit_If(self, node: cst.If) -> None:
        self.current_function_depth += 1
        self.stats["nested_depths"].append(self.current_function_depth)

    def leave_If(self, original_node: cst.If) -> None:
        self.current_function_depth -= 1

    def visit_For(self, node: cst.For) -> None:
        self.current_function_depth += 1
        self.stats["nested_depths"].append(self.current_function_depth)

    def leave_For(self, original_node: cst.For) -> None:
        self.current_function_depth -= 1

    def visit_While(self, node: cst.While) -> None:
        self.current_function_depth += 1
        self.stats["nested_depths"].append(self.current_function_depth)

    def leave_While(self, original_node: cst.While) -> None:
        self.current_function_depth -= 1

    def visit_Try(self, node: cst.Try) -> None:
        self.stats["try_except_blocks"] += 1

    def visit_With(self, node: cst.With) -> None:
        self.stats["with_statements"] += 1

    def visit_Assert(self, node: cst.Assert) -> None:
        self.stats["assert_statements"] += 1

    def visit_Raise(self, node: cst.Raise) -> None:
        self.stats["raise_statements"] += 1

    def visit_Return(self, node: cst.Return) -> None:
        self.stats["return_statements"] += 1

    def visit_Yield(self, node: cst.Yield) -> None:
        self.stats["yield_expressions"] += 1


class PandasCodeRefactorer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, file_path: str, refactoring_summary: RefactoringSummary, verbose: bool = False):
        super().__init__()
        self.file_path = file_path
        self.refactoring_summary = refactoring_summary
        self.refactor_count = 0
        self.verbose = verbose

        # 详细的修改统计
        self.refactor_stats = {
            "empty_pass_removed": 0,
            "isinstance_refactored": 0,
            "type_comparison_refactored": 0,
            "import_simplified": 0,
            "unused_vars_removed": 0,
        }

        self.changes: List[Dict] = []

    def _get_line(self, node: cst.CSTNode):
        return self.get_metadata(cst.metadata.PositionProvider, node).start.line