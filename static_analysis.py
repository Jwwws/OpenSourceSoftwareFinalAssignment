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

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # 检查是否为只有pass的空函数
        if m.matches(updated_node.body, m.IndentedBlock(body=[m.SimpleStatementLine(body=[m.Pass()])])):
            self.refactor_stats["empty_pass_removed"] += 1
            self.refactor_count += 1

            line = self._get_line(original_node)
            description = f"移除空函数 '{original_node.name.value}' (line {line})"

            self.changes.append({
                "type": "empty_pass_removed",
                "line": line,
                "function": original_node.name.value,
                "description": description
            })

            if self.verbose:
                print(f"  [重构] {description}")

            # 记录到总统计
            self.refactoring_summary.add_refactoring(
                "empty_pass_removed",
                self.file_path,
                line,
                f"移除了空函数: {original_node.name.value}"
            )

            # 实际上我们返回原节点，因为完全移除函数可能会破坏代码
            # 这里我们保留函数但移除pass语句
            return updated_node

        # 检查函数体是否只有docstring和pass
        if len(updated_node.body.body) == 1:
            stmt = updated_node.body.body[0]
            if isinstance(stmt, cst.SimpleStatementLine):
                body_item = stmt.body[0] if stmt.body else None
                if isinstance(body_item, cst.Pass):
                    # 函数只有pass，但没有docstring
                    self.refactor_stats["empty_pass_removed"] += 1
                    self.refactor_count += 1

                    line = self._get_line(original_node)
                    description = f"优化只有pass的函数 '{original_node.name.value}' (line {line})"

                    self.changes.append({
                        "type": "empty_pass_removed",
                        "line": line,
                        "function": original_node.name.value,
                        "description": description
                    })

                    if self.verbose:
                        print(f"  [重构] {description}")

                    self.refactoring_summary.add_refactoring(
                        "empty_pass_removed",
                        self.file_path,
                        line,
                        f"优化了只有pass的函数: {original_node.name.value}"
                    )

        return updated_node

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.Comparison:
        left = original_node.left

        # 检查是否为 type(obj) == SomeClass 或 type(obj) != SomeClass
        if isinstance(left, cst.Call) and isinstance(left.func, cst.Name) and left.func.value == "type":
            if len(left.args) > 0:
                for comparator in original_node.comparisons:
                    operator = comparator.operator
                    right = comparator.comparator
                    obj = left.args[0].value

                    line = self._get_line(original_node)

                    # 处理相等比较
                    if isinstance(operator, cst.Equal):
                        self.refactor_stats["type_comparison_refactored"] += 1
                        self.refactor_count += 1

                        description = f"将 type(...) == 转换为 isinstance(...) (line {line})"

                        self.changes.append({
                            "type": "type_comparison_refactored",
                            "line": line,
                            "description": description
                        })

                        if self.verbose:
                            print(f"  [重构] {description}")

                        self.refactoring_summary.add_refactoring(
                            "type_comparison_refactored",
                            self.file_path,
                            line,
                            "将 type(obj) == Class 转换为 isinstance(obj, Class)"
                        )

                        # 创建新的isinstance调用
                        new_call = cst.Call(
                            func=cst.Name("isinstance"),
                            args=[cst.Arg(value=obj), cst.Arg(value=right)]
                        )

                        return new_call

                    # 处理不等比较
                    elif isinstance(operator, cst.NotEqual):
                        self.refactor_stats["type_comparison_refactored"] += 1
                        self.refactor_count += 1

                        description = f"将 type(...) != 转换为 not isinstance(...) (line {line})"

                        self.changes.append({
                            "type": "type_comparison_refactored",
                            "line": line,
                            "description": description
                        })

                        if self.verbose:
                            print(f"  [重构] {description}")

                        self.refactoring_summary.add_refactoring(
                            "type_comparison_refactored",
                            self.file_path,
                            line,
                            "将 type(obj) != Class 转换为 not isinstance(obj, Class)"
                        )

                        # 创建not isinstance调用
                        new_call = cst.Call(
                            func=cst.Name("isinstance"),
                            args=[cst.Arg(value=obj), cst.Arg(value=right)]
                        )
                        new_not_call = cst.UnaryOperation(operator=cst.Not(), expression=new_call)

                        return new_not_call

        return updated_node

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.Import:
        # 简化重复的导入语句（示例规则）
        # 这里只是示例，实际实现需要更复杂的逻辑
        return updated_node


# 添加pysnooper装饰器用于调试
@pysnooper.snoop(output="./code_analysis.log", depth=2)
def analyze_source_code(target_dir, apply_fixes=False, verbose=False):
    target_path = Path(target_dir).resolve()
    print(f"正在检查绝对路径: {target_path}")

    if not target_path.exists():
        print(f"错误: 路径 {target_path} 不存在！")
        return

    total_stats = {
        "class_defs": 0, "func_defs": 0, "unsafe_eval_usage": 0, "docstring_missing": 0,
        "import_star_count": 0, "magic_methods": 0, "decorators": 0, "lambda_count": 0,
        "list_comprehensions": 0, "dict_comprehensions": 0, "set_comprehensions": 0,
        "generator_expressions": 0, "async_functions": 0, "global_vars": 0,
        "nonlocal_vars": 0, "nested_depths": [], "isinstance_calls": 0, "type_calls": 0,
        "try_except_blocks": 0, "with_statements": 0, "assert_statements": 0,
        "raise_statements": 0, "return_statements": 0, "yield_expressions": 0,
    }

    total_issues = []
    file_count = 0

    # 创建重构统计汇总
    refactoring_summary = RefactoringSummary()
    modified_files = []

    py_files = list(target_path.rglob("*.py"))
    print(f"找到待处理的 Python 文件总数: {len(py_files)}")

    for file_path in py_files:
        # 跳过测试目录和缓存目录
        if any(x in str(file_path) for x in ["tests", "__pycache__", ".pyc"]) or file_path.name.startswith("_"):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            if verbose:
                print(f"\n分析文件: {file_path}")

            # 使用 MetadataWrapper 包装
            wrapper = cst.metadata.MetadataWrapper(cst.parse_module(code))

            # 1. 分析
            visitor = PandasPatternVisitor(str(file_path))
            wrapper.visit(visitor)

            # 合并数据
            for k in total_stats:
                if k in visitor.stats:
                    if isinstance(visitor.stats[k], list):
                        total_stats[k].extend(visitor.stats[k])
                    else:
                        total_stats[k] += visitor.stats[k]
            total_issues.extend(visitor.issues)

            # 2. 重构
            transformer = PandasCodeRefactorer(str(file_path), refactoring_summary, verbose=verbose)
            modified_tree = wrapper.visit(transformer)

            if transformer.refactor_count > 0:
                refactoring_summary.files_with_refactorings += 1

                if verbose:
                    print(f"  在 {file_path} 中发现 {transformer.refactor_count} 处可重构")
                    for change in transformer.changes:
                        print(f"    - {change['type']}: {change.get('description', '')}")

                # 如果设置了应用修复，则实际写入文件
                if apply_fixes:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(modified_tree.code)
                        modified_files.append(str(file_path))
                        if verbose:
                            print(f"  已修改文件: {file_path}")
                    except Exception as e:
                        print(f"  写入文件失败 {file_path}: {e}")

            file_count += 1
            if file_count % 50 == 0:
                print(f"已分析 {file_count} 个文件...")

        except Exception as e:
            if verbose:
                print(f"无法解析文件 {file_path.name}: {e}")
            continue

    # 更新重构统计汇总
    refactoring_summary.total_files_analyzed = file_count
    refactoring_summary.modified_files = modified_files if apply_fixes else []

    # 输出结果
    avg_depth = statistics.mean(total_stats["nested_depths"]) if total_stats["nested_depths"] else 0

    print("\n" + "=" * 60)
    print("代码分析完成!")
    print("=" * 60)
    print(f"有效分析文件: {file_count}")
    print(f"类定义: {total_stats['class_defs']}")
    print(f"函数定义: {total_stats['func_defs']}")
    print(f"平均嵌套深度: {avg_depth:.2f}")
    print(f"发现的问题数: {len(total_issues)}")

    # 按严重程度统计问题
    if total_issues:
        severity_counts = Counter(issue.severity for issue in total_issues)
        print("\n问题严重程度统计:")
        for severity in ["critical", "high", "medium", "low"]:
            if severity in severity_counts:
                print(f"  {severity}: {severity_counts[severity]} 个")

    # 打印重构统计
    refactoring_summary.print_summary()

    # 将统计数据保存到JSON文件
    output_data = {
        "stats": total_stats,
        "issues": [issue.to_dict() for issue in total_issues],
        "file_count": file_count,
        "refactoring_summary": {
            "total_files_analyzed": refactoring_summary.total_files_analyzed,
            "files_with_refactorings": refactoring_summary.files_with_refactorings,
            "total_refactorings": refactoring_summary.total_refactorings,
            "refactorings_by_type": refactoring_summary.refactorings_by_type,
            "refactorings_by_file": refactoring_summary.refactorings_by_file,
            "modified_files": refactoring_summary.modified_files,
            "refactoring_details": refactoring_summary.refactoring_details[:100]  # 只保存前100条详情
        }
    }

    # 保存到文件
    output_file = Path("./code_analysis_report.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存到: {output_file}")

    return {
        "stats": total_stats,
        "issues": total_issues,
        "file_count": file_count,
        "refactoring_summary": refactoring_summary
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python代码分析和重构工具")
    parser.add_argument("target", nargs="?", default="./pandas", help="目标目录或文件")
    parser.add_argument("--apply-fixes", action="store_true", help="应用重构修改")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument("--output", type=str, default="./code_analysis_report.json", help="输出报告文件路径")
    args = parser.parse_args()

    # 执行分析
    result = analyze_source_code(args.target, apply_fixes=args.apply_fixes, verbose=args.verbose)

    # 输出总结信息
    if result:
        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"总分析文件: {result['file_count']}")
        print(f"总重构条数: {result['refactoring_summary'].total_refactorings}")
        print(f"已修改文件: {len(result['refactoring_summary'].modified_files)}")
        print("=" * 60)