import pandas as pd
import numpy as np
from z3 import (Solver, Int, Real, String, Bool, Distinct, If, And, Or, Not,
                sat, is_true, IntNumRef)
import traceback
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import os
import io


# 兼容性检查函数
def is_real_value(val):
    """检查是否为实数类型"""
    return hasattr(val, 'as_decimal') or hasattr(val, 'as_fraction')


class EnhancedZ3PandasFuzzer:
    """
    Z3模糊测试器：包含更复杂约束和更多Pandas函数测试
    """

    def __init__(self, max_rows: int = 15, max_cols: int = 6):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.bug_reports = []
        self.test_counter = 0
        self.complex_constraints_used = set()

        # 创建输出目录
        os.makedirs("enhanced_bug_reports", exist_ok=True)
        os.makedirs("enhanced_test_cases", exist_ok=True)

    def generate_complex_constraints(self) -> Tuple[pd.DataFrame, Dict]:
        """
        生成包含复杂Z3约束的DataFrame
        """
        solver = Solver()

        # 随机确定DataFrame大小（更灵活）
        num_rows = random.randint(2, self.max_rows)
        num_cols = random.randint(2, self.max_cols)

        # 存储所有变量和约束
        z3_vars = {}
        constraints = []

        # 1. 时间序列约束（生成日期时间列）
        if random.random() < 0.4:  # 40%概率生成时间序列
            time_col_idx = random.randint(0, num_cols - 1)
            constraints.append(f"TIME_SERIES_COL{time_col_idx}")

            # 创建时间序列变量（单调递增）
            time_values = []
            for r in range(num_rows):
                time_var = Real(f'time_row{r}')
                if r == 0:
                    solver.add(time_var >= 0)
                    solver.add(time_var <= 100)
                else:
                    # 确保时间单调递增（可能有不连续的间隔）
                    prev_var = Real(f'time_row{r - 1}')
                    solver.add(time_var >= prev_var)
                    solver.add(time_var <= prev_var + 10)
                z3_vars[(r, time_col_idx)] = (time_var, 'time')

        # 2. 分类数据约束（有限集合的值）
        if random.random() < 0.5:
            cat_col_idx = random.randint(0, num_cols - 1)
            cat_size = random.randint(2, 5)  # 分类数量

            # 确保分类列不是时间序列列
            if f"TIME_SERIES_COL{cat_col_idx}" not in constraints:
                constraints.append(f"CATEGORICAL_COL{cat_col_idx}_SIZE{cat_size}")

                # 创建分类变量约束
                for r in range(num_rows):
                    cat_var = Int(f'cat_row{r}_col{cat_col_idx}')
                    solver.add(cat_var >= 0)
                    solver.add(cat_var < cat_size)
                    z3_vars[(r, cat_col_idx)] = (cat_var, 'categorical')

        # 3. 数值分布约束
        if random.random() < 0.3:
            dist_col_idx = random.randint(0, num_cols - 1)
            dist_type = random.choice(['normal', 'uniform', 'skewed'])
            constraints.append(f"DISTRIBUTION_{dist_type}_COL{dist_col_idx}")

            for r in range(num_rows):
                if dist_type == 'normal':
                    # 近似正态分布
                    var = Real(f'norm_row{r}_col{dist_col_idx}')
                    solver.add(var >= -10)
                    solver.add(var <= 10)
                elif dist_type == 'skewed':
                    # 偏态分布
                    var = Real(f'skew_row{r}_col{dist_col_idx}')
                    if r < num_rows * 0.8:
                        solver.add(var >= 0)
                        solver.add(var <= 10)
                    else:
                        solver.add(var >= -10)
                        solver.add(var < 0)
                else:  # uniform
                    var = Real(f'unif_row{r}_col{dist_col_idx}')
                    solver.add(var >= -10)
                    solver.add(var <= 10)

                z3_vars[(r, dist_col_idx)] = (var, dist_type)

        # 4. 相关性约束（简化版）
        if num_cols >= 2 and random.random() < 0.4:
            col_a = random.randint(0, num_cols - 1)
            col_b = random.randint(0, num_cols - 1)
            while col_b == col_a:
                col_b = random.randint(0, num_cols - 1)

            relation = random.choice(['linear', 'quadratic'])
            constraints.append(f"CORRELATION_{relation}_COL{col_a}_COL{col_b}")

            for r in range(num_rows):
                var_a = Real(f'corr_a_row{r}_col{col_a}')
                var_b = Real(f'corr_b_row{r}_col{col_b}')

                solver.add(var_a >= -10)
                solver.add(var_a <= 10)
                solver.add(var_b >= -10)
                solver.add(var_b <= 10)

                if relation == 'linear':
                    solver.add(var_b == var_a)
                elif relation == 'quadratic':
                    solver.add(var_b == var_a * var_a)

                z3_vars[(r, col_a)] = (var_a, 'correlation_a')
                z3_vars[(r, col_b)] = (var_b, 'correlation_b')

        # 填补剩余单元格
        for r in range(num_rows):
            for c in range(num_cols):
                if (r, c) not in z3_vars:
                    # 随机选择类型
                    cell_type = random.choice(['int', 'float', 'str'])

                    if cell_type == 'int':
                        var = Int(f'fill_int_row{r}_col{c}')
                        solver.add(var >= -100)
                        solver.add(var <= 100)
                    elif cell_type == 'float':
                        var = Real(f'fill_float_row{r}_col{c}')
                        solver.add(var >= -100.0)
                        solver.add(var <= 100.0)
                    else:  # str
                        var = String(f'fill_str_row{r}_col{c}')

                    z3_vars[(r, c)] = (var, cell_type)

        # 求解约束
        if solver.check() == sat:
            model = solver.model()
            return self._build_dataframe_from_model(z3_vars, num_rows, num_cols, model, constraints)
        else:
            return self._create_fallback_dataframe(), {'fallback': True}

    def _build_dataframe_from_model(self, z3_vars, num_rows, num_cols, model, constraints):
        """从Z3模型构建DataFrame"""
        data = {}
        column_info = {}

        # 按列收集数据
        for c in range(num_cols):
            col_data = []
            col_type = 'mixed'

            for r in range(num_rows):
                var_info = z3_vars.get((r, c))
                if not var_info:
                    col_data.append(None)
                    continue

                var, var_type = var_info

                try:
                    # 根据变量类型从模型中提取值
                    if var_type in ['int', 'categorical']:
                        val = model[var]
                        if hasattr(val, 'as_long'):
                            col_data.append(val.as_long())
                        else:
                            col_data.append(random.randint(-10, 10))

                    elif var_type in ['float', 'time', 'normal', 'uniform', 'skewed',
                                      'correlation_a', 'correlation_b']:
                        val = model[var]
                        # 使用兼容性函数检查实数类型
                        if is_real_value(val) or hasattr(val, 'as_fraction'):
                            try:
                                # 尝试转换为浮点数
                                col_data.append(float(str(val)))
                            except:
                                col_data.append(random.uniform(-10, 10))
                        else:
                            col_data.append(random.uniform(-10, 10))

                    elif var_type == 'str':
                        val = str(model[var])
                        # 清理Z3字符串表示
                        if 'String("' in val:
                            val = val.split('"')[1]
                        elif val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        col_data.append(val[:20])

                    else:
                        col_data.append(self._get_default_value(var_type))

                except Exception as e:
                    print(f"处理变量时出错: {var}, 类型: {var_type}, 错误: {e}")
                    col_data.append(self._get_default_value(var_type))

            # 应用后处理（根据约束）
            col_data = self._apply_constraint_postprocessing(c, col_data, constraints)

            # 确定列类型
            if all(isinstance(x, (int, np.integer)) for x in col_data if x is not None):
                col_type = 'int'
            elif all(isinstance(x, (float, np.floating)) for x in col_data if x is not None):
                col_type = 'float'
            elif all(isinstance(x, str) for x in col_data if x is not None):
                col_type = 'str'

            # 创建列名
            type_suffix = col_type
            for constr in constraints:
                if f"COL{c}" in constr:
                    type_suffix = constr.split('_')[0].lower()
                    break

            col_name = f"{type_suffix}_col{c}"
            data[col_name] = col_data
            column_info[col_name] = {'type': col_type, 'constraints': []}

            # 记录该列的约束
            for constr in constraints:
                if f"COL{c}" in constr:
                    column_info[col_name]['constraints'].append(constr)

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 应用全局后处理
        df = self._apply_global_postprocessing(df, constraints)

        metadata = {
            'num_rows': num_rows,
            'num_cols': num_cols,
            'constraints': constraints,
            'column_info': column_info,
            'generated_at': datetime.now().isoformat(),
            'complex_constraints': len([c for c in constraints if any(keyword in c for keyword in
                                                                      ['TIME', 'CATEGORICAL', 'DISTRIBUTION',
                                                                       'CORRELATION', 'PERIODICITY'])])
        }

        return df, metadata

    def _apply_constraint_postprocessing(self, col_idx, col_data, constraints):
        """根据约束对列数据进行后处理"""
        result = col_data.copy()

        for constr in constraints:
            if f"COL{col_idx}" in constr:
                # 处理时间序列
                if "TIME_SERIES" in constr:
                    base_date = datetime(2020, 1, 1)
                    for i in range(len(result)):
                        if isinstance(result[i], (int, float)):
                            try:
                                result[i] = base_date + timedelta(days=int(result[i]))
                            except:
                                result[i] = base_date + timedelta(days=i)

                # 处理分类数据
                elif "CATEGORICAL" in constr:
                    # 提取分类数量
                    for c in constraints:
                        if f"CATEGORICAL_COL{col_idx}" in c:
                            try:
                                size = int(c.split('SIZE')[1])
                                categories = [f'Cat_{chr(65 + i)}' for i in range(size)]
                                for i in range(len(result)):
                                    if isinstance(result[i], (int, np.integer)):
                                        idx = result[i] % size
                                        result[i] = categories[idx]
                            except:
                                pass

                # 注入特殊值
                if random.random() < 0.3 and len(result) > 0:
                    special_idx = random.randint(0, len(result) - 1)
                    special_type = random.choice(['nan', 'inf', '-inf', 'none', 'empty'])

                    if special_type == 'nan':
                        result[special_idx] = np.nan
                    elif special_type == 'inf':
                        result[special_idx] = np.inf
                    elif special_type == '-inf':
                        result[special_idx] = -np.inf
                    elif special_type == 'none':
                        result[special_idx] = None
                    elif special_type == 'empty' and isinstance(result[special_idx], str):
                        result[special_idx] = ""

        return result

    def _apply_global_postprocessing(self, df, constraints):
        """应用全局后处理"""
        # 设置索引
        if random.random() < 0.3 and len(df) > 1:
            # 创建重复索引
            indices = list(range(len(df)))
            indices[1] = indices[0]
            df.index = indices

        # 添加缺失值
        if random.random() < 0.2:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    for i in range(len(df)):
                        if i % 3 == 0:
                            df.iloc[i, df.columns.get_loc(col)] = np.nan

        return df

    def _get_default_value(self, dtype: str) -> Any:
        """获取默认值"""
        defaults = {
            'int': 0,
            'float': 0.0,
            'str': '',
            'time': datetime.now(),
            'categorical': 'A',
            'normal': 0.0,
            'uniform': 0.0,
            'skewed': 0.0,
            'correlation_a': 0.0,
            'correlation_b': 0.0,
        }
        return defaults.get(dtype, 0)

    def _create_fallback_dataframe(self) -> pd.DataFrame:
        """创建回退DataFrame"""
        data = {
            'int_col': [1, 2, 3, np.nan, 5],
            'float_col': [1.1, np.inf, 3.3, 4.4, -np.inf],
            'str_col': ['A', 'B', '', 'D', 'E'],
            'bool_col': [True, False, True, np.nan, False]
        }
        return pd.DataFrame(data)

    def test_extended_pandas_functions(self, df: pd.DataFrame, metadata: Dict) -> Dict:
        """
        测试更多Pandas函数
        """
        self.test_counter += 1
        test_id = f"enhanced_test_{self.test_counter:04d}"

        result = {
            'test_id': test_id,
            'metadata': metadata,
            'operations_tested': [],
            'operations_passed': [],
            'operations_failed': [],
            'errors': [],
            'crashes': [],
            'start_time': datetime.now().isoformat()
        }

        try:
            # === 1. 基本数据操作 ===
            self._safe_operation(lambda: df.describe(), "describe", result)

            if len(df.columns) > 0:
                self._safe_operation(lambda: df.sort_values(by=df.columns[0]), "sort_values", result)

            # === 2. 分组操作 ===
            group_cols = []
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
                    if len(df[col].unique()) < len(df) and len(df[col].unique()) > 0:
                        group_cols.append(col)

            for group_col in group_cols[:2]:
                self._safe_operation(
                    lambda col=group_col: df.groupby(col).mean(),
                    f"groupby_mean_{col}", result
                )

            # === 3. 数据透视 ===
            if len(df.columns) >= 3:
                str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                if len(str_cols) >= 2 and len(num_cols) >= 1:
                    self._safe_operation(
                        lambda: df.pivot_table(
                            index=str_cols[0],
                            columns=str_cols[1],
                            values=num_cols[0],
                            aggfunc='mean'
                        ),
                        "pivot_table", result
                    )

            # === 4. 窗口函数 ===
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols and len(df) > 2:
                num_col = num_cols[0]
                self._safe_operation(
                    lambda: df[num_col].rolling(window=2).mean(),
                    "rolling_mean", result
                )

                self._safe_operation(
                    lambda: df[num_col].ewm(span=2).mean(),
                    "ewm_mean", result
                )

            # === 5. 数据清理 ===
            self._safe_operation(lambda: df.dropna(), "dropna", result)
            self._safe_operation(lambda: df.fillna(0), "fillna", result)

            # === 6. 类型转换 ===
            self._safe_operation(lambda: df.convert_dtypes(), "convert_dtypes", result)

            # === 7. 序列化 ===
            self._safe_operation(
                lambda: pd.read_json(io.StringIO(df.to_json())),
                "json_roundtrip", result
            )

            # === 8. 字符串操作 ===
            str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
            for str_col in str_cols[:1]:
                if len(df) > 0 and df[str_col].notna().any():
                    self._safe_operation(
                        lambda col=str_col: df[col].str.upper(),
                        f"str_upper_{col}", result
                    )

            # === 9. 合并与连接 ===
            if len(df) > 1:
                df2 = df.copy()
                df2.columns = [f"{col}_2" for col in df2.columns]

                self._safe_operation(
                    lambda: pd.merge(df, df2, left_index=True, right_index=True),
                    "merge", result
                )

                self._safe_operation(
                    lambda: pd.concat([df, df2], axis=0),
                    "concat_axis0", result
                )

            # === 10. 高级分组 ===
            for group_col in group_cols[:1]:
                self._safe_operation(
                    lambda col=group_col: df.groupby(col).agg(['mean', 'sum', 'count']),
                    f"groupby_agg_{col}", result
                )

                self._safe_operation(
                    lambda col=group_col: df.groupby(col).transform('mean'),
                    f"groupby_transform_{col}", result
                )

            result['end_time'] = datetime.now().isoformat()
            result['success'] = len(result['crashes']) == 0

        except Exception as e:
            crash_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'dataframe_shape': df.shape,
            }
            result['crashes'].append(crash_info)
            result['success'] = False

        return result

    def _safe_operation(self, operation_func, operation_name: str, result: Dict):
        """安全执行操作并记录结果"""
        try:
            output = operation_func()
            result['operations_passed'].append(operation_name)
            result['operations_tested'].append(operation_name)

        except (ValueError, TypeError, KeyError, IndexError,
                pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            # 预期的业务逻辑错误
            result['operations_failed'].append({
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e)[:200]
            })
            result['operations_tested'].append(operation_name)

        except Exception as e:
            # 未预期的异常，可能是BUG！
            crash_info = {
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            result['crashes'].append(crash_info)
            result['operations_tested'].append(operation_name)

            # 记录潜在BUG
            self._record_potential_bug(operation_name, e, result.get('test_id', 'unknown'))

    def _record_potential_bug(self, operation: str, error: Exception, test_id: str):
        """记录潜在BUG"""
        bug_report = {
            'test_id': test_id,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
        }

        self.bug_reports.append(bug_report)

        bug_file = f"enhanced_bug_reports/bug_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(bug_file, 'w') as f:
            json.dump(bug_report, f, indent=2)

        print(f"发现潜在BUG! 操作: {operation}, 错误: {type(error).__name__}")
        print(f"已保存到: {bug_file}")

    def save_test_case(self, df: pd.DataFrame, metadata: Dict, result: Dict):
        """保存测试用例"""
        # 简化DataFrame以便保存
        df_simplified = df.copy()
        for col in df_simplified.columns:
            if pd.api.types.is_datetime64_any_dtype(df_simplified[col]):
                df_simplified[col] = df_simplified[col].astype(str)

        test_case = {
            'dataframe': df_simplified.head(10).to_dict(),  # 只保存前10行
            'metadata': metadata,
            'result': {
                'test_id': result.get('test_id'),
                'operations_tested': result.get('operations_tested', []),
                'operations_passed': result.get('operations_passed', []),
                'crashes_count': len(result.get('crashes', [])),
                'success': result.get('success', False)
            }
        }

        test_file = f"enhanced_test_cases/{result.get('test_id', 'unknown')}.json"
        with open(test_file, 'w') as f:
            json.dump(test_case, f, indent=2, default=str)

    def run_enhanced_fuzzing(self, num_tests: int = 50):
        """运行增强版模糊测试"""
        print("=" * 80)
        print("增强版Z3约束模糊测试启动")
        print(f"Pandas版本: {pd.__version__}")
        print(f"Z3版本: 兼容性修复版")
        print(f"测试数量: {num_tests}")
        print("=" * 80)

        stats = {
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'crashes_found': 0,
            'potential_bugs': 0,
            'operations_tested_total': 0,
            'operations_passed_total': 0,
            'start_time': datetime.now().isoformat()
        }

        for i in range(num_tests):
            print(f"\n[{i + 1}/{num_tests}] 生成测试用例...")

            # 生成DataFrame
            df, metadata = self.generate_complex_constraints()
            complex_count = metadata.get('complex_constraints', 0)

            print(f"  形状: {df.shape}, 约束: {len(metadata.get('constraints', []))}")

            # 运行测试
            result = self.test_extended_pandas_functions(df, metadata)

            # 更新统计
            stats['total_tests'] += 1
            stats['operations_tested_total'] += len(result.get('operations_tested', []))
            stats['operations_passed_total'] += len(result.get('operations_passed', []))

            if result.get('success', False):
                stats['tests_passed'] += 1
                passed = len(result.get('operations_passed', []))
                failed = len(result.get('operations_failed', []))
                print(f"通过 (操作: {passed}通过/{failed}失败)")
            else:
                stats['tests_failed'] += 1
                crashes = len(result.get('crashes', []))
                if crashes > 0:
                    stats['crashes_found'] += crashes
                    stats['potential_bugs'] += 1
                    print(f"失败 (发现{crashes}个崩溃)")
                else:
                    print(f"失败 (预期错误)")

            # 保存测试用例
            self.save_test_case(df, metadata, result)

        # 完成统计
        stats['end_time'] = datetime.now().isoformat()
        stats['bug_reports_count'] = len(self.bug_reports)
        stats['avg_operations_per_test'] = stats['operations_tested_total'] / max(1, stats['total_tests'])
        stats['success_rate'] = stats['tests_passed'] / max(1, stats['total_tests'])

        # 保存统计
        stats_file = "enhanced_bug_reports/enhanced_fuzzing_summary.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # 打印摘要
        print("\n" + "=" * 80)
        print("增强版模糊测试完成")
        print("=" * 80)
        print(f"总测试数: {stats['total_tests']}")
        print(f"通过: {stats['tests_passed']} (成功率: {stats['success_rate']:.1%})")
        print(f"失败: {stats['tests_failed']}")
        print(f"发现崩溃: {stats['crashes_found']}")
        print(f"潜在BUG数: {stats['potential_bugs']}")
        print(f"详细报告: {stats_file}")

        if self.bug_reports:
            print("\n发现的潜在BUG:")
            for i, bug in enumerate(self.bug_reports[:5], 1):
                print(f"  {i}. {bug['test_id']}: {bug['operation']} -> {bug['error_type']}")
            if len(self.bug_reports) > 5:
                print(f"  ... 还有{len(self.bug_reports) - 5}个")

        return stats


def main():
    """主函数"""
    fuzzer = EnhancedZ3PandasFuzzer(max_rows=12, max_cols=5)
    stats = fuzzer.run_enhanced_fuzzing(num_tests=5000)



if __name__ == "__main__":
    main()