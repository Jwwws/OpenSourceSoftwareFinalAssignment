import pandas as pd
from pydriller import Repository
import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import warnings
import json
from typing import Dict, List, Tuple, Optional
import hashlib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GitHistoryAnalyzer:
    """Git提交历史深度分析器"""

    def __init__(self, repo_path: str):
        """
        初始化分析器

        Args:
            repo_path: Git仓库路径（本地或GitHub URL）
        """
        self.repo_path = repo_path
        self.df = None
        self.bug_fixes_df = None
        self.non_bug_fixes_df = None
        self.analysis_results = {}

        # Bug相关关键词
        self.bug_keywords = [
            r'\bbug\b', r'\bfix\b', r'\bissue\b', r'\b#\d+\b',  # #1234格式的issue编号
            r'\bcrash\b', r'\berror\b', r'\bfail\b', r'\bpanic\b',
            r'\bdefect\b', r'\bproblem\b', r'\bpatch\b', r'\bresolve\b',
            r'\brepair\b', r'\bhotfix\b', r'\bbroken\b', r'\bregression\b',
            r'\b\bfault\b', r'\bexception\b', r'\bleak\b', r'\bhang\b',
            r'\bdeadlock\b', r'\bperformance\b', r'\bmemory\b', r'\bsecurity\b',
            r'\bcve\b', r'\bvulnerability\b', r'\bexploit\b'
        ]

        # 代码复杂度相关关键词
        self.complexity_keywords = [
            r'refactor', r'optimize', r'cleanup', r'simplify',
            r'improve', r'enhance', r'redesign', r'rework'
        ]

        # 文件类型分类
        self.file_categories = {
            'python': ['.py'],
            'cython': ['.pyx', '.pxd'],
            'c_cpp': ['.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'],
            'test': ['test_', '_test.py', '/test/', '/tests/'],
            'documentation': ['.md', '.rst', '.txt', '.tex'],
            'configuration': ['.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'],
            'build': ['setup.py', 'setup.cfg', 'pyproject.toml', 'Makefile', 'CMakeLists.txt'],
            'data': ['.csv', '.tsv', '.json', '.xml', '.yaml'],
            'notebook': ['.ipynb']
        }

    def extract_git_history(self, limit: int = 5000) -> pd.DataFrame:
        """
        提取Git提交历史

        Args:
            limit: 最大提交数量限制

        Returns:
            包含详细提交信息的DataFrame
        """
        print(f"开始提取 {self.repo_path} 的Git历史...")

        data = []
        repo = Repository(self.repo_path, order='reverse')

        count = 0
        for commit in repo.traverse_commits():
            if count >= limit:
                break

            # 分析提交消息
            msg_lower = commit.msg.lower()

            # 检测是否为bug修复
            is_bug_fix = False
            bug_pattern = None
            for keyword in self.bug_keywords:
                if re.search(keyword, msg_lower):
                    is_bug_fix = True
                    bug_pattern = keyword
                    break

            # 检测是否包含复杂度相关修改
            is_refactor = any(re.search(keyword, msg_lower) for keyword in self.complexity_keywords)

            # 分析修改的文件
            modified_files = []
            file_types = []
            file_categories = []

            for mod_file in commit.modified_files:
                filename = mod_file.filename
                modified_files.append(filename)

                # 判断文件类型
                file_ext = os.path.splitext(filename)[1]
                file_types.append(file_ext)

                # 判断文件类别
                category = 'other'
                for cat, patterns in self.file_categories.items():
                    for pattern in patterns:
                        if pattern.startswith('.') and filename.endswith(pattern):
                            category = cat
                            break
                        elif pattern in filename:
                            category = cat
                            break
                    if category != 'other':
                        break
                file_categories.append(category)

            # 统计各类文件数量
            file_type_counter = Counter(file_types)
            file_category_counter = Counter(file_categories)

            # 计算提交复杂度指标
            total_lines = commit.insertions + commit.deletions
            net_change = commit.insertions - commit.deletions
            churn_rate = total_lines / (commit.lines if commit.lines > 0 else 1)

            # 提取时间特征
            commit_date = commit.committer_date
            hour_of_day = commit_date.hour
            day_of_week = commit_date.weekday()  # 0=Monday, 6=Sunday
            month = commit_date.month
            is_weekend = day_of_week >= 5

            # 生成提交的"指纹" - 用于识别相似的提交
            commit_fingerprint = hashlib.md5(
                f"{commit.hash}{commit.msg}{commit.author.name}".encode()
            ).hexdigest()[:8]

            data.append({
                # 基础信息
                'commit_hash': commit.hash,
                'commit_fingerprint': commit_fingerprint,
                'author': commit.author.name,
                'author_email': commit.author.email,
                'commit_date': commit_date,
                'commit_message': commit.msg,
                'message_length': len(commit.msg),

                # Bug相关
                'is_bug_fix': is_bug_fix,
                'bug_pattern': bug_pattern,
                'is_refactor': is_refactor,

                # 文件修改统计
                'files_changed': len(modified_files),
                'modified_files': ','.join(modified_files[:10]),  # 只存储前10个文件
                'insertions': commit.insertions,
                'deletions': commit.deletions,
                'total_lines_changed': total_lines,
                'net_change': net_change,
                'churn_rate': churn_rate,

                # 文件类型统计
                'py_files': file_category_counter.get('python', 0),
                'cython_files': file_category_counter.get('cython', 0),
                'c_cpp_files': file_category_counter.get('c_cpp', 0),
                'test_files': file_category_counter.get('test', 0),
                'doc_files': file_category_counter.get('documentation', 0),
                'config_files': file_category_counter.get('configuration', 0),
                'build_files': file_category_counter.get('build', 0),

                # 时间特征
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'month': month,
                'is_weekend': is_weekend,
                'time_category': self._categorize_time(hour_of_day),

                # 其他
                'has_merge': commit.merge,
                'parents_count': len(commit.parents),
            })

            count += 1

            if count % 100 == 0:
                print(f"已处理 {count} 个提交...")

        self.df = pd.DataFrame(data)

        # 分离bug修复和非bug修复
        self.bug_fixes_df = self.df[self.df['is_bug_fix']].copy()
        self.non_bug_fixes_df = self.df[~self.df['is_bug_fix']].copy()

        print(f"提取完成！共处理 {len(self.df)} 个提交")
        print(f"其中 {len(self.bug_fixes_df)} 个是Bug修复提交 ({len(self.bug_fixes_df) / len(self.df) * 100:.1f}%)")

        return self.df

    def _categorize_time(self, hour: int) -> str:
        """将小时分类为时间段"""
        if 0 <= hour < 6:
            return '深夜 (0-6)'
        elif 6 <= hour < 12:
            return '上午 (6-12)'
        elif 12 <= hour < 18:
            return '下午 (12-18)'
        else:
            return '晚上 (18-24)'

    def analyze_bug_patterns(self) -> Dict:
        """分析Bug产生的模式和规律"""
        print("\n开始分析Bug产生模式...")

        results = {}

        if self.bug_fixes_df.empty:
            print("没有找到Bug修复提交")
            return results

        # 1. Bug修复的时间分布
        results['time_analysis'] = {
            'by_hour': self.bug_fixes_df['hour_of_day'].value_counts().sort_index().to_dict(),
            'by_day': self.bug_fixes_df['day_of_week'].value_counts().sort_index().to_dict(),
            'by_month': self.bug_fixes_df['month'].value_counts().sort_index().to_dict(),
            'by_time_category': self.bug_fixes_df['time_category'].value_counts().to_dict(),
            'weekend_vs_weekday': {
                'weekend': self.bug_fixes_df[self.bug_fixes_df['is_weekend']].shape[0],
                'weekday': self.bug_fixes_df[~self.bug_fixes_df['is_weekend']].shape[0]
            }
        }

        # 2. 作者分析
        author_stats = self.bug_fixes_df['author'].value_counts()
        results['author_analysis'] = {
            'top_bug_fixers': author_stats.head(10).to_dict(),
            'bug_fixes_per_author': len(self.bug_fixes_df) / len(self.df['author'].unique()),
            'author_concentration': author_stats.head(5).sum() / len(self.bug_fixes_df) * 100
        }

        # 3. 文件类型分析
        bug_file_types = []
        for files in self.bug_fixes_df['modified_files']:
            if files:
                bug_file_types.extend([os.path.splitext(f)[1] for f in files.split(',')])

        results['file_type_analysis'] = {
            'most_common_extensions': dict(Counter(bug_file_types).most_common(10)),
            'py_files_in_bug_fixes': self.bug_fixes_df['py_files'].mean(),
            'cython_files_in_bug_fixes': self.bug_fixes_df['cython_files'].mean(),
            'test_files_in_bug_fixes': self.bug_fixes_df['test_files'].mean(),
        }

        # 4. 提交大小分析
        results['commit_size_analysis'] = {
            'avg_files_in_bug_fix': self.bug_fixes_df['files_changed'].mean(),
            'avg_lines_in_bug_fix': self.bug_fixes_df['total_lines_changed'].mean(),
            'avg_insertions_in_bug_fix': self.bug_fixes_df['insertions'].mean(),
            'avg_deletions_in_bug_fix': self.bug_fixes_df['deletions'].mean(),
            'bug_fix_vs_non_bug_fix_size_ratio': (
                self.bug_fixes_df['total_lines_changed'].mean() /
                self.non_bug_fixes_df['total_lines_changed'].mean()
                if not self.non_bug_fixes_df.empty else 0
            )
        }

        # 5. Bug关键词分析
        bug_patterns = self.bug_fixes_df['bug_pattern'].dropna().value_counts()
        results['bug_keyword_analysis'] = {
            'most_common_keywords': bug_patterns.head(10).to_dict()
        }

        # 6. 高风险时间段识别
        bug_rate_by_hour = []
        for hour in range(24):
            hour_commits = self.df[self.df['hour_of_day'] == hour]
            if len(hour_commits) > 0:
                bug_rate = hour_commits['is_bug_fix'].mean()
                bug_rate_by_hour.append((hour, bug_rate, len(hour_commits)))

        bug_rate_by_hour.sort(key=lambda x: x[1], reverse=True)
        results['high_risk_periods'] = [
            {'hour': h, 'bug_rate': br, 'commit_count': cc}
            for h, br, cc in bug_rate_by_hour[:5]
        ]

        # 7. Bug修复的季节性模式
        seasonal_patterns = {}
        for month in range(1, 13):
            month_commits = self.df[self.df['month'] == month]
            if len(month_commits) > 10:  # 确保有足够的数据
                seasonal_patterns[month] = {
                    'bug_rate': month_commits['is_bug_fix'].mean(),
                    'total_commits': len(month_commits),
                    'bug_fixes': month_commits['is_bug_fix'].sum()
                }

        results['seasonal_patterns'] = seasonal_patterns

        self.analysis_results = results
        return results

    def identify_high_risk_patterns(self) -> Dict:
        """识别高风险模式（最容易产生Bug的情况）"""
        print("\n识别高风险模式...")

        patterns = {}

        # 1. 高风险文件类型
        file_risk_scores = {}
        for idx, row in self.df.iterrows():
            if row['modified_files']:
                files = row['modified_files'].split(',')
                for file in files:
                    ext = os.path.splitext(file)[1]
                    file_risk_scores.setdefault(ext, {'bug_count': 0, 'total_count': 0})
                    file_risk_scores[ext]['total_count'] += 1
                    if row['is_bug_fix']:
                        file_risk_scores[ext]['bug_count'] += 1

        # 计算风险分数
        risk_scores = {}
        for ext, counts in file_risk_scores.items():
            if counts['total_count'] > 5:  # 至少有5次修改
                risk_score = counts['bug_count'] / counts['total_count']
                risk_scores[ext] = {
                    'risk_score': risk_score,
                    'bug_count': counts['bug_count'],
                    'total_count': counts['total_count']
                }

        patterns['high_risk_file_types'] = dict(
            sorted(risk_scores.items(), key=lambda x: x[1]['risk_score'], reverse=True)[:10]
        )
        # 2. 高风险时间段
        time_risk = {}
        for time_cat in ['深夜 (0-6)', '上午 (6-12)', '下午 (12-18)', '晚上 (18-24)']:
            time_commits = self.df[self.df['time_category'] == time_cat]
            if len(time_commits) > 0:
                bug_rate = time_commits['is_bug_fix'].mean()
                time_risk[time_cat] = {
                    'bug_rate': bug_rate,
                    'total_commits': len(time_commits)
                }

        patterns['high_risk_times'] = time_risk

        # 3. 高风险提交特征
        patterns['high_risk_commit_characteristics'] = {
            'large_commits_bug_rate': self.df[self.df['total_lines_changed'] > 100]['is_bug_fix'].mean(),
            'small_commits_bug_rate': self.df[self.df['total_lines_changed'] <= 10]['is_bug_fix'].mean(),
            'multi_file_bug_rate': self.df[self.df['files_changed'] > 5]['is_bug_fix'].mean(),
            'single_file_bug_rate': self.df[self.df['files_changed'] == 1]['is_bug_fix'].mean(),
            'refactor_bug_rate': self.df[self.df['is_refactor']]['is_bug_fix'].mean(),
            'non_refactor_bug_rate': self.df[~self.df['is_refactor']]['is_bug_fix'].mean()
        }

        # 4. 高风险作者（引入Bug最多的作者）
        author_risk = {}
        for author in self.df['author'].unique():
            author_commits = self.df[self.df['author'] == author]
            if len(author_commits) > 10:  # 至少有10个提交
                bug_rate = author_commits['is_bug_fix'].mean()
                author_risk[author] = {
                    'bug_rate': bug_rate,
                    'total_commits': len(author_commits),
                    'bug_count': author_commits['is_bug_fix'].sum()
                }

        patterns['high_risk_authors'] = dict(
            sorted(author_risk.items(), key=lambda x: x[1]['bug_rate'], reverse=True)[:10]
        )

        return patterns

    def generate_summary_report(self, output_dir: str = "git_analysis_report"):
        """生成完整的分析报告"""
        print(f"\n生成分析报告到目录: {output_dir}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存原始数据
        if self.df is not None:
            self.df.to_csv(os.path.join(output_dir, "all_commits.csv"), index=False, encoding='utf-8')
            self.bug_fixes_df.to_csv(os.path.join(output_dir, "bug_fixes.csv"), index=False, encoding='utf-8')

        # 2. 分析bug模式
        bug_patterns = self.analyze_bug_patterns()
        high_risk_patterns = self.identify_high_risk_patterns()

        # 3. 生成文本报告
        report_content = self._create_text_report(bug_patterns, high_risk_patterns)

        report_file = os.path.join(output_dir, "analysis_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 4. 生成JSON报告
        json_report = {
            'repository': self.repo_path,
            'analysis_date': datetime.now().isoformat(),
            'summary_statistics': {
                'total_commits': len(self.df),
                'bug_fix_commits': len(self.bug_fixes_df),
                'bug_fix_percentage': len(self.bug_fixes_df) / len(self.df) * 100,
                'unique_authors': len(self.df['author'].unique()),
                'time_period_covered': {
                    'first_commit': self.df['commit_date'].min().isoformat() if not self.df.empty else None,
                    'last_commit': self.df['commit_date'].max().isoformat() if not self.df.empty else None
                }
            },
            'bug_patterns': bug_patterns,
            'high_risk_patterns': high_risk_patterns
        }

        with open(os.path.join(output_dir, "analysis_results.json"), 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)

        # 5. 生成可视化图表
        self._generate_visualizations(output_dir, bug_patterns, high_risk_patterns)

        print(f"报告已生成！包含以下文件:")
        print(f"  - {os.path.join(output_dir, 'analysis_report.md')}")
        print(f"  - {os.path.join(output_dir, 'analysis_results.json')}")
        print(f"  - {os.path.join(output_dir, 'all_commits.csv')}")
        print(f"  - {os.path.join(output_dir, 'bug_fixes.csv')}")
        print(f"  - {os.path.join(output_dir, 'visualizations/')} (包含所有图表)")

        return report_file
    def _create_text_report(self, bug_patterns: Dict, high_risk_patterns: Dict) -> str:
        """创建文本分析报告"""
        report = f"""# Git提交历史分析报告

## 仓库信息
- **分析仓库**: {self.repo_path}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总提交数**: {len(self.df)}
- **Bug修复提交数**: {len(self.bug_fixes_df)}
- **Bug修复比例**: {len(self.bug_fixes_df) / len(self.df) * 100:.1f}%
- **参与开发者数**: {len(self.df['author'].unique())}

## 关键发现

### 1. Bug产生的高风险模式

#### 1.1 高风险时间段
"""

        # 添加高风险时间段分析
        if 'high_risk_times' in high_risk_patterns:
            for time_cat, data in high_risk_patterns['high_risk_times'].items():
                report += f"- **{time_cat}**: Bug率 {data['bug_rate'] * 100:.1f}% (共{data['total_commits']}个提交)\n"

        report += """
#### 1.2 高风险文件类型
"""

        if 'high_risk_file_types' in high_risk_patterns:
            for ext, data in list(high_risk_patterns['high_risk_file_types'].items())[:5]:
                report += f"- **{ext if ext else '无扩展名'}**: Bug率 {data['risk_score'] * 100:.1f}% ({data['bug_count']}/{data['total_count']})\n"

        report += """
#### 1.3 高风险提交特征
"""

        if 'high_risk_commit_characteristics' in high_risk_patterns:
            patterns = high_risk_patterns['high_risk_commit_characteristics']
            report += f"""
| 特征 | Bug率 | 说明 |
|------|-------|------|
| 大型提交 (>100行) | {patterns.get('large_commits_bug_rate', 0) * 100:.1f}% | 修改行数多的提交更容易引入Bug |
| 小型提交 (≤10行) | {patterns.get('small_commits_bug_rate', 0) * 100:.1f}% | 小修改相对安全 |
| 多文件提交 (>5文件) | {patterns.get('multi_file_bug_rate', 0) * 100:.1f}% | 同时修改多个文件风险较高 |
| 单文件提交 | {patterns.get('single_file_bug_rate', 0) * 100:.1f}% | 风险相对较低 |
| 重构提交 | {patterns.get('refactor_bug_rate', 0) * 100:.1f}% | 重构代码有一定风险 |
| 非重构提交 | {patterns.get('non_refactor_bug_rate', 0) * 100:.1f}% | 普通功能开发风险 |

"""

        report += """
### 2. Bug修复模式分析

#### 2.1 时间分布
"""

        if 'time_analysis' in bug_patterns:
            time_analysis = bug_patterns['time_analysis']
            report += f"""
- **工作日 vs 周末**:
  - 工作日Bug修复: {time_analysis['weekend_vs_weekday']['weekday']} 个
  - 周末Bug修复: {time_analysis['weekend_vs_weekday']['weekend']} 个
  - 周末Bug修复占比: {time_analysis['weekend_vs_weekday']['weekend'] / (time_analysis['weekend_vs_weekday']['weekend'] + time_analysis['weekend_vs_weekday']['weekday']) * 100:.1f}%

- **时间段分布**:
"""

            for time_cat, count in time_analysis['by_time_category'].items():
                percentage = count / len(self.bug_fixes_df) * 100 if len(self.bug_fixes_df) > 0 else 0
                report += f"  - {time_cat}: {count} 个 ({percentage:.1f}%)\n"

        report += """
#### 2.2 作者贡献分析
"""

        if 'author_analysis' in bug_patterns:
            author_analysis = bug_patterns['author_analysis']
            report += f"""
- **Top 5 Bug修复者贡献了 {author_analysis['author_concentration']:.1f}% 的Bug修复**
- **平均每个作者修复Bug数**: {author_analysis['bug_fixes_per_author']:.1f}

**Top Bug修复者**:
"""

            for author, count in list(author_analysis['top_bug_fixers'].items())[:10]:
                report += f"  - {author}: {count} 个Bug修复\n"

        report += """
#### 2.3 文件类型分析
"""

        if 'file_type_analysis' in bug_patterns:
            file_analysis = bug_patterns['file_type_analysis']
            report += f"""
- **平均每个Bug修复涉及的文件类型**:
  - Python文件: {file_analysis['py_files_in_bug_fixes']:.1f} 个
  - Cython文件: {file_analysis['cython_files_in_bug_fixes']:.1f} 个
  - 测试文件: {file_analysis['test_files_in_bug_fixes']:.1f} 个

**最常出现Bug的文件扩展名**:
"""

            for ext, count in list(file_analysis['most_common_extensions'].items())[:10]:
                report += f"  - {ext if ext else '无扩展名'}: {count} 次\n"

        report += """
### 3. 建议与改进措施

#### 3.1 开发流程建议
1. **避免高风险时间段提交**: 减少在Bug率高发时间段(深夜、周末)提交重要代码变更
2. **代码审查重点**: 对高风险文件类型(如上文分析)的修改进行更严格的代码审查
3. **提交大小控制**: 建议将大型变更拆分为多个小提交，降低风险
4. **测试覆盖**: 确保高风险修改有充分的测试覆盖

#### 3.2 团队管理建议
1. **知识共享**: 将Top Bug修复者的经验在团队内部分享
2. **新成员指导**: 为新成员提供高风险区域的代码审查和指导
3. **代码所有权**: 为高风险模块指定明确的代码负责人

#### 3.3 技术改进建议
1. **自动化测试**: 为高风险文件类型增加自动化测试
2. **静态分析**: 对高风险代码区域实施更严格的静态分析
3. **监控预警**: 建立高风险提交的预警机制

## 附录

### 数据分析方法
1. **Bug识别**: 基于关键词匹配(包含{len(self.bug_keywords)}个关键词)
2. **风险计算**: Bug率 = Bug修复提交数 / 总提交数
3. **统计分析**: 基于{len(self.df)}个提交样本进行统计推断

### 数据质量说明
- 数据来源: Git提交历史
- 时间范围: {self.df['commit_date'].min().strftime('%Y-%m-%d') if not self.df.empty else 'N/A'} 至 {self.df['commit_date'].max().strftime('%Y-%m-%d') if not self.df.empty else 'N/A'}
- 样本大小: {len(self.df)}个提交

---
*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return report

    def _generate_visualizations(self, output_dir: str, bug_patterns: Dict, high_risk_patterns: Dict):
        """生成可视化图表"""
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Bug修复时间分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bug修复时间分布分析', fontsize=16, fontweight='bold')

        # 1.1 按小时分布
        ax1 = axes[0, 0]
        if 'time_analysis' in bug_patterns and 'by_hour' in bug_patterns['time_analysis']:
            hours = list(bug_patterns['time_analysis']['by_hour'].keys())
            counts = list(bug_patterns['time_analysis']['by_hour'].values())
            ax1.bar(hours, counts, color='skyblue', edgecolor='black')
            ax1.set_xlabel('小时 (0-23)')
            ax1.set_ylabel('Bug修复数量')
            ax1.set_title('Bug修复按小时分布')
            ax1.grid(True, alpha=0.3)

        # 1.2 按星期分布
        ax2 = axes[0, 1]
        if 'time_analysis' in bug_patterns and 'by_day' in bug_patterns['time_analysis']:
            days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            day_counts = []
            for day in range(7):
                day_counts.append(bug_patterns['time_analysis']['by_day'].get(day, 0))

            colors = ['lightblue' if i < 5 else 'lightcoral' for i in range(7)]
            ax2.bar(days, day_counts, color=colors, edgecolor='black')
            ax2.set_xlabel('星期')
            ax2.set_ylabel('Bug修复数量')
            ax2.set_title('Bug修复按星期分布')
            ax2.grid(True, alpha=0.3)

        # 1.3 按时间段分布
        ax3 = axes[1, 0]
        if 'high_risk_times' in high_risk_patterns:
            time_cats = list(high_risk_patterns['high_risk_times'].keys())
            bug_rates = [data['bug_rate'] * 100 for data in high_risk_patterns['high_risk_times'].values()]

            bars = ax3.bar(time_cats, bug_rates, color='salmon', edgecolor='black')
            ax3.set_xlabel('时间段')
            ax3.set_ylabel('Bug率 (%)')
            ax3.set_title('各时间段Bug率比较')
            ax3.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, rate in zip(bars, bug_rates):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{rate:.1f}%', ha='center', va='bottom')

        # 1.4 Bug修复者排行榜
        ax4 = axes[1, 1]
        if 'author_analysis' in bug_patterns and 'top_bug_fixers' in bug_patterns['author_analysis']:
            top_authors = list(bug_patterns['author_analysis']['top_bug_fixers'].items())[:10]
            authors = [a[0][:15] + '...' if len(a[0]) > 15 else a[0] for a in top_authors]
            counts = [a[1] for a in top_authors]

            bars = ax4.barh(authors[::-1], counts[::-1], color='lightgreen', edgecolor='black')
            ax4.set_xlabel('Bug修复数量')
            ax4.set_title('Top 10 Bug修复者')
            ax4.grid(True, alpha=0.3)

            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, counts[::-1])):
                ax4.text(count + 0.5, bar.get_y() + bar.get_height() / 2,
                         str(count), ha='left', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'bug_time_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 高风险文件类型图
        if 'high_risk_file_types' in high_risk_patterns and high_risk_patterns['high_risk_file_types']:
            fig, ax = plt.subplots(figsize=(12, 8))

            file_types = list(high_risk_patterns['high_risk_file_types'].keys())[:15]
            risk_scores = [data['risk_score'] * 100 for data in
                           list(high_risk_patterns['high_risk_file_types'].values())[:15]]

            # 创建渐变色
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(file_types)))

            bars = ax.barh(file_types[::-1], risk_scores[::-1], color=colors, edgecolor='black')
            ax.set_xlabel('Bug率 (%)', fontsize=12)
            ax.set_title('高风险文件类型 (Bug率排名)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # 添加详细数值
            for i, (bar, score, file_type) in enumerate(zip(bars, risk_scores[::-1], file_types[::-1])):
                data = high_risk_patterns['high_risk_file_types'][file_type]
                label = f'{score:.1f}% ({data["bug_count"]}/{data["total_count"]})'
                ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                        label, ha='left', va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'high_risk_file_types.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Bug修复提交特征对比图
        if not self.non_bug_fixes_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Bug修复 vs 非Bug修复提交特征对比', fontsize=16, fontweight='bold')

            # 3.1 提交大小对比
            ax1 = axes[0, 0]
            bug_sizes = self.bug_fixes_df['total_lines_changed'].values
            non_bug_sizes = self.non_bug_fixes_df['total_lines_changed'].values

            # 限制大小以便可视化
            bug_sizes_limited = np.clip(bug_sizes, 0, 500)
            non_bug_sizes_limited = np.clip(non_bug_sizes, 0, 500)

            ax1.hist(bug_sizes_limited, bins=30, alpha=0.7, label='Bug修复', color='salmon')
            ax1.hist(non_bug_sizes_limited, bins=30, alpha=0.7, label='非Bug修复', color='lightblue')
            ax1.set_xlabel('修改行数 (限制在500以内)')
            ax1.set_ylabel('提交数量')
            ax1.set_title('提交大小分布对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 3.2 文件数量对比
            ax2 = axes[0, 1]
            bug_files = self.bug_fixes_df['files_changed'].values
            non_bug_files = self.non_bug_fixes_df['files_changed'].values

            ax2.boxplot([non_bug_files, bug_files], labels=['非Bug修复', 'Bug修复'])
            ax2.set_ylabel('修改文件数')
            ax2.set_title('修改文件数对比')
            ax2.grid(True, alpha=0.3)

            # 3.3 作者集中度
            ax3 = axes[1, 0]
            if 'author_analysis' in bug_patterns:
                author_counts = list(bug_patterns['author_analysis']['top_bug_fixers'].values())
                cumulative = np.cumsum(author_counts) / np.sum(author_counts) * 100

                ax3.plot(range(1, len(cumulative) + 1), cumulative, 'o-', linewidth=2, markersize=8)
                ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% 线')
                ax3.set_xlabel('作者排名')
                ax3.set_ylabel('累计Bug修复占比 (%)')
                ax3.set_title('作者集中度分析 (洛伦兹曲线)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            # 3.4 时间分布对比
            ax4 = axes[1, 1]
            if 'high_risk_times' in high_risk_patterns:
                time_cats = list(high_risk_patterns['high_risk_times'].keys())
                bug_rates = [data['bug_rate'] * 100 for data in high_risk_patterns['high_risk_times'].values()]

                ax4.plot(time_cats, bug_rates, 'o-', linewidth=2, markersize=10, color='darkred')
                ax4.fill_between(time_cats, bug_rates, alpha=0.3, color='salmon')
                ax4.set_xlabel('时间段')
                ax4.set_ylabel('Bug率 (%)')
                ax4.set_title('各时间段Bug率变化')
                ax4.grid(True, alpha=0.3)

                # 标记最高点
                max_idx = np.argmax(bug_rates)
                ax4.annotate(f'最高风险\n{bug_rates[max_idx]:.1f}%',
                             xy=(time_cats[max_idx], bug_rates[max_idx]),
                             xytext=(0, 20),
                             textcoords='offset points',
                             ha='center',
                             arrowprops=dict(arrowstyle='->', color='red'))

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'bug_vs_nonbug_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"已生成 {len(os.listdir(viz_dir))} 个可视化图表")

    def run_complete_analysis(self, limit: int = 5000, output_dir: str = "git_analysis_report"):
        """
        运行完整的分析流程

        Args:
            limit: 分析的最大提交数量
            output_dir: 输出目录

        Returns:
            分析报告文件路径
        """
        print("=" * 60)
        print("开始Git提交历史深度分析")
        print("=" * 60)

        # 1. 提取历史数据
        self.extract_git_history(limit=limit)

        # 2. 生成分析报告
        report_file = self.generate_summary_report(output_dir)

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

        return report_file

def main():
    """
    主函数：演示如何使用分析器
    """
    import sys

    print("Git提交历史深度分析工具")
    print("-" * 40)

    # 获取仓库路径
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = input("请输入Git仓库路径 (本地路径或GitHub URL): ").strip()

        if not repo_path:
            # 默认示例（需要修改为实际路径）
            repo_path = "/path/to/your/repository"
            print(f"使用默认路径: {repo_path}")

    # 创建分析器
    analyzer = GitHistoryAnalyzer(repo_path)

    # 运行完整分析
    try:
        report_file = analyzer.run_complete_analysis(
            limit=2000,  # 分析最近的2000个提交
            output_dir="analysis_results"  # 输出目录
        )

        print(f"\n分析报告已保存到: {report_file}")

        # 显示关键结论
        print("\n关键结论摘要:")
        print("-" * 40)

        if analyzer.analysis_results:
            bug_patterns = analyzer.analysis_results

            # 计算总体Bug率
            total_commits = len(analyzer.df)
            bug_fixes = len(analyzer.bug_fixes_df)
            bug_rate = bug_fixes / total_commits * 100 if total_commits > 0 else 0

            print(f"1. 总体Bug率: {bug_rate:.1f}% ({bug_fixes}/{total_commits})")

            # 找出最高风险时间段
            if 'high_risk_periods' in bug_patterns and bug_patterns['high_risk_periods']:
                highest_risk = bug_patterns['high_risk_periods'][0]
                print(f"2. 最高风险时间段: {highest_risk['hour']}时 (Bug率: {highest_risk['bug_rate'] * 100:.1f}%)")

            # 找出最高风险文件类型
            if hasattr(analyzer, 'high_risk_patterns') and 'high_risk_file_types' in analyzer.high_risk_patterns:
                file_types = analyzer.high_risk_patterns['high_risk_file_types']
                if file_types:
                    highest_risk_file = list(file_types.items())[0]
                    print(
                        f"3. 最高风险文件类型: {highest_risk_file[0]} (Bug率: {highest_risk_file[1]['risk_score'] * 100:.1f}%)")

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()