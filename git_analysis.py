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


if __name__ == "__main__":
    main()