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




if __name__ == "__main__":
    main()