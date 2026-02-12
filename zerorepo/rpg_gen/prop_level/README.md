# Property-Level Builder (PropBuilder)

主类串联 **feature selection** → **feature refactoring** 的完整流程。

## 核心功能

```python
from rpg_building.prop_level.prop_builder import create_prop_builder

# 创建构建器
builder = create_prop_builder(llm_cfg)

# 完整流程：特征选择 -> 特征重构
result = builder.build_property_tree(repo_data)
```

## 工作流程

```
Repository Data
       ↓
[Feature Selection]    # PropLevelAgent: 生成特征树
       ↓
   Feature Tree
       ↓  
[Feature Refactoring]  # FeatureRefactorAgent: 组织成组件
       ↓
   Organized Components
```

## 主要方法

### 1. 完整流程
```python
result = builder.build_property_tree(
    repo_data=repo_data,
    feature_gen_iterations=10,
    refactor_iterations=10,
    mode="feature"  # or "simple"
)

# 输出结构
{
    "Feature_tree": {...},           # 生成的特征树
    "Component": [...],              # 重构的组件
    "pipeline_summary": {
        "total_feature_paths": 150,
        "organized_components": 5,
        "coverage_rate": 0.92
    }
}
```

### 2. 分步执行
```python
# 只运行特征选择
selection_result = builder.run_selection_only(repo_data)

# 只运行特征重构  
refactor_result = builder.run_refactoring_only(feature_tree, repo_data)
```

### 3. 批量处理
```python
from rpg_building.prop_level.prop_builder import process_repository_batch

output_files = process_repository_batch(
    repo_files=["repo1.json", "repo2.json"],
    llm_cfg=llm_cfg,
    output_dir="results/"
)
```

## 关键改进

### 基于叶子节点的监督信号
- 使用实际特征内容而非路径字符串进行进度跟踪
- 解决了路径重组后监督信号不准确的问题
- 遵循 `refactored_iteration.py` 的方法

### 路径深度验证
- 只处理深度为4的有效路径 (`Level1/Level2/Level3/Feature`)
- 过滤无效或不符合要求的路径结构

### 智能代理管理
- 懒加载：代理在需要时才创建
- 配置传递：LLM配置自动传递给两个代理
- 错误处理：每个阶段都有完善的错误处理

## 配置示例

```python
from rpg_building.base.llm_client import LLMConfig

llm_cfg = LLMConfig(
    model="gpt-4",
    api_key="your-api-key",
    max_tokens=4000,
    temperature=0.7
)

repo_data = {
    "repository_name": "example-repo",
    "description": "Repository description", 
    "main_purpose": "Main functionality",
    "programming_language": "Python",
    "domain": "web_development"
}

# 运行完整流程
builder = create_prop_builder(llm_cfg)
result = builder.build_property_tree(repo_data)
```

## 输出结果

### 流程统计
```json
{
    "pipeline_summary": {
        "total_feature_paths": 150,      // 生成的特征路径数
        "organized_components": 5,       // 组织的组件数
        "coverage_rate": 0.92,           // 覆盖率
        "selection_iterations": 8,       // 选择阶段迭代次数
        "refactor_iterations": 6         // 重构阶段迭代次数
    }
}
```

### 组织的组件
```json
{
    "Component": [
        {
            "name": "Authentication System",
            "purpose": "User authentication and security",
            "refactored_subtree": {
                "Auth": {
                    "Login": ["form_validation", "session_mgmt"],
                    "Security": ["password_hash", "token_verify"]
                }
            },
            "actual_size": 25,           // 包含的特征数
            "util_percent": 0.18         // 占总特征的百分比
        }
    ]
}
```

## 便捷函数

### 单个仓库处理
```python
from rpg_building.prop_level.prop_builder import process_repository

result = process_repository(
    repo_data=repo_data,
    llm_cfg=llm_cfg,
    output_path="result.json"
)
```

### 批量处理
```python
output_files = process_repository_batch(
    repo_files=["repo1.json", "repo2.json", "repo3.json"],
    llm_cfg=llm_cfg,
    output_dir="batch_results/",
    feature_gen_iterations=5,
    refactor_iterations=5
)
```

## 使用场景

1. **完整属性级构建**：从仓库数据生成完整的特征树和组件组织
2. **分阶段调试**：独立运行选择或重构阶段进行调试
3. **批量处理**：高效处理多个仓库
4. **管道集成**：作为更大构建管道中的一个阶段

## 错误处理

```python
result = builder.build_property_tree(repo_data)

if "error" in result:
    print(f"Workflow failed: {result['error']}")
else:
    print(f"Success! Generated {len(result['Component'])} components")
```

## 日志记录

```python
import logging

logger = logging.getLogger(__name__)
builder = create_prop_builder(llm_cfg, logger=logger)

# 详细的日志输出包括：
# - 各阶段的进度
# - 特征生成统计
# - 组件组织结果  
# - 错误和警告信息
```

这个 PropBuilder 提供了一个简洁、强大的接口来执行完整的 select feature → refactor subtree 工作流程，同时保持了良好的模块化和可扩展性。