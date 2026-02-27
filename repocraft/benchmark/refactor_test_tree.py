import os
import json

class TestClassifier:

    # 可以根据需要扩展这个集合，加入更多通用的、无意义的目录名
    MEANINGLESS_DIRS = {"test", "tests"}

    @classmethod
    def _find_category(cls, path: str) -> str | None:
        """
        根据路径智能查找最适合的分类名称。
        从文件路径中提取有意义的目录名作为分类。
        """
        # 获取文件所在的目录路径
        dir_path = os.path.dirname(path)
        path_parts = dir_path.strip('/').split('/')

        # 策略一（核心策略）：由下至上反向迭代路径部分
        for part in reversed(path_parts):
            if part.lower() not in cls.MEANINGLESS_DIRS:
                return part

        # 策略二（备用）：回退到使用 'repos' 的子目录
        try:
            full_path_parts = path.strip('/').split('/')
            repo_index = full_path_parts.index('repos')
            if repo_index < len(full_path_parts) - 1:
                return full_path_parts[repo_index + 1]
        except (ValueError, IndexError):
            pass

        return None

    @classmethod
    def build_classification_tree(cls, file_structure: dict) -> dict:
        """
        将扁平的文件结构转换为按最适合的类别和模块分类的树状结构。

        Args:
            file_structure: 扁平的 {file_path: content} 映射

        Returns:
            分类树 {category: {module_name: content}}
        """
        tree = {}

        for full_path, content in file_structure.items():
            category_name = cls._find_category(full_path)

            if not category_name:
                print(f"Warning: Could not determine category for path: {full_path}")
                continue

            base_name = os.path.basename(full_path)
            module_name, _ = os.path.splitext(base_name)

            if category_name not in tree:
                tree[category_name] = {}

            tree[category_name][module_name] = content

        return tree
