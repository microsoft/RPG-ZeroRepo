import os
import json
import random


def sample_tests(
    category,
    test_data, 
    num_files=2, 
    num_classes_or_functions_per_file=1,
    num_modules_per_class=2
):
    selected_tests = {}

    # 第一层：文件采样
    files = list(test_data.keys())

    # 排除文件名中包含 'base' 的文件
    files = [file for file in files if 'base' not in file and "issue" not in file]
    
    num_files_to_sample = min(len(files), num_files)  # 确保不采样超过文件总数
    sampled_files = random.sample(files, num_files_to_sample) 
    
    # if category != "metrics" \
    #    else ["test_regression", "test_classification", "test_ranking"]
    
    for file_name in sampled_files:
        selected_tests[file_name] = {}

        # 第二层：从文件中选择类或functions
        file_data = test_data[file_name]

        class_or_function_keys = [k for k in file_data.keys() if 'base' not in k and "issue" not in k]  # 排除包含 base 的类
        num_classes_or_functions = min(len(class_or_function_keys), num_classes_or_functions_per_file)  # 不超过实际数量
        sampled_classes_or_functions = random.sample(class_or_function_keys, num_classes_or_functions)

        for class_or_function in sampled_classes_or_functions:
            selected_tests[file_name][class_or_function] = {}

            # 第三层：从类或functions中选择模块
            modules = file_data[class_or_function]
            num_modules = min(len(modules), num_modules_per_class)  # 不超过实际模块数量
            sampled_modules = random.sample(list(modules.keys()), num_modules)

            for module in sampled_modules:
                selected_tests[file_name][class_or_function][module] = modules[module]

    return selected_tests

import json


def count_sampled_algorithms(sampled_data):
    total_functions = 0
    total_modules = 0
    total_classes_or_functions = 0
    total_files = 0
    total_categories = 0

    # 遍历每个类别
    for typo, test_data in sampled_data.items():
        total_categories += 1
        print(f"\nCategory: {typo}")
        
        # 遍历每个文件
        for file_name, classes_or_functions in test_data.items():
            total_files += 1
            print(f"  File: {file_name}")
            
            # 遍历每个类或函数
            for class_or_function, modules in classes_or_functions.items():
                total_classes_or_functions += 1
                print(f"    Class/Function: {class_or_function}")
                

                # 遍历每个模块
                for module, selected_functions in modules.items():
                    num_functions = len(selected_functions)
                    total_functions += num_functions
                    total_modules += 1
                    print(f"      Module: {module}, Functions: {num_functions}")

    # 输出总计
    print(f"\nSummary:")
    print(f"  Total Categories: {total_categories}")
    print(f"  Total Files: {total_files}")
    print(f"  Total Classes/Functions: {total_classes_or_functions}")
    print(f"  Total Modules: {total_modules}")
    print(f"  Total Test Functions: {total_functions}")

# 假设 sampled_data 是从之前的采样函数得到的数据
# sampled_data = sample_tests(test_data, num_files=5, num_classes_or_functions_per_file=10, num_modules_per_class=4)




if __name__ == "__main__":
    # 按每个分类 sample
    repo_name = "sympy"
    file_path = f"/mnt/jianwen/RepoEncoder/all_results/refactored_test/{repo_name}.json"
    output_path = f"/mnt/jianwen/RepoEncoder/all_results/sampled_test/sample_{repo_name}.json"

    with open(file_path, 'r') as f:
        data = json.load(f)

    file_map = data["files"]
    refactored_file_map = data["refactor"]

    sampled_data = {}

    for typo, test_data in refactored_file_map.items():
        sampled_tests = sample_tests(
            typo,
            test_data,
            num_files=12,
            num_classes_or_functions_per_file=20,
            num_modules_per_class=10
        )

        sampled_data[typo] = sampled_tests

        for file_name, classes_or_functions in sampled_tests.items():
            print(f"File: {file_name}")
            for class_or_function, modules in classes_or_functions.items():
                print(f"  Class/Function: {class_or_function}")
                for module, selected_functions in modules.items():
                    print(f"    Module: {module}, Selected Functions: {selected_functions}")

    with open(output_path, 'w') as out_file:
        json.dump(sampled_data, out_file, indent=4)

    count_sampled_algorithms(sampled_data)