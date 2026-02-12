import json
import os

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

from zerorepo.rpg_gen.base import RPG


repos = [
    "HttpEasy", "MLKit-Py", "SymbolicMath", "SymbolicMath_2", "StatModeler", "PyWebEngine", "TableKit"
]

top_dir = "/home/v-jianwenluo/temp/repo_encoder_exp"

models = ["gpt-4.1", "gpt-5-mini"]


for model in models:
    model_dir = os.path.join(top_dir, model, "ref")
    
    for repo in repos:
        repo_model_dir = os.path.join(model_dir, repo)
        
        if not os.path.exists(repo_model_dir):
            continue
        
        cur_rpg_path = os.path.join(repo_model_dir, "checkpoints", "cur_repo_rpg.json")
        repo_data_path = os.path.join(repo_model_dir, "checkpoints", "repo_data.json")
        
        with open(cur_rpg_path, 'r') as f:
            rpg_dict = json.load(f)
        with open(repo_data_path, 'r') as f:
            repo_data_dict = json.load(f)
        
        from copy import deepcopy
        new_repo_data = deepcopy(repo_data_dict)
        
        repo_rpg = RPG.from_dict(rpg_dict)
        cur_component = repo_rpg.get_functionality_graph()
        
        new_repo_data["Component"] = cur_component
        
        new_repo_data_path = os.path.join(repo_model_dir, "checkpoints", "new_repo_data.json")

        with open(new_repo_data_path, 'w') as f:
            json.dump(new_repo_data, f, indent=4)



repos = [
    "feature_file", "feature_only"
]

top_dir = "/home/v-jianwenluo/temp/repo_encoder_exp"

models = ["gpt-5-mini"]


for model in models:
    model_dir = os.path.join(top_dir, model, "ablation")
    
    for repo in repos:
        repo_model_dir = os.path.join(model_dir, repo)
        
        if not os.path.exists(repo_model_dir):
            continue
        
        cur_rpg_path = os.path.join(repo_model_dir, "checkpoints", "cur_repo_rpg.json")
        repo_data_path = os.path.join(repo_model_dir, "checkpoints", "repo_data.json")
        
        with open(cur_rpg_path, 'r') as f:
            rpg_dict = json.load(f)
        with open(repo_data_path, 'r') as f:
            repo_data_dict = json.load(f)
        
        from copy import deepcopy
        new_repo_data = deepcopy(repo_data_dict)
        
        repo_rpg = RPG.from_dict(rpg_dict)
        cur_component = repo_rpg.get_functionality_graph()
        
        new_repo_data["Component"] = cur_component
        
        new_repo_data_path = os.path.join(repo_model_dir, "checkpoints", "new_repo_data.json")
        with open(new_repo_data_path, 'w') as f:
            json.dump(new_repo_data, f, indent=4)
        
        