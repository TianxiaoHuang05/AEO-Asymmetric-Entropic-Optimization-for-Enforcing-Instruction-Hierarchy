import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
try:
    from .label_utils import get_checker, check_rule_following
except ImportError:
    from label_utils import get_checker, check_rule_following
try:
    from .template_utils import get_model_template
except ImportError:
    from template_utils import get_model_template

# ===============================
# 1. 配置与推理函数
# ===============================
def generate_responses(model, tokenizer, data, device, model_path, max_new_tokens=256):
    """
    为数据批量生成模型回复。
    """
    model.eval()
    for item in tqdm(data, desc="Inference"):
        # 兼容两种字段名
        system = item.get("system") or item.get("system_message", "")
        # IHEval 使用 user, new_data 使用 user_message + task
        user_input = item.get("user") or f"{item.get('user_message', '')}\n{item.get('task', '')}"
        
        # 使用通用模板函数生成 Prompt
        prompt = get_model_template(model_path, system, user_input)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 替换或新增模型回复
        item["model_response"] = response
    return data

# ===============================
# 2. 标注函数
# ===============================
def label_data(data, filename, checker):
    """
    根据模型回复生成 response_label。
    """
    for item in tqdm(data, desc=f"Labeling {os.path.basename(filename)}"):
        system = item.get("system") or item.get("system_message", "")
        response = item.get("model_response", "")
        
        # 使用 label_utils 中的校验逻辑
        is_follow = checker(response, system)
        item["response_label"] = "follow_system" if is_follow else "violate_system"
    return data

# ===============================
# 3. 特征提取函数
# ===============================
def extract_hidden_states(model, tokenizer, data, device, model_path):
    """
    提取 hd (last token) 和 hd3 (mean response) 特征。
    """
    model.eval()
    for item in tqdm(data, desc="Extracting HD"):
        system = item.get("system") or item.get("system_message", "")
        user_input = item.get("user") or f"{item.get('user_message', '')}\n{item.get('task', '')}"
        response = item.get("model_response", "")
        
        # 使用通用模板函数生成 Full Prompt
        full_prompt = get_model_template(model_path, system, user_input)
        input_ids = tokenizer.encode(full_prompt + " " + response, return_tensors="pt").to(model.device)
        
        # 找到回复部分的起始索引
        prompt_ids = tokenizer.encode(full_prompt, return_tensors="pt")
        response_start_idx = prompt_ids.shape[1]
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            # 最后一层的隐藏状态: [batch, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1][0] 
            
            # hd: 最后一个 Token 的特征
            hd = last_hidden[-1].cpu().numpy().tolist()
            
            # hd3: 回复部分的平均特征
            response_hidden = last_hidden[response_start_idx:]
            if response_hidden.size(0) > 0:
                hd3 = torch.mean(response_hidden, dim=0).cpu().numpy().tolist()
            else:
                hd3 = hd # Fallback
            
            item["hd"] = hd
            item["hd3"] = hd3
    return data

# ===============================
# 4. 主逻辑
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Integrated Data Processing Pipeline (Inference, Labeling, HD Extraction).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LLM.")
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU ID(s) to use. If multiple, will use device_map='auto'.")
    parser.add_argument("--new_data_dir", type=str, default="./data/new_data")
    parser.add_argument("--iheval_dir", type=str, default="./data/iheval")
    parser.add_argument("--debug", type=int, default=None, help="If set, only process first N JSON files per directory.")
    
    args = parser.parse_args()
    model_name = os.path.basename(args.model_path.rstrip("/"))
    
    # 强制即时刷新
    print(f"=== Script Starting: {model_name} ===", flush=True)
    
    # 1. 加载模型与分词器
    print(f">>> Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    if len(args.gpu) > 1:
        # 多 GPU 分片加载
        print(f">>> Multi-GPU mode enabled: {args.gpu}. Using device_map='auto'.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        device = model.device # 通常是第一张卡
    else:
        # 单 GPU 精确加载
        device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
        print(f">>> Single-GPU mode: {device}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map={"": device}
        )

    # 定义任务路径
    task_dirs = {
        "new_data": args.new_data_dir,
        "iheval": args.iheval_dir
    }

    for task_key, base_dir in task_dirs.items():
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found. Skipping {task_key}.")
            continue
            
        print(f"\n<<< Processing Task: {task_key} from {base_dir} >>>")
        
        # 创建输出目录
        output_base = os.path.join(base_dir, model_name)
        hd_output_base = os.path.join(output_base, f"_{model_name}_hd")
        os.makedirs(hd_output_base, exist_ok=True)

        json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]
        if args.debug is not None:
            json_files = json_files[:args.debug]
            print(f">>> DEBUG MODE: Only processing first {args.debug} files in {task_key}.")
        
        for filename in json_files:
            file_path = os.path.join(base_dir, filename)
            base_filename = filename.replace(".json", "")
            
            # 1. 预先构建最终保存路径以检查是否已存在 (断点续跑)
            hd_save_path = os.path.join(hd_output_base, f"{base_filename}_{model_name}_hd.json")
            if os.path.exists(hd_save_path):
                print(f"\n>>> Skipping {filename}: Result already exists at {hd_save_path}")
                continue

            print("-" * 20)
            print(f"Processing File: {filename}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict): data = [data]
            
            # 级联应用 debug 参数：限制每个文件内的样本数
            if args.debug is not None:
                data = data[:args.debug]
                print(f">>> DEBUG: Using first {len(data)} samples from this file.")

            # A. 推理生成 model_response
            data = generate_responses(model, tokenizer, data, device, args.model_path)

            # B. 自动标注 response_label
            # 对于 new_data，如果名字含有 rule-following 或类似，可用对应 checker
            # 否则默认使用 rule-following 的逻辑作为通用基准
            checker = get_checker(filename) 
            if task_key == "new_data" and "rule-following" not in filename:
                # 给 new_data 默认使用 rule-following 校验
                checker = check_rule_following
            
            data = label_data(data, filename, checker)

            # 保存推理+标注后的 json
            save_name = filename.replace(".json", f"_{model_name}.json")
            save_path = os.path.join(output_base, save_name)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved inference result to: {save_path}")

            # C. 提取 hd/hd3 特征
            data = extract_hidden_states(model, tokenizer, data, device, args.model_path)

            # 保存特征后的 json
            hd_save_name = filename.replace(".json", f"_{model_name}_hd.json")
            hd_save_path = os.path.join(hd_output_base, hd_save_name)
            with open(hd_save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved feature result to: {hd_save_path}")

    print("\n" + "="*40)
    print("All Data Processing Completed Successfully!")

if __name__ == "__main__":
    main()
