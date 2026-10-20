def _normalize_messages(messages):
    normalized = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = item.get("content", "")
        normalized.append(
            {
                "role": role,
                "content": "" if content is None else str(content),
            }
        )
    return normalized


def _build_chatml_prompt(messages, add_generation_prompt):
    parts = []
    for item in messages:
        parts.append(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _build_phi3_prompt(messages, add_generation_prompt):
    tag_map = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    parts = []
    for item in messages:
        parts.append(f"{tag_map[item['role']]}\n{item['content']}<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _build_llama3_prompt(messages, add_generation_prompt):
    parts = ["<|begin_of_text|>"]
    for item in messages:
        parts.append(f"<|start_header_id|>{item['role']}<|end_header_id|>\n\n{item['content']}<|eot_id|>")
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _build_inst_prompt(messages, add_generation_prompt, include_sys_tags):
    messages = _normalize_messages(messages)
    system_blocks = [m["content"].strip() for m in messages if m["role"] == "system" and m["content"].strip()]
    conversation = [m for m in messages if m["role"] != "system"]
    system_text = "\n\n".join(system_blocks).strip()

    prompt_parts = []
    first_user = True
    idx = 0

    while idx < len(conversation):
        item = conversation[idx]
        if item["role"] != "user":
            idx += 1
            continue

        user_text = item["content"].strip()
        if first_user and system_text:
            if include_sys_tags:
                user_text = f"<<SYS>>\n{system_text}\n<</SYS>>\n\n{user_text}".strip()
            else:
                user_text = f"{system_text}\n\n{user_text}".strip()
        first_user = False

        turn_text = f"<s>[INST] {user_text} [/INST]"

        if idx + 1 < len(conversation) and conversation[idx + 1]["role"] == "assistant":
            assistant_text = conversation[idx + 1]["content"].strip()
            if assistant_text:
                turn_text += f" {assistant_text} </s>"
            else:
                turn_text += " </s>"
            idx += 2
        else:
            idx += 1
            if not add_generation_prompt:
                turn_text += " </s>"

        prompt_parts.append(turn_text)

    if not prompt_parts:
        if system_text:
            if include_sys_tags:
                return f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>> [/INST]"
            return f"<s>[INST] {system_text} [/INST]"
        return ""

    return "".join(prompt_parts)


def build_chat_prompt(model_path, messages, add_generation_prompt=True):
    """
    根据模型路径识别模板，并将多轮 messages 渲染为完整对话 Prompt。
    """
    path_lower = (model_path or "").lower()
    messages = _normalize_messages(messages)

    if "llama-3" in path_lower or "llama3" in path_lower or "llama-3.1" in path_lower or "llama3.1" in path_lower:
        return _build_llama3_prompt(messages, add_generation_prompt)

    if "qwen" in path_lower:
        return _build_chatml_prompt(messages, add_generation_prompt)
    if "phi3" in path_lower or "phi-3" in path_lower:
        return _build_phi3_prompt(messages, add_generation_prompt)
    if "mistral" in path_lower:
        return _build_inst_prompt(messages, add_generation_prompt, include_sys_tags=False)
    return _build_inst_prompt(messages, add_generation_prompt, include_sys_tags=True)


def get_model_template(model_path, system, user, response=""):
    """
    根据模型路径识别模型类型并生成对应的 Prompt 模板。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
    if response:
        messages.append({"role": "assistant", "content": response})
    return build_chat_prompt(model_path, messages, add_generation_prompt=(not bool(response)))
