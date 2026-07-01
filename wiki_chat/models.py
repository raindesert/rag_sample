"""模型加载工具：下载/读取缓存 + 实例化 SentenceTransformer."""

import os


def get_model_path(model_name: str, use_modelscope: bool) -> str:
    """下载模型到本地并返回路径，支持 ModelScope 或 HuggingFace.

    snapshot_download 自带缓存检查：本地已存在则直接返回缓存路径，不重复下载。
    """
    cache_dir = os.environ.get("MODEL_CACHE", "./model_cache")

    if use_modelscope:
        try:
            from modelscope import snapshot_download
            cache = os.environ.get("MODELSCOPE_CACHE", cache_dir)
            return snapshot_download(model_name, cache_dir=cache)
        except ImportError:
            print("WARNING: modelscope not installed, falling back to HuggingFace")

    from huggingface_hub import snapshot_download
    cache = os.environ.get("HF_HOME", cache_dir)
    return snapshot_download(model_name, cache_dir=cache)


def load_cross_encoder(model_name: str, use_modelscope: bool, device: str | None = None):
    """加载 CrossEncoder 模型，先解析本地路径（命中缓存即复用）.

    device: 传给 CrossEncoder 的设备（None 则用库默认自动选择）。
    """
    from sentence_transformers import CrossEncoder

    local_path = get_model_path(model_name, use_modelscope)
    if device is None:
        return CrossEncoder(local_path)
    return CrossEncoder(local_path, device=device)


def load_sentence_transformer(model_name: str, use_modelscope: bool, device: str | None = None):
    """加载 SentenceTransformer 模型，先解析本地路径（命中缓存即复用）.

    device: 传给 SentenceTransformer 的设备（None 则用库默认自动选择）。
    """
    from sentence_transformers import SentenceTransformer

    local_path = get_model_path(model_name, use_modelscope)
    if device is None:
        return SentenceTransformer(local_path)
    return SentenceTransformer(local_path, device=device)