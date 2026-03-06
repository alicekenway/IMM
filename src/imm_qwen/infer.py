import argparse
import json
from pathlib import Path

from .config import (
    ImmQwenProjectConfig,
    InferenceToolConfig,
    MemoryDimensionsConfig,
    ModelBuildConfig,
)
from .infer_tools import InferenceEngine
from .train_tools import build_model_with_imm


def load_project_config(config_path: str) -> ImmQwenProjectConfig:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_config = ModelBuildConfig(**payload["model"])
    memory_dim = MemoryDimensionsConfig(**payload["memory_dimensions"])
    return ImmQwenProjectConfig(model=model_config, memory_dimensions=memory_dim)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IMM-Qwen inference engine.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON.")
    parser.add_argument("--session_id", type=str, default="demo-session", help="Session identifier.")
    parser.add_argument("--text", type=str, required=True, help="User text for one turn.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_config = load_project_config(args.config)
    artifacts = build_model_with_imm(project_config)
    options = InferenceToolConfig()
    engine = InferenceEngine(
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        controller=artifacts.controller,
        template_memory_state=artifacts.memory_state,
        options=options,
    )
    response = engine.generate_response(
        session_id=args.session_id,
        user_text=args.text,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()

