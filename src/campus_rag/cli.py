from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .document_loader import SUPPORTED_SUFFIXES
from .pipeline import CampusRagPipeline
from .schema import ensure_dir


app = typer.Typer(help="校园规章智能问答系统命令行工具")


@app.command("build")
def build_index(config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径")) -> None:
    """读取 data/raw 并构建检索索引。"""

    pipeline = CampusRagPipeline(load_config(config))
    result = pipeline.build_index()
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="用户问题"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径"),
) -> None:
    """基于已构建索引回答问题。"""

    pipeline = CampusRagPipeline(load_config(config))
    result = pipeline.ask(question)
    typer.echo(result.answer)
    if result.hits:
        typer.echo("\n命中片段：")
        for index, hit in enumerate(result.hits, start=1):
            typer.echo(
                f"{index}. 分数 {hit.final_score:.3f}；来源 {hit.chunk.source}；页码 {hit.chunk.page_start or '未知'}；章节 {hit.chunk.chapter_title or '未知'}"
            )


@app.command("init-dirs")
def init_dirs(config: Optional[Path] = typer.Option(None, "--config", "-c", help="配置文件路径")) -> None:
    """创建数据目录。"""

    cfg = load_config(config)
    for path in [cfg.raw_dir, cfg.processed_dir, cfg.index_dir, cfg.finetune_dir]:
        ensure_dir(path)
        typer.echo(f"已确认目录：{path}")


@app.command("formats")
def formats() -> None:
    """查看支持导入的知识库格式。"""

    typer.echo("支持格式：" + "、".join(sorted(SUPPORTED_SUFFIXES)))


if __name__ == "__main__":
    app()
