"""Task metadata resolution from ``tasks/`` directory.

Each task (dataset split) is described by a ``metadata.yaml`` that
specifies the data path, prompt templates, and chat template.  This
module locates and parses those files at training time.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIRECTORY = REPO_ROOT / "tasks"
# PHASE-Tree: chat templates live under models/chat_templates/ (no top-level symlink).
CHAT_TEMPLATES_DIRECTORY = REPO_ROOT / "models" / "chat_templates"


def get_metadata(ds_names, use_per_task_emb):
    metadata = dict()
    for ds_name in ds_names:
        metadata[ds_name] = get_metadata_for_task(ds_name)
    return metadata


def get_metadata_for_task(task_name: str) -> dict:
    """Return metadata for a single task."""
    metadata = {}
    task_dir = TASKS_DIRECTORY / task_name
    with (task_dir / "metadata.yaml").open() as f:
        metadata = yaml.safe_load(f.read())
        # Add task name based on the directory name.
        metadata["task_name"] = task_dir.name
    return metadata


def get_all_metadata() -> list:
    """Return metadata for all tasks, sorted alphabetically."""
    task_dirs = [path for path in TASKS_DIRECTORY.iterdir() if path.is_dir()]

    metadata_list = [get_metadata_for_task(task_dir.name) for task_dir in task_dirs]

    return sorted(metadata_list, key=lambda x: x["task_name"])


def get_all_metadata_as_dict() -> dict:
    """Return metadata for all tasks as a dictionary."""
    metadata = get_all_metadata()
    metadata_dict = {}
    for task in metadata:
        metadata_dict[task["task_name"]] = task
    return metadata_dict
