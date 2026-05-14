from __future__ import annotations

from typing import Any, Dict

from .config import Settings
from .schemas import ModelEnum, VideoGenerationRequest

# Default size per model (first supported size from WAN_CONFIGS)
_MODEL_DEFAULT_SIZE = {
    ModelEnum.t2v_a14b.value: "1280*720",
    ModelEnum.i2v_a14b.value: "832*480",
    ModelEnum.ti2v_5b.value: "1280*704",
    ModelEnum.animate_14b.value: "720*1280",
    ModelEnum.s2v_14b.value: "832*480",
}

# Sub-directory name under ckpt_dir for each model
_MODEL_CKPT_SUBDIR = {
    ModelEnum.t2v_a14b.value: "Wan2.2-T2V-A14B",
    ModelEnum.i2v_a14b.value: "Wan2.2-I2V-A14B",
    ModelEnum.ti2v_5b.value: "Wan2.2-TI2V-5B",
    ModelEnum.animate_14b.value: "Wan2.2-Animate-14B",
    ModelEnum.s2v_14b.value: "Wan2.2-S2V-14B",
}


def request_to_job(
    req: VideoGenerationRequest,
    task_id: str,
    settings: Settings,
) -> Dict[str, Any]:
    """Build a flat job dict for ``generate.args_from_job_dict``."""
    model = req.model.value
    job: Dict[str, Any] = {"model": model}
    job.update(req.input.model_dump(exclude_none=True))
    job.update(req.parameters.model_dump(exclude_none=True))

    # Map VideoInput.video → src_root_path (generate.py uses --src_root_path, not --video)
    # Always remove "video" from job dict — generate.py doesn't have --video arg
    if "video" in job:
        video_val = job.pop("video")
        if "src_root_path" not in job:
            job["src_root_path"] = video_val

    # ckpt_dir: parameters.ckpt_dir overrides everything,
    # otherwise auto-append model-specific subdirectory to global ckpt_dir
    if job.get("ckpt_dir"):
        pass  # user explicitly specified, keep it
    elif settings.ckpt_dir:
        subdir = _MODEL_CKPT_SUBDIR.get(model)
        if subdir:
            job["ckpt_dir"] = f"{settings.ckpt_dir.rstrip('/')}/{subdir}"
        else:
            job["ckpt_dir"] = settings.ckpt_dir

    if not job.get("save_file"):
        job["save_file"] = f"{settings.output_dir.rstrip('/')}/{task_id}.mp4"

    # Fill default size if not provided
    if not job.get("size"):
        job["size"] = _MODEL_DEFAULT_SIZE.get(model, "832*480")

    # Auto-enable memory-saving defaults for A100 40GB / dual-expert models
    # FSDP shards model across GPUs — incompatible with offload_model
    # DDP (no FSDP) requires offload_model to fit 14B in 40GB
    nproc = settings.nproc_per_node
    nnodes = settings.nnodes
    world_size = nproc * nnodes

    if job.get("dit_fsdp") is None:
        job["dit_fsdp"] = True
    if job.get("t5_cpu") is None:
        job["t5_cpu"] = True
    if job.get("offload_model") is None:
        job["offload_model"] = False if job.get("dit_fsdp") else True

    return job