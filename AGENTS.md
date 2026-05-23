# AGENTS.md

Project-level instructions for all AI coding sessions in this repository.

These instructions are mandatory. If they conflict with a direct user message in the current session, follow the user message and then update the project plan or instructions only if the user explicitly asks.

## Project Source Of Truth

- `SAM2_PEFT_Project_Plan.md` is the complete execution plan for this project.
- Before making project changes, read the relevant phase in `SAM2_PEFT_Project_Plan.md`.
- Do not skip phases, pass thresholds, chapter problems, or visualization outputs unless the user explicitly changes the project scope.
- If implementation details are unclear, ask before coding.
- Treat every phase as incomplete until its listed checks pass with real outputs.

## Hard Completion Criteria

The project is not complete until all of the following are achieved and documented with real experiment results:

1. SAM2, approximately 300M parameters, is adapted for humanoid robot component segmentation.
2. Parameter-efficient fine-tuning is implemented with adapter layers and mask-decoder training.
3. Fewer than 2% of SAM2 parameters are updated during the PEFT run.
4. The dataset contains only about 350-500 annotated humanoid robot component images.
5. The work demonstrates strong segmentation under severe annotation scarcity by leveraging pretrained SAM2 representations.
6. The PEFT model recovers at least 85% of the full fine-tune mIoU improvement over zero-shot SAM2.
7. The results include evidence that PEFT resists catastrophic forgetting on out-of-distribution objects better than, or at least equal to, full fine-tuning.
8. A four-way benchmark is completed and documented:
   - zero-shot SAM2
   - ViT baseline
   - SAM2 PEFT
   - full SAM2 fine-tune
9. The final benchmark quantifies the annotation-efficiency boundary using mIoU, trainable parameter count, and latency.
10. The final README/report contains the results table, key visualizations, reproduction steps, and a concise project summary.

The project plan's stronger goal of recovering more than 90% of full fine-tune mIoU remains the target. The 85% recovery threshold above is the minimum claim that must be satisfied for the professor-facing project statement.

## Execution Rules

- Work phase by phase: Phase 0 through Phase 6 in `SAM2_PEFT_Project_Plan.md`.
- For each phase, implement only what is needed to pass that phase's chapter problems.
- Save the required visualizations into `viz/` using the filenames specified in the plan.
- Keep experiment metrics reproducible. Record commands, configs, seeds where practical, checkpoint paths, and raw result files.
- Use the same test split and mIoU implementation across all benchmark methods.
- Do not compare methods using different datasets, prompts, preprocessing, or evaluation code unless the difference is explicitly documented and justified.
- Prefer Colab/A100 execution for heavy SAM2 training work, as stated in the plan.
- Local code should be lightweight, reproducible, and suitable for packaging into Colab workflows.

## Planning And Parallelization

- When planning next steps, analyze the workflow for independent tasks that can run in parallel.
- If spawning multiple agents can perform those independent tasks in parallel and save a meaningful amount of time, proactively recommend delegating those tasks to multiple agents.
- Each delegation recommendation must include the detailed analysis behind it: the tasks that can run independently, what each agent would own, expected time saved, coordination risks, and why parallelization is or is not worthwhile.
- Do not recommend parallel agents for trivial, tightly coupled, or low-benefit work where coordination overhead outweighs the likely time savings.

## Engineering Constraints

- Keep changes minimal and directly tied to the current phase.
- Do not add speculative features, broad abstractions, or unrelated cleanup.
- Match the existing project style once files exist.
- Preserve user-created files and experiment outputs unless the user explicitly approves deletion.
- Do not fabricate metrics, figures, checkpoints, or experiment results.
- If a metric is not measured yet, write `TBD` or leave it incomplete rather than inventing a value.
- When a check fails, stay in that phase and debug before moving forward.

## Branch And Git Workflow

- If this folder is initialized as a Git repository, create a new branch for each feature or phase-level change.
- Do not make direct feature changes on `main`.
- Commit after each completed, verified feature or phase.
- Keep `README.md` updated as the project becomes reproducible.
- If this folder is not yet a Git repository, state that clearly before relying on branch or commit workflow.

## Terminal Safety

Run read-only inspection, local tests, and normal development commands freely.

Ask the user before running destructive, remote-mutating, system-level, or secret-touching commands, including:

- `rm` or `rmdir`
- `git reset --hard`
- `git clean -f` or `git clean -fd`
- `git push`
- `sudo`
- `chmod` or `chown`
- `dd`, `mkfs`, or `mount`
- `crontab` or `systemctl`
- mutating API calls such as `curl -X POST`, `PUT`, or `DELETE`
- package publishing commands
- `docker run` or `kubectl apply`
- `ssh`, `scp`, or remote `rsync`
- commands that read, write, print, or modify `.env` or other secret files

## Session Startup Checklist

At the start of each coding session:

1. Confirm the current working directory.
2. Check whether the folder is a Git repository and identify the branch if available.
3. Read this `AGENTS.md`.
4. Read the relevant section of `SAM2_PEFT_Project_Plan.md`.
5. State assumptions, the current phase, and success checks before editing.

## Current Project Status

Phase 0 has a T4 smoke-test artifact, but the official A100 target-environment check is still pending. Phase 1 now has a four-class COCO segmentation dataset in `dataset/` with `arm`, `leg`, `torso`, and `head`; validation, visualization, and dataloader smoke checks pass. Phase 2 zero-shot SAM2 evaluation tooling exists and has passed a one-image local MPS smoke test, but official full-test metrics should be generated on Colab/A100. The ViT baseline remains blocked until the original ViT model code/checkpoint is available.
