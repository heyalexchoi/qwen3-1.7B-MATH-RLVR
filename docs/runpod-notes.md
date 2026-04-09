# RunPod Lessons Learned

## Base Image
- **Use:** `runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04` (or latest stable)
- **NOT:** `pytorch:2.1.0` — too old for modern transformers, requires PyTorch 2.4+

## Pod Reuse Policy
- Create ONE pod for entire pipeline
- Scripts 00-05 all run on same instance
- Only stop when pipeline complete or explicitly asked

## Polling Requirements
- Long-running jobs need cron-based poll or background `nohup` + periodic checks
- Don't rely on single long poll — connections drop
- Use `nohup` + log tail for anything >10 min

## This Run
- Pod: `ewl8ddikromfav` (RTX 3090, $0.22/hr)
- Baseline: 74.60% (higher than expected 40-50%)
- Output: `outputs/baseline_results.json` (896KB, full per-example traces)

## Next Steps (same pod)
- Script 02: Generate Claude traces (requires ANTHROPIC_API_KEY)
- Script 03: SFT training
- Script 04: GRPO training
- Script 05: Final eval
