#!/usr/bin/env python3
"""Thin wrapper: rescore the rerun traces with math-verify."""
import sys
sys.argv = ['09_rescore_mathverify.py', '--input',
            '/home/dev/.openclaw/workspace/qwen3-gsm8k-demo/data/traces/qwen32b_math_traces_rerun.jsonl']
exec(open('/home/dev/.openclaw/workspace/qwen3-gsm8k-demo/scripts/09_rescore_mathverify.py').read())
