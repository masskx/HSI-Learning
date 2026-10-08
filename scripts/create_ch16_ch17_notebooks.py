"""Compatibility entry: rebuild ch16/17 from the canonical lecture templates.

Atomic execution preserves the old notebook if an execution error occurs.
Prepare results/teaching_ready first. No separate map patch step is required.
"""
from build_lecture_notebooks import build

if __name__ == '__main__':
    build(['16_protonet_teaching.ipynb', '17_openset_teaching.ipynb'])
