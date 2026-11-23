"""Split a collected round file into batch text files for subagents (8 loops each).
usage: python split.py round<R>.json <R>"""
import json
import os
import sys

d = json.load(open(sys.argv[1]))
R = sys.argv[2]
HERE = os.getcwd()   # batches go next to the round file (story1_claude/ or a model subfolder)
os.makedirs(f"{HERE}/r{R}", exist_ok=True)
keys = sorted(d)
for b in range(0, len(keys), 8):
    with open(f"{HERE}/r{R}/batch{b // 8 + 1}.txt", "w") as f:
        f.write(f"SYSTEM INSTRUCTION FOR EVERY ITEM:\n{d[keys[b]]['system']}\n\n")
        for k in keys[b:b + 8]:
            f.write(f"===== ITEM {k} =====\n{d[k]['user']}\n\n")
print((len(keys) + 7) // 8, "batches in", f"{HERE}/r{R}")
