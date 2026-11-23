"""Write the exact prompts reader.py would send (family format + show.py view of one network), one JSON per set,
for readers that answer outside this pipeline (e.g. Claude subagents). No data or truth is included.
usage: python build_reader_prompts.py <out.json> <set> [<set> ...]"""
import json
import subprocess
import sys

import reader as R
import s3

out, sets = sys.argv[1], sys.argv[2:]
items = []
for st in sets:
    for law in s3.laws_in(st):
        fam = law.split("_")[0]
        for N in s3.NS:
            for seed in (0, 1, 2):
                nf = f"{s3.ROOT}/sets/{st}/nets/{law}_n{N}_s{seed}.json"
                view = subprocess.run([sys.executable, f"{s3.HERE}/show.py", nf], capture_output=True, text=True).stdout
                items.append(dict(id=f"{st}__{law}__n{N}__s{seed}", set=st, law=law, fam=fam, N=N, seed=seed,
                                  system=R.SYSTEM, prompt=R.FORMAT[fam] + "\n\nNetwork:\n" + view))
json.dump(items, open(out, "w"), indent=1)
print(len(items), "prompts ->", out)
