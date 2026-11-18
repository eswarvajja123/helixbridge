"""The multiple-sequence alignment for AlphaFold-Multimer, built from the target's own alignment (Phase 5, Task 5.4). Pure Python.

A de novo binder has no homologs, so the only informative alignment is the target's. This builds the ColabFold complex-a3m for one design
from the 1,022-row target-only MSA the team already holds plus the binder as a single sequence. Nothing is sent to a public MSA server,
so a candidate sequence leaves the container nowhere, and the run does not wait in anyone's queue.

Layout (ColabFold's own serialized form): `#<len target>,<len binder>\t1,1`, the query row (target + binder), every target homolog row
padded with gaps over the binder columns, and the binder query on its own row padded with gaps over the target columns.
"""
AMINO = set("ACDEFGHIKLMNPQRSTVWY")


def read_rows(text):
    """[(header, aligned sequence)] from an A3M; insertions (lowercase) and '.' dropped so every row has the query's length."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            rows.append([line, ""])
        elif line and rows:
            rows[-1][1] += line
    return [(h, "".join(c for c in s if not c.islower() and c != ".")) for h, s in rows]


def complex_a3m(target_a3m, target, binder):
    """The a3m text for one complex. Raises ValueError with a plain reason if the alignment is not for this target."""
    if not target or set(target) - AMINO:
        raise ValueError("The target must be canonical amino acids")
    if not binder or set(binder) - AMINO:
        raise ValueError("The binder must be canonical amino acids")
    rows = read_rows(target_a3m)
    if not rows or rows[0][1] != target:
        raise ValueError("The alignment's first row is not the target sequence")
    bad = [h for h, s in rows if len(s) != len(target)]
    if bad:
        raise ValueError(f"{len(bad)} alignment rows do not have the target's length")
    gap_binder, gap_target = "-" * len(binder), "-" * len(target)
    out = [f"#{len(target)},{len(binder)}\t1,1", ">101\t102", target + binder]
    seen = {target}
    for header, seq in rows[1:]:
        if seq in seen or not set(seq) - {"-"}:                       # an exact repeat, or a row with no residues, adds nothing
            continue
        seen.add(seq)
        out += [header.split()[0], seq + gap_binder]
    out += [">102", gap_target + binder]
    return "\n".join(out) + "\n"
