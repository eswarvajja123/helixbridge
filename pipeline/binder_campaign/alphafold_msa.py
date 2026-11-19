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


def _homolog_rows(a3m, query, seen):
    """The aligned rows of `a3m` (insertions removed) other than its query, skipping repeats and rows with no residues."""
    rows = read_rows(a3m)
    if not rows or rows[0][1] != query:
        raise ValueError("The alignment's first row is not the sequence it is meant for")
    if any(len(s) != len(query) for _, s in rows):
        raise ValueError("Some alignment rows do not have the query's length")
    out = []
    for header, seq in rows[1:]:
        if seq in seen or not set(seq) - {"-"}:
            continue
        seen.add(seq)
        out.append((header.split()[0], seq))
    return out


def complex_a3m(target_a3m, target, binder, binder_a3m=None):
    """The a3m text for one complex. The binder is a single sequence unless `binder_a3m` (its own alignment) is given.
    Raises ValueError with a plain reason if an alignment is not for its sequence."""
    if not target or set(target) - AMINO:
        raise ValueError("The target must be canonical amino acids")
    if not binder or set(binder) - AMINO:
        raise ValueError("The binder must be canonical amino acids")
    try:
        target_rows = _homolog_rows(target_a3m, target, {target})
    except ValueError as error:
        message = str(error)
        raise ValueError("The alignment's first row is not the target sequence" if "first row" in message else message.replace("query's", "target's")) from None
    gap_binder, gap_target = "-" * len(binder), "-" * len(target)
    out = [f"#{len(target)},{len(binder)}\t1,1", ">101\t102", target + binder]
    for header, seq in target_rows:
        out += [header, seq + gap_binder]
    if not target_rows:                                # no homologs: the target still needs its own unpaired row, which ColabFold requires
        out += [">101", target + gap_binder]
    out += [">102", gap_target + binder]
    if binder_a3m:
        for header, seq in _homolog_rows(binder_a3m, binder, {binder}):
            out += [header, gap_target + seq]
    return "\n".join(out) + "\n"


def single_a3m(target):
    """A one-row alignment (the sequence alone), for a target that has no alignment of its own: decoy proteins, or the real target in a
    like-for-like comparison with them."""
    if not target or set(target) - AMINO:
        raise ValueError("The target must be canonical amino acids")
    return f">query\n{target}\n"


def alignment_for(alignments, target, mode="deep"):
    """The alignment text to use for one target. `alignments` is one A3M text (for its own query) or a {sequence: A3M text} mapping.
    'deep' uses the alignment held for exactly this target; any target without one, and every target in 'single' mode, gets the sequence alone."""
    if mode not in ("deep", "single"):
        raise ValueError("mode must be 'deep' or 'single'")
    if isinstance(alignments, str):
        alignments = {read_rows(alignments)[0][1]: alignments}
    if mode == "deep" and target in alignments:
        return alignments[target]
    return single_a3m(target)
