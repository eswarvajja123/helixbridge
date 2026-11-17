"""Build the PUBLIC sequence database used by the novelty pre-screen (Phase 5, Task 5.1).

Sources (both public): UniProtKB/Swiss-Prot (reviewed entries, every organism) from rest.uniprot.org, and the PDB's protein sequences
from files.rcsb.org (`pdb_seqres.txt.gz`, protein chains of at least 20 residues, identical sequences kept once). The result is an
MMseqs2 database plus a manifest recording what was downloaded, when, with SHA-256 and counts. This is NOT the organizers' database:
see novelty.py for what a search against it can and cannot show.
"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .runner import write_json

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=fasta&compressed=true"
PDB_URL = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
OK_LETTERS = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
MIN_PDB_LENGTH = 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, timeout=900):
    request = urllib.request.Request(url, headers={"User-Agent": "helixbridge-novelty-db"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(path, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    with gzip.open(path, "rb") as check:                     # a truncated download fails here instead of later
        while check.read(1 << 20):
            pass


def combine(swissprot_gz, pdb_gz, out_fasta):
    """Write one FASTA from the two downloads. Returns the counts."""
    counts = {"swissprot": 0, "pdb_protein_chains": 0, "pdb_unique_sequences": 0}
    seen = set()
    with open(out_fasta, "w") as out:
        with gzip.open(swissprot_gz, "rt") as stream:
            for line in stream:
                out.write(line)
                counts["swissprot"] += line.startswith(">")
        header = None
        with gzip.open(pdb_gz, "rt") as stream:
            for line in stream:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    header = line
                    continue
                if header is None or "mol:protein" not in header:
                    continue
                sequence = line.strip()
                if len(sequence) < MIN_PDB_LENGTH or set(sequence) - OK_LETTERS:
                    continue
                counts["pdb_protein_chains"] += 1
                key = hashlib.sha1(sequence.encode()).digest()
                if key in seen:
                    continue
                seen.add(key)
                counts["pdb_unique_sequences"] += 1
                out.write(">pdb|" + header[1:].split()[0] + "\n" + sequence + "\n")
    if counts["swissprot"] == 0 or counts["pdb_unique_sequences"] == 0:
        raise ValueError("A download held no sequences; refusing to build an empty database")
    return counts


def build(out_dir, binary="mmseqs", fetch=download, log=print):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    out_dir.mkdir(parents=True)
    swiss, pdb = out_dir / "swissprot.fasta.gz", out_dir / "pdb_seqres.txt.gz"
    log("downloading Swiss-Prot ...")
    fetch(UNIPROT_URL, swiss)
    log("downloading PDB protein sequences ...")
    fetch(PDB_URL, pdb)
    counts = combine(swiss, pdb, out_dir / "public.fasta")
    log("building the MMseqs2 database ...")
    for command in ([binary, "createdb", str(out_dir / "public.fasta"), str(out_dir / "publicDB"), "-v", "1"],
                    [binary, "createindex", str(out_dir / "publicDB"), str(out_dir / "tmp"), "-v", "1"]):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError("MMseqs2 failed: " + (done.stderr or done.stdout).strip()[-400:])
    version = subprocess.run([binary, "version"], capture_output=True, text=True).stdout.strip()
    manifest = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mmseqs_version": version, **counts,
                "sources": [{"url": UNIPROT_URL, "sha256": sha256_file(swiss), "bytes": swiss.stat().st_size},
                            {"url": PDB_URL, "sha256": sha256_file(pdb), "bytes": pdb.stat().st_size}],
                "database": str(out_dir / "publicDB"), "fasta_sha256": sha256_file(out_dir / "public.fasta"),
                "scope": "Public sequences only (Swiss-Prot + PDB protein chains). Not the organizers' database."}
    write_json(out_dir / "public-db-manifest.json", manifest)
    return manifest
