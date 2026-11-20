# Exposure record: binder sequences sent to a public server (first AlphaFold attempt, 18 Nov 2026)

**What happened.** The first AlphaFold-Multimer attempt (`a1`) ran ColabFold with its default MSA setting, which sends each chain of a complex to the
public ColabFold server (`api.colabfold.com`, an MMseqs2 service) to build alignments. The binder sequences are unpublished candidate designs. I chose
that setting without weighing that the sequences are unpublished; the run waited in that server's queue for 52 minutes per chunk and I then stopped it and
changed the method so that no sequence leaves the container (own target alignment, binder as a single sequence; a test now rejects any run in
which the server is contacted).

**What was sent.** Each ColabFold chunk handles its designs one after another, and the log of each chunk shows the first design's request in the
server queue when the chunk timed out or was stopped. So **at least 3 designs** (the first query of each of the 3 chunks) were certainly
sent, and **up to 12** (all designs of those 3 chunks) may have been. The exact number cannot be read back: the run folders are on the first Modal
account, which this session no longer has. The 12 designs are listed below. Only sequences were sent, not structures, names or any other data,
but the server may have logged or cached them; there is no way to retract them.

| Chunk | Design | In the final shortlist `s5` |
|---|---|---|
| c1 | `il7ra-rfd4-08f9233a8a90` | yes |
| c1 | `il7ra-rfd4-a9ca4147761f` | no |
| c1 | `il7ra-rfd4-8d368322e6bf` | no |
| c1 | `il7ra-rfd4-90b6d9399028` | yes |
| c2 | `il7ra-rfd4-67f05b49fd19` | no |
| c2 | `il7ra-rfd4-7fc8ab07c02f` | no |
| c2 | `il7ra-rfd4-13af2bd8fd8a` | yes |
| c2 | `il7ra-rfd4-2e2ec88e93b8` | yes |
| c3 | `il7ra-rfd4-9111856e8955` | no |
| c3 | `il7ra-rfd4-b25d020a592d` | yes |
| c3 | `il7ra-rfd4-b71bcf79d966` | yes |
| c3 | `il7ra-rfd4-bf3f3264bcd4` | no |

**Of these 12 designs, 6 are in the final shortlist** (08f9233a8a90, 90b6d9399028, 13af2bd8fd8a, 2e2ec88e93b8, b25d020a592d, b71bcf79d966).

**Every later run.** No later AlphaFold run contacted the server for a binder: the predict stage raises an error if the server is contacted, and
the decoy-alignment stage sends only the public GFP, lysozyme and IL-7 sequences (a test checks that no binder can enter that stage).

**Decision for the team (not for me).** If the unpublished status of these designs matters (competition rules, novelty checks against the
organizers' private database), treat the designs above as disclosed to a third party and decide whether to drop or replace them in the shortlist.
