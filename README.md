# Fill contracts for REST tool catalogs

Vendor REST catalogs tell an LLM **which** endpoint exists. They rarely say
**which earlier call** produces the path id that endpoint needs. The model
then invents an id, HTTP rejects the call, and the trace can still list every
gold API name.

This repository compiles two short contracts from a RestBench catalog (no gold
paths) and writes them into the slots a released DFSDT controller already
reads: a **fill** line on hop APIs (`choose_parameter` / `d_t`) and a
**response** line on search/list APIs. It does not train a rewriter and does
not change the controller.

**Research question.** Does a typed producer path in the fill slot raise
*execution-valid* Correct Path on live TMDB relative to (i) the vendor
catalog and (ii) DRAFT’s free-form purpose rewrite?

Preprint: [`paper/main.pdf`](paper/main.pdf) (source: `paper/main.tex`).

DRAFT and EasyTool rewrite the purpose slot. A learned one-field description
is scored under teacher forcing, where the gold API is supplied at every
step. DFSDT reads two strings: `choose_tool` sees `u_t`, then
`choose_parameter` sees `d_t`, then live HTTP can reject a missing id. The
object here is a typed producer path in that second slot.

## Locked TMDB result (9 seeds)

Execution-valid CP, leftover harness errors dropped from *n*. DeepSeek is
DeepSeek V4. Ling is Ling-3.0-Flash.

| | Initial | DRAFT | Ours |
| --- | ---: | ---: | ---: |
| Ling | 51.0 | 15.3 | **57.0** |
| DeepSeek V4 | **63.1** | 23.8 | 62.2 (tie vs Initial) |

Ours beats DRAFT on both families. Versus the catalog, Ling wins and DeepSeek
V4 matches. The fill line transfers; `not GET_` on the catalog selection
surface helps Ling and blocks DeepSeek V4. Seed-level numbers and CIs:
`artifacts/results/final_tables.json`.

## Repository layout

```
src/tooldoc_nir/     compiler, DFSDT driver, RestBench scoring
scripts/             freeze dumps, rebuild tables, audit bind vs gold
tests/               compiler and harness contracts
artifacts/documentation/
  TMDB_Ours.json     frozen fill/response payload used in the tables
  Spotify_Ours.json  same compiler, no sibling negation on search
  G3_nested.json     ToolBench G3 nested-slice index
paper/               preprint (PDF + TeX) and figures
```

Per-query traces and operator logs stay on the machine that ran them. They
are not required to read the result or to re-run a seed.

## Setup

Python 3.11+. Clone the released DRAFT tree (RestBench dumps +
`Inference_DFSDT.py`) into `external/DRAFT`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

`.env` (never commit it): `TMDB_API_KEY`, Spotify OAuth if you score that
split, and an OpenRouter key for the agent.

## Reproduce

Decoding is the released DFSDT setting: `T=0.2`, `top_p=1`, `max_tokens=2000`,
`retrieval_num=5`.

Compile the frozen documentation (does not read gold paths):

```powershell
$env:PYTHONPATH='src'
python scripts/rebuild_ours_docs.py
```

One TMDB seed, three arms (catalog / DRAFT / Ours):

```powershell
python -m tooldoc_nir.restbench_table --dataset TMDB --rows DFSDT DRAFT Ours --seed 1
```

The nine-seed Ling panel is seeds `1`–`9`; DeepSeek V4 uses the same seeds.
Do not retune the payload from the holdout or Screen A panels
(`artifacts/documentation/TMDB_*_protocol.json`).

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests -q
```
