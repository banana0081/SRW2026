# Execution-Grounded Canonical Tool Documentation

Local research prototype for the NIR project:
**“Adapting tool documentation for usage in Large Language Model systems.”**

## Research question

Can a canonical, execution-grounded representation of tool documentation:

1. reduce tool/API selection errors relative to Raw documentation and DRAFT; and
2. preserve successful tool use when live APIs become unavailable or their
   request/response contracts change?

The project extends DRAFT's trial-and-error refinement with typed multi-field
documentation, evidence-gated updates and explicit temporal state. Retrieval,
selection and post-change execution are all primary evaluation targets.

## Hypotheses

- **H1 — selection:** under the same candidates, retriever, agent and evaluation
  model, canonical execution-refined documentation reduces wrong tool/API
  choices and improves NDCG/Recall and Correct Path Rate relative to Raw and
  DRAFT.
- **H2 — temporal robustness:** after the same bounded trial budget, the proposed
  pipeline improves executable/path success under contract drift or API removal
  relative to Raw and DRAFT.
- **H3 — update safety:** typed evidence-backed patches reduce unsupported
  documentation facts, false-down decisions and exact-contract errors relative
  to DRAFT's free-form rewrite.
- **H4 — generalization and efficiency:** documents refined with one model retain
  gains across evaluation model families without an unacceptable token or
  latency increase.

## Method under test

```text
raw API documentation
  -> MFTR-compatible canonical fields + typed contract snapshot v0
  -> bounded diverse calls to a real, replayed, simulated, or controlled API
  -> timestamped typed observations
  -> liveness aggregation + observed contract snapshot
  -> deterministic structural delta
  -> evidence-gated documentation patch v1
  -> field-aware retrieval and downstream tool selection/use
```

The canonical schema contains:

- MFTR-compatible retrieval fields: description, parameters, response and
  grounded examples;
- identity: category, tool, API name, endpoint and HTTP method;
- executable request/response contracts with requiredness, types and defaults;
- constraints: alternatives, limits, enumerations and dependencies;
- temporal state: liveness, confidence, checked-at time and time-to-live;
- provenance: source fields, evidence IDs, observation timestamps and
  transformation version.

The structural delta is not a free-form rewrite. It records typed changes such
as parameter additions/removals, requiredness or type changes, HTTP method
changes, output-schema changes and liveness transitions. Natural-language
documentation is generated only after the delta has been validated.

## Evaluation protocol

RestBench tool-use CP% uses the released `Inference_DFSDT.py` controller and
live vendor HTTP. Retrieval, ToolBench G3, and the temporal split are the
remaining evaluation targets.

### Experiment 1 — DRAFT-compatible retrieval

Reproduce DRAFT's Raw/DRAFT comparison on RestBench TMDB and Spotify with BM25
and Contriever, then add Ours under the same candidate universe. Extend the
evaluation to the ToolBench retrieval set. Report NDCG, Recall, MRR and direct
wrong-selection rates.

### Experiment 2 — DRAFT-compatible end-to-end tool use

Use the released 100-query ToolBench G3/I3-Instruction test and the same
DFSDT/StableToolBench harness for Raw, DRAFT and Ours. Report Correct Path Rate,
wrong-tool/API rates, parameter validity, executable-call rate and, secondarily,
StableToolBench SoPR/SoWR.

### Experiment 3 — temporal extension

Apply held-out contract and liveness changes to ToolBench-seeded APIs. DRAFT and
Ours receive the same five-observation budget before evaluation on fresh
queries. Test parameter additions/removals/renames/types, method and response
changes, endpoint deletion, transient failures, auth failures, rate limits and
recovery. Include unchanged controls and live alternatives.

Primary dynamic metrics are post-change Correct Path Rate, executable-call rate,
contract-error rate, fallback-selection rate, false-down rate, exact patch match
and unsupported-fact rate.

Every transformation is cached and versioned. Evaluation queries are not used
during refinement. Failed or ambiguous observations remain `unknown`; they are
not silently converted into documentation claims.

## Integration smoke test

The first GLM-5.2 run only verifies that a schema patch reaches the tool
definition and that an explicit `UNAVAILABLE` label suppresses a call:

Copy `.env.example` to `.env`, set `OPENROUTER_API_KEY`, then run:

```powershell
python -m tooldoc_nir.llm_eval `
  --model z-ai/glm-5.2:free `
  --provider Decart `
  --cases-per-kind 8
```

The command checkpoints every response to
`artifacts/results/glm52_free_initial.json` and resumes completed pairs after
transient provider failures. `.env` and result artifacts are ignored by Git.

The run is not evidence for H1 or H2 and is excluded from research results.

## RestBench tool-use table

End-to-end CP% vs DRAFT on RestBench-TMDB and RestBench-Spotify uses the
released `Inference_DFSDT.py` controller and live vendor HTTP:

```powershell
$env:PYTHONPATH='src'
python -m tooldoc_nir.restbench_table --dataset TMDB --rows DFSDT DRAFT Ours
python -m tooldoc_nir.restbench_table --dataset Spotify --rows DFSDT DRAFT Ours
```

Ours appends hop fill-contracts to `Initial` (`src/tooldoc_nir/restbench_ours.py`).
Locked summaries: `artifacts/results/restbench_tmdb_table`,
`restbench_tmdb_ours_v3`, `restbench_tmdb_flash`, `restbench_spotify_ling`.

## Claim boundary

Beating DRAFT on static retrieval alone is insufficient because MFTR already
studies multi-field canonicalization. The intended contribution is the combined
result: competitive selection on unchanged tools plus measurably safer and more
successful behavior after temporal contract or availability changes.

