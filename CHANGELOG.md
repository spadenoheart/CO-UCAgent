# Changelog

All notable public changes to CO-UCAgent are documented in this file.

## [0.2.0] - 2026-09-21

### Added

- Verification-aware structured events and six-way test outcome classification.
- Hierarchical context management, stage state packages, failure-aware context, and observation masking.
- Cache-like long-term memory with stage prefetch, reuse metrics, and quality feedback.
- Verifier-grounded Repair Episode extraction, role-typed credit, strategy curation, and contract-based admission.
- Progress-aware runtime control and mutation-budget enforcement.
- Trace-tree compilation, interactive trajectory visualization, and live experiment monitoring.
- Resumable multi-DUT benchmark runner, frozen experiment manifests, token accounting, and stage replay.
- Provenance-preserving trajectory dataset construction and LLaMA-Factory export.
- Seven installable toolkit commands: `co-llm-profiler`, `co-context`, `co-memory-cache`, `co-trace`,
  `co-strategy`, `co-bench`, and `co-trajectory-data`.

### Changed

- Synchronized the public runtime with the newer UCAgent workflow, checker, Skill, TUI, Web, and archive interfaces.
- Unified public installation around the `co-ucagent` distribution and CLI while retaining `ucagent` compatibility.
- Corrected Makefile workspace routing and added a headless `make run_<DUT>` entry point.

### Compatibility

- Python 3.11 or newer.
- OpenAI-compatible model endpoints, including local Ollama deployments.
- Existing `ucagent` command lines remain supported.

## [0.1.0]

- Initial CO-UCAgent research release based on UCAgent.

[0.2.0]: https://github.com/spadenoheart/CO-UCAgent/releases/tag/v0.2.0
[0.1.0]: https://github.com/spadenoheart/CO-UCAgent/releases/tag/v0.1.0
