# Troubleshooting

This is the short entry point for the verified macOS workflow. The detailed,
command-level record remains in [TROUBLESHOOTING_MACOS_FMM.md](TROUBLESHOOTING_MACOS_FMM.md).

| Symptom | First check | Verified direction |
|---|---|---|
| `fmm` or `ubodt_gen` missing | `command -v fmm` | Re-run `scripts/bootstrap_macos.sh`; activate `pipeline` |
| `dyld` library error | architecture and loader paths | Use the pinned Apple-Silicon FMM build; do not mix Intel binaries |
| Python/FMM exits 139 | `python -I -c 'import fmm'` | Use the documented Python 3.11 arm64 build and loader-relative patch |
| Enriched network missing | `verify_pipeline_outputs.py --stage enrichment` | Run the full pipeline or use an approved cache/reuse mode |
| Existing matched CSV | wrapper error message | Choose `--overwrite-matched` or `--reuse-matched` explicitly |
| Report inputs missing | Phase 2 paths in the report log | Rebuild Phase 2 or use `--reuse-phase2` only after validation |
| Google request blocked | `--google-mode` and environment | Use `offline`/`cache`, or correct billing/API authorization without exposing the key |

Always run a wrapper with `--help` first. See the macOS guide for clean versus
cached runs and the detailed troubleshooting guide for segmentation faults,
architecture mismatch, missing libraries, external volumes, and safe resume.
