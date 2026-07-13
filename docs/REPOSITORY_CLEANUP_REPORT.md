# Repository cleanup and public-release report

Status: curated-boundary implementation completed 2026-07-13; historical-risk
review remains open. This document does not authorize history rewriting,
publication beyond `outputs/public/`, or a change in repository visibility.

## Executive finding

The current working repository is suitable for continued private research but
is not ready to publish as a complete open dataset. The new generalized
Driver 1003 behavior JSON and verification map passed direct checks for API-key
material, an exact home coordinate, and a precise home address. The surrounding
repository still mixes public reports, private analysis inputs, large generated
data, duplicate delivery trees, and older route maps created before the current
home-privacy standard.

The recommended release unit is therefore a small curated artifact package,
not the entire working tree. `outputs/public/` is the explicit publication
boundary. All other `outputs/` content remains private by default.

## Evidence snapshot

The audit observed the following worktree state before release-hygiene files
were added:

| Category | File count | Approximate size |
|---|---:|---:|
| Tracked files | 460 | 1,097.7 MiB |
| Modified tracked files | 68 | mixed source and generated deliverables |
| Untracked, not ignored | 241 | 171.6 MiB |
| Ignored local files | 1,252 | 12,007.7 MiB |
| Git object directory | — | 362 MiB |

Tracked content was dominated by approximately 571 MiB of deliverables and
525 MiB of GIS data. Untracked content was dominated by generated
research-process HTML, CSV, PNG, GIF, and frame sequences. Ignored content was
dominated by local network builds, routing tables, raw/derived GPS products,
caches, and copied delivery bundles.

No Git LFS configuration was present. Several committed individual blobs were
tens of megabytes, and the tracked deliverable tree contained more than one
hundred standalone HTML files.

## Git topology

At audit time:

- `main` and `origin/main` referenced the same commit with no ahead/behind
  difference.
- `main` was 21 commits ahead of `upstream/main` and zero behind it.
- Both configured remotes used GitHub and neither remote URL embedded
  credentials.
- Several historical feature, review, backup, and redundant phase branches
  remained, along with a local stash.
- No release tag was present.

Release work should occur on a dedicated branch from `main` and target
`origin`. Branch deletion, stash deletion, or upstream synchronization is a
separate maintenance decision.

## Secret and configuration audit

The audit searched the current worktree and all Git refs for the current Google
credential value and common Google, GitHub, AWS, Mapillary, and private-key
signatures. No credential match was found. No environment file, private-key
file, or credential file was found in the working tree. Google enrichment cache
files were under an ignored cache directory.

The repository did contain a tracked `config.yaml`. Its current token field was
placeholder-like, but tracking a writable local configuration file creates a
future secret-commit risk. The safe end state is:

- keep `config.example.yaml` tracked;
- ignore `config.yaml` and local variants;
- remove `config.yaml` from the Git index only in a separately reviewed change;
- continue reading Google credentials from the environment only.

Adding an ignore rule does not untrack a file that is already committed.

## Driver 1003 privacy findings

### Public candidates that passed direct checks

The following working outputs passed direct scans for the exact private home
values used during validation and for API credential material:

- `outputs/driver_1003_real_world_behavior_insights.json`
- `outputs/driver_1003_poi_route_insights_map.html`

The public JSON's likely-home object contained generalized evidence and no
populated exact-address, exact-coordinate, or exact-location map-link fields.
The map used a generalized home area and privacy-clipped route geometry. These
checks establish compliance with the repository's technical policy; they do
not make detailed mobility patterns anonymous.

The curated RCCI report also passed the direct secret and exact-home scans, but
its interactive-map link pointed to an ignored working-output path. It was not
portable from a tracked-only archive and therefore was not release-ready.

### Private and restricted products

The following output classes must remain outside the public boundary:

- trip-level summaries with exact endpoints and timestamps;
- location-cluster tables with centroid or medoid coordinates;
- POI-enriched cluster tables and reverse-geocoded private fields;
- Google response caches;
- raw or matched GPS data;
- debugging, inspection, and intermediate frame files.

Detailed recurring-destination and OD-route-change tables did not contain the
exact home values checked during the audit, but they expose granular routines,
visit timing, named non-home locations, and route behavior. They are classified
as restricted research data rather than open-public artifacts.

### Legacy route-map risk

Automated proximity screening found 66 already-tracked monthly or comparison
HTML maps with route geometry inside the private home privacy buffer. Those
maps do not need to label a residence for repeated route endpoints and
corridors to create a re-identification risk. The existing documentation's
description of the whole Driver 1003 folder as a clean share folder is therefore
not compatible with the newer privacy standard.

If the repository is or becomes public, removing those files in a later commit
does not remove prior versions from Git history. Appropriate remediation may
require access restriction, privacy-regenerated maps, a new sanitized release
repository, or a coordinated history rewrite. A history rewrite must not be
performed unilaterally because it affects every clone and branch.

## Duplicate and generated outputs

Two differing tracked copies of the RCCI report were present: the canonical
Driver 1003 copy and an older Google-Drive-bundle copy. Ignore rules do not hide
files that are already tracked. The canonical release source should be
`deliverables/driver_1003/`; the tracked bundle copy should be removed or
synchronized in a separate reviewed cleanup.

The audit also found more than two hundred untracked research-process files,
including raw-GPS visualizations, large HTML pages, images, animations, matrix
exports, and per-frame PNG sequences. These are reproducibility or review
inputs, not publication artifacts. Curated figures should be copied explicitly
to `outputs/public/` after privacy review rather than committing the generation
tree.

## Documentation and manifest gaps

The existing deliverable README files and report manifest described the Phase 3
package but did not list the v2 behavior JSON or verification map. The manifest
also lacked artifact hashes, privacy classifications, source revision, API
request/cache counts, and a validation record.

A public release should include:

- a deterministic SHA-256 manifest generated from the final public directory;
- a public/restricted/private classification for each output class;
- the source Git revision and generation command;
- aggregate API request and cache-hit counts, never credentials or request URLs;
- the privacy-validation result and manual visual-review record;
- package-relative links verified from a clean export.

The repository had no CI workflow, no declared pytest/dev dependency, no
release tag, and no repository-level license or citation file. A targeted
seven-test behavior suite passed with the standard-library unittest runner. A
broader unittest discovery run passed eight tests, but the repository contained
many additional pytest-style tests and pytest was not installed in the active
environment. Full-suite release validation was therefore incomplete.

## Release hygiene implemented by this change

This hygiene change is intentionally non-destructive. It adds:

- private-by-default output ignores with `outputs/public/` as the opt-in
  publication boundary;
- local config, secret, environment, cache, log, inspection, and generated
  research-process ignore rules;
- `scripts/generate_output_manifest.py` for deterministic public manifests;
- `scripts/validate_public_release.py` for offline structural, secret, privacy,
  link, and manifest checks;
- an offline CI workflow for the release tooling and public boundary;
- `outputs/README.md` documenting artifact policy and release commands.

It does not delete, untrack, stage, commit, relocate, or publish an artifact.

## Implemented commit sequence

The dedicated v2 branch now contains separate behavior, longitudinal, report,
and cleanup checkpoints. `outputs/public/` contains only the generalized report,
map, behavior JSON, policy README, and deterministic manifest. Package-relative
links, private-value validation, manifest `--check`, and the non-empty offline CI
gate pass. Unrelated regenerated comparisons, animations, and exploratory
scripts remain outside these commits.

Repository visibility, legacy-map remediation, license selection, participant
data-sharing approval, and third-party POI publication policy remain governance
decisions rather than implementation tasks.

## Remaining conditions for an open-repository release

- Legacy tracked maps predate the current home-privacy boundary and remain in
  Git history even though they are excluded from `outputs/public/`.
- Repository-wide licensing, participant-data sharing approval, and third-party
  POI publication policy require an owner/governance decision.
- A GitHub-hosted clean-clone CI run has not yet been observed; the same offline
  workflow passes locally.

Until those conditions are resolved, use restricted project channels rather
than presenting the repository as an anonymous public mobility dataset.
