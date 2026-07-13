# Driver 1003 deliverables

The canonical technical report is:

`route_choice_change_index/visuals/driver_1003_route_choice_change_index_report.html`

For the reviewed public package—including the report, generalized verification
map, privacy-filtered JSON, and integrity manifest—start in
[`outputs/public/`](../../outputs/public/).

## Privacy boundary

Detailed monthly maps, comparison maps, route animations, timeline maps, and
intermediate RCCI data can expose route geometry, timestamps, or endpoint
locations. They are therefore local restricted research products and are not
part of the tracked public-release tree. A clone contains the source code,
documentation, tests, canonical report, and curated public outputs needed to
understand and reproduce the approved workflow; it intentionally does not
contain restricted participant-level artifacts.

Local approved-research copies may remain in the ignored paths listed in
[`legacy/README.md`](../../legacy/README.md). Do not add them back to Git or
publish them without a separate data-governance review.

`assets/` contains package-level CSS and the report manifest for the canonical
report. See `docs/DATA_AND_PRIVACY.md` for the full publication policy.
