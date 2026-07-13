# Data, privacy, and publication policy

## Classification

| Class | Examples | Release rule |
|---|---|---|
| Restricted input | Raw GPS, sensor sessions, source GIS, matched trajectories | Never commit or publish |
| Restricted analysis | Exact endpoints, trip summaries, cluster tables, visit timing, API caches | Keep in ignored local storage |
| Public-reviewed output | Generalized report, map, JSON, road-class summary, manifest | Publish only after automated and visual review |

## Home and participant privacy

The public report must not reveal an exact home address, exact home coordinate,
or home-centered Google Maps URI. The code generalizes home context to a broad
area. Named non-home POIs can still create re-identification risk when combined
with other information; publication requires the applicable data-use and IRB
review, not just a passing script.

## Secrets and Google Maps

`GOOGLE_MAPS_API_KEY` is read only from the server environment in network mode.
It must never be placed in source, YAML, HTML, JSON, logs, notebooks, or cache
files. `config.yaml`, `.env*`, `cache/`, credentials, and outputs are ignored.
The public validator scans common credential signatures and can compare public
artifacts with an ignored local forbidden-values file without printing values.

## Output policy

Only `outputs/public/` is eligible for a public commit. Run:

```bash
python scripts/validate_public_release.py --public-dir outputs/public --require-manifest
python scripts/generate_output_manifest.py --root outputs/public --output outputs/public/manifest.json --check
```

Detailed historical maps and local GIS are intentionally excluded from the
tracked tree. Their prior Git history may still require a separate history
rewrite before converting an existing remote into a fully open public archive.

## Attribution and usage

FMM, OpenStreetMap/OSMnx, FDOT, county GIS, Mapillary, and Google Maps Platform
retain their respective terms and attribution requirements. No repository-wide
license is asserted here because source-data redistribution rights vary. Use
the project only with approval from the repository owner and governing data-use
agreement.
