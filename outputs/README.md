# Output publication policy

`outputs/` is a private working directory by default. Analysis runs may place
exact trip endpoints, timestamps, cluster coordinates, reverse-geocoded
addresses, cached enrichment data, and detailed behavioral tables here. Those
files are intentionally ignored by Git.

Only `outputs/public/` is eligible for version control. Moving a file into that
directory is an explicit publication decision, not proof that the file is
anonymous or safe. Every public candidate still requires automated validation,
manual visual review where applicable, and the project's data-sharing approval.

## Artifact classes

| Class | Examples | Git policy |
|---|---|---|
| Private analysis | Trip summaries, raw or matched GPS, exact location clusters, POI-enriched cluster tables, API caches | Keep outside `outputs/public/`; never commit |
| Restricted research | Detailed recurring-destination and OD-route tables, visit dates/times, inferred activity records | Share only through approved restricted channels |
| Public candidate | Privacy-generalized behavior JSON, generalized verification map, curated RCCI report | Copy to `outputs/public/` only after review |
| Public manifest | `outputs/public/manifest.json` | Generate last and commit with the reviewed artifacts |

Named non-home places and route patterns can still identify a participant when
combined with outside information. “Public candidate” means the artifact meets
the repository's technical release checks; it does not replace an IRB, consent,
licensing, or data-governance decision.

## Generate the analysis

The server-side build reads Google credentials from the environment and writes
private working products under `outputs/`:

```bash
python scripts/build_driver_1003_real_world_behavior.py \
  --output-dir outputs \
  --cache-dir cache/google_maps
```

Do not copy the trip summary, exact cluster tables, POI-enriched cluster table,
API cache, inspection output, or raw research-process visuals into the public
directory.

## Prepare a curated public directory

After manually reviewing the generalized JSON, map, and report, copy only the
approved artifacts into `outputs/public/`. Keep the report and its local map at
package-relative paths so a clean archive does not depend on ignored files.

Create the deterministic manifest after the artifact set is final:

```bash
python scripts/generate_output_manifest.py \
  --root outputs/public \
  --output outputs/public/manifest.json
```

Validate without network access:

```bash
python scripts/validate_public_release.py \
  --public-dir outputs/public \
  --manifest outputs/public/manifest.json \
  --require-manifest
```

For a release-candidate run, also provide an ignored local JSON file containing
the exact private values that must not occur in public artifacts. The validator
reports labels only and never prints those values:

```bash
python scripts/validate_public_release.py \
  --public-dir outputs/public \
  --manifest outputs/public/manifest.json \
  --require-manifest \
  --forbidden-values-file /path/outside/repository/private_validation_values.json \
  --require-private-validation
```

The validation file can use labels such as `exact_home_address`,
`exact_home_coordinates`, and `exact_home_uri`. It must remain outside version
control. If `GOOGLE_MAPS_API_KEY` is present in the environment, the validator
also checks automatically that its value is absent without displaying it.

## Release checklist

1. Confirm every file in `outputs/public/` is intentional.
2. Run the validator with private validation values.
3. Manually inspect every map, image, animation, PDF, and slide deck.
4. Open the public directory from a clean export and test every local link.
5. Generate the manifest last, then run the manifest `--check` command.
6. Stage explicit paths only; never use a broad add command for research data.
