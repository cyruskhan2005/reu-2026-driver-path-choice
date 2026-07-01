from __future__ import annotations

import base64
from pathlib import Path
import tempfile

from roadnet.html_assets import embed_local_html_assets


def test_local_img_src_is_embedded_as_base64_data_uri() -> None:
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        figure = root / "figures" / "fid_membership_map.png"
        figure.parent.mkdir()
        figure.write_bytes(one_pixel_png)
        document = '<html><body><img src="figures/fid_membership_map.png"></body></html>'
        embedded = embed_local_html_assets(document, root)
    assert 'src="data:image/png;base64,' in embedded
    assert "figures/fid_membership_map.png" not in embedded


def test_external_cdn_and_tile_urls_are_preserved() -> None:
    document = (
        '<link rel="stylesheet" href="https://cdn.example/style.css">'
        '<script src="https://cdn.example/app.js"></script>'
        '<img src="data:image/png;base64,abc">'
    )
    assert embed_local_html_assets(document, Path(".")) == document
