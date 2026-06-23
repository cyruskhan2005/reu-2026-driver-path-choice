"""Regression coverage for the SWIG MatchResult lifetime in split matching."""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MIAMI_DIR = ROOT / "sflorida_outputs" / "Miami_Dade_County"
GPS_PATH = MIAMI_DIR / "Miami-Dade County_gps.csv"
SHP_PATH = MIAMI_DIR / "fmm" / "edges.shp"
UBODT_PATH = MIAMI_DIR / "fmm" / "ubodt.txt"


def _match_trip_one_prefix(result_pipe) -> None:
    from fmm import FastMapMatch, FastMapMatchConfig, Network, NetworkGraph, UBODT

    from roadnet.fmm_pipeline import _match_with_splits

    gps = pd.read_csv(GPS_PATH, sep=";")
    trip = gps.loc[gps["id"] == 1, ["lon", "lat"]].head(261)
    coords = list(trip.itertuples(index=False, name=None))

    network = Network(str(SHP_PATH), "fid", "u", "v")
    graph = NetworkGraph(network)
    ubodt = UBODT.read_ubodt_csv(str(UBODT_PATH))
    model = FastMapMatch(network, graph, ubodt)
    config = FastMapMatchConfig(
        16, 500 / 110000, 100 / 110000, reverse_tolerance=1
    )

    results = _match_with_splits(coords, model, config)
    result_pipe.send(len(results))
    result_pipe.close()
    os._exit(0)


class FMMResultLifetimeTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("fmm") is not None,
        "FMM Python bindings are not installed",
    )
    def test_split_match_keeps_native_result_alive_for_miami_trip_one(self) -> None:
        """The old temporary-result expression segfaulted on this trace."""
        if not all(path.exists() for path in (GPS_PATH, SHP_PATH, UBODT_PATH)):
            self.skipTest("Miami-Dade FMM regression artifacts are unavailable")

        ctx = mp.get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_match_trip_one_prefix, args=(child_pipe,))
        process.start()
        child_pipe.close()
        process.join(60)

        if process.is_alive():
            process.kill()
            process.join()
            self.fail("isolated split match exceeded 60 seconds")

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(parent_pipe.recv(), 261)
        parent_pipe.close()


if __name__ == "__main__":
    unittest.main()
