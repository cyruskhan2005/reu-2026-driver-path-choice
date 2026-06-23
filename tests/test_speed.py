import numpy as np
import pandas as pd

from roadnet.speed import arbitrate_speed, resolve_connector_speeds


def test_connector_speed_is_rounded_to_nearest_5():
    network = pd.DataFrame(
        [
            {"u": 1, "v": 2, "highway": "primary", "osm_maxspeed": 45.0, "oneway": "yes"},
            {"u": 2, "v": 3, "highway": "motorway_link", "osm_maxspeed": np.nan, "oneway": "yes"},
            {"u": 3, "v": 4, "highway": "motorway", "osm_maxspeed": 65.0, "oneway": "yes"},
        ]
    )

    resolved = resolve_connector_speeds(network)

    assert resolved.loc[1, "connector_transition"] == "acceleration"
    assert resolved.loc[1, "osm_maxspeed"] == 60.0


def test_graph_inferred_speed_is_labeled_as_estimated():
    network = pd.DataFrame(
        [
            {
                "highway": "motorway_link",
                "osm_maxspeed": 60.0,
                "connector_transition": "acceleration",
                "name": "SB95 from BlueHeron",
            }
        ]
    )

    result = arbitrate_speed(network, has_fdot=False)

    assert result.loc[0, "estimated_speed_limit"] == 60.0
    assert result.loc[0, "speed_source"] == "graph_acceleration"
    assert bool(result.loc[0, "speed_limit_is_estimated"])
    assert result.loc[0, "speed_limit_label"] == "Estimated speed limit"


def test_direct_osm_speed_is_not_labeled_as_estimated():
    network = pd.DataFrame(
        [
            {
                "highway": "secondary",
                "osm_maxspeed": 35.0,
                "name": "North Ocean Drive",
            }
        ]
    )

    result = arbitrate_speed(network, has_fdot=False)

    assert result.loc[0, "estimated_speed_limit"] == 35.0
    assert result.loc[0, "speed_source"] == "osm"
    assert not bool(result.loc[0, "speed_limit_is_estimated"])
    assert result.loc[0, "speed_limit_label"] == "Speed limit"
