# Research methodology

## Study inputs and map matching

Restricted GPS sessions are assigned to county road networks. Each county
network combines OpenStreetMap topology with available FDOT/county attributes
such as road name, highway class, and speed context. Fast Map Matching (FMM)
maps a recorded trajectory to an ordered sequence of network feature IDs (FIDs).
An FID is an internal road-segment identifier, not a behavioral finding.

## Route change measures

For consecutive months, RCCI combines changes in the sets and weighted use of
matched road segments and transitions. Weighted overlap gives more importance
to segments repeatedly used in a month. RCCI is a descriptive measure of route
network change; it does not identify why the change occurred.

## OD and longitudinal analysis

Trips are summarized with explicit trip/session boundaries where available;
otherwise a documented inactivity rule is used. Spatially close endpoints are
clustered over several candidate radii and checked for stability. Common
origin–destination (OD) pairs are analyzed by month to distinguish:

- the same generalized destinations using different corridors;
- a new or disappearing destination; and
- a temporary deviation versus a sustained route transition.

For major OD pairs, road distance is summarized by OSM class (motorway, trunk,
primary, secondary, tertiary, residential, and service). This supports plain
language statements such as “more local-road travel,” provided the calculated
shares support it.

## Behavioral interpretation and uncertainty

Place roles combine timing, recurrence, dwell opportunity, OD centrality, and
map context. A nearby POI alone is never proof of a visit. Home is generalized
and assessed with morning departures, evening returns, recurrence, and
residential context; no exact address is published. Workplace, school/daycare,
and healthcare conclusions use conservative wording and are withheld if the
recording pattern is insufficient.

GPS cannot establish congestion, construction, toll avoidance, diagnosis,
employment, household membership, or trip purpose. See the results guide for
how confidence and limitations are communicated.
