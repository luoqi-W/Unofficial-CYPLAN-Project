#!/usr/bin/env python3
"""Step 3: Assign Census Tracts and sum completed housing units by tract.

Inputs:
  housing_production_located.csv
  Census_2020__Tracts_for_San_Francisco_20261020.csv

Outputs:
  housing_production_with_tract.csv
  tract_housing_growth.csv
"""

import argparse

import geopandas as gpd
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--housing", default="../raw_data/housing_production_located.csv")
    parser.add_argument(
        "--tracts", default="../raw_data/Census_2020__Tracts_for_San_Francisco.csv"
    )
    parser.add_argument("--detail-output", default="housing_production_with_tract.csv")
    parser.add_argument("--summary-output", default="tract_housing_growth.csv")
    args = parser.parse_args()

    # 1. Read the Step 2 output and 2020 Census Tract boundaries.
    housing = pd.read_csv(args.housing, dtype={"BlockLot": "string"}, low_memory=False)
    tracts = pd.read_csv(args.tracts, dtype={"GEOID": "string"}, low_memory=False)
    housing["Net Units Completed"] = pd.to_numeric(
        housing["Net Units Completed"], errors="raise"
    )

    assert housing["Net Units Completed"].notna().all(), "Missing housing-unit values."
    assert tracts["GEOID"].notna().all() and tracts["GEOID"].is_unique, (
        "Census Tract GEOIDs must be non-null and unique."
    )

    # Census Tract polygons are stored as WKT coordinates.
    tract_shapes = gpd.GeoDataFrame(
        tracts[["GEOID"]].copy(),
        geometry=gpd.GeoSeries.from_wkt(tracts["the_geom"], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    assert tract_shapes.geometry.notna().all() and tract_shapes.geometry.is_valid.all(), (
        "Census Tract geometries must be present and valid."
    )

    # 2. Convert located Housing records to points, preserving original row indices.
    latitude = pd.to_numeric(housing["latitude"], errors="coerce")
    longitude = pd.to_numeric(housing["longitude"], errors="coerce")
    has_coordinates = latitude.notna() & longitude.notna()

    points = gpd.GeoDataFrame(
        housing.loc[has_coordinates, []].copy(),
        geometry=gpd.points_from_xy(longitude[has_coordinates], latitude[has_coordinates]),
        crs="EPSG:4326",
    )

    # 3. Spatial join: assign each point the GEOID of the containing tract.
    joined = gpd.sjoin(points, tract_shapes, how="left", predicate="within")
    assert joined.index.is_unique, "A housing record matched multiple Census Tracts."

    detailed = housing.copy()
    detailed["GEOID"] = pd.Series(pd.NA, index=detailed.index, dtype="string")
    detailed.loc[joined.index, "GEOID"] = joined["GEOID"].astype("string")
    assigned = detailed["GEOID"].notna()

    # 4. Group records by tract and sum Net Units Completed.
    tract_sums = (
        detailed.loc[assigned]
        .groupby("GEOID")["Net Units Completed"]
        .sum()
    )
    summary = tracts[["GEOID"]].copy()
    summary["net_units_added"] = summary["GEOID"].map(tract_sums).fillna(0).astype(int)
    summary = summary.sort_values("GEOID").reset_index(drop=True)

    # 5. Validate: keep all original records; never double-count housing units.
    assert len(detailed) == len(housing), "Housing row count changed."
    assert detailed["Net Units Completed"].equals(housing["Net Units Completed"]), (
        "Housing unit values or row order changed."
    )
    assert summary["net_units_added"].sum() == detailed.loc[
        assigned, "Net Units Completed"
    ].sum(), "Tract aggregation changed the allocated housing-unit total."

    # 6. Export record-level GEOIDs and tract-level totals.
    detailed.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print("=== Step 3: Census Tract Aggregation ===")
    print("Total housing records:", len(housing))
    print("Records with coordinates:", int(has_coordinates.sum()))
    print("Records assigned to tracts:", int(assigned.sum()))
    print("Records without coordinates:", int((~has_coordinates).sum()))
    print("Located but outside/at tract boundaries:", int((has_coordinates & ~assigned).sum()))
    print("Net units assigned to tracts:", int(summary["net_units_added"].sum()))
    print("Net units without tract assignment:", int(detailed.loc[~assigned, "Net Units Completed"].sum()))
    print("Total Census Tracts:", len(summary))
    print("Tracts with zero located net units:", int(summary["net_units_added"].eq(0).sum()))
    print("Record-level output:", args.detail_output)
    print("Tract-level output:", args.summary_output)


if __name__ == "__main__":
    main()
