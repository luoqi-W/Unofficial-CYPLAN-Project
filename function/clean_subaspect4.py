#!/usr/bin/env python3
"""Subaspect 4: compare full 2005–2025 housing growth by 2004/2023 land use.

Run: python clean_subaspect4.py
All paths are relative to this script, not the terminal's working directory.
Inputs in input/:
  planning_landuse.shp (with .shx, .dbf and .prj alongside it)
  land_use_2023.csv
  housing_production_located_recovered.csv
Outputs ONLY 3 CSVs in output/:
  landuse_comparison_2004_2023.csv
  landuse_transitions_2004_2023.csv
  housing_growth_2005_2025_common_sample.csv
Requires pandas, geopandas, shapely, numpy.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"

CATEGORY_MAP = {
    "RESIDENT": "Residential", "MIXRES": "Mixed Residential",
    "MIXED": "Mixed Nonresidential", "MIPS": "Nonresidential",
    "RETAIL/ENT": "Nonresidential", "PDR": "Nonresidential",
    "CIE": "Nonresidential", "MED": "Nonresidential",
    "VISITOR": "Nonresidential", "VACANT": "Vacant",
    "OPENSPACE": "Open Space", "ROW": "Right-of-Way",
    "UNKNOWN": "Unknown",
}
EXCLUDED = {"Unknown", "Unmatched"}


def save(df, name, index=False):
    df.to_csv(OUTPUT / name, index=index)


def clean_landuse(raw, year, parcel_col, use_col):
    """One land-use category per parcel; reject conflicting duplicate classifications."""
    df = raw[[parcel_col, use_col]].rename(
        columns={parcel_col: "mapblklot", use_col: "landuse"}
    ).copy()
    df["mapblklot"] = df["mapblklot"].astype("string").str.strip()
    df["landuse"] = df["landuse"].astype("string").str.strip().str.upper()
    df["mapblklot"] = df["mapblklot"].replace("", pd.NA)
    df["landuse"] = df["landuse"].replace({"": pd.NA, "NONE": pd.NA})
    df = df.dropna(subset=["mapblklot"])

    conflicts = df.groupby("mapblklot")["landuse"].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError(f"{year}: {(conflicts > 1).sum()} parcels have conflicting uses")

    df = df.drop_duplicates("mapblklot").copy()
    df["landuse"] = df["landuse"].fillna("UNKNOWN")
    unexpected = set(df["landuse"]) - set(CATEGORY_MAP)
    if unexpected:
        raise ValueError(f"{year}: unexpected LANDUSE codes: {unexpected}")
    df["category"] = df["landuse"].map(CATEGORY_MAP)
    df["year"] = year
    return df[["year", "mapblklot", "landuse", "category"]].reset_index(drop=True)


def prepare_polygons(raw, parcel_col, use_col):
    """Normalize geometry categories, keep duplicated geometries, repair if invalid."""
    polygons = raw[[parcel_col, use_col, "geometry"]].rename(
        columns={parcel_col: "mapblklot", use_col: "landuse"}
    ).copy()
    polygons["category"] = (
        polygons["landuse"].astype("string").str.strip().str.upper()
        .map(CATEGORY_MAP).fillna("Unknown")
    )
    # Check empty geometries before missing values to avoid GeoPandas warnings.
    polygons = polygons.loc[~polygons.geometry.is_empty].copy()
    polygons = polygons.loc[polygons.geometry.notna()].copy()
    bad = ~polygons.geometry.is_valid
    if bad.any():
        polygons.loc[bad, "geometry"] = polygons.loc[bad, "geometry"].apply(make_valid)
    if not polygons.geometry.is_valid.all():
        raise ValueError("Polygon geometry repair failed")
    return polygons[["mapblklot", "category", "geometry"]]


def compare_landuse(a, b):
    """Entire-city composition and same-parcel transition matrix."""
    combined = pd.concat([a, b], ignore_index=True)
    counts = pd.crosstab(combined["category"], combined["year"])
    summary = pd.DataFrame({
        "count_2004": counts[2004], "count_2023": counts[2023],
    })
    summary["share_2004"] = summary["count_2004"] / len(a) * 100
    summary["share_2023"] = summary["count_2023"] / len(b) * 100
    summary["count_change"] = summary["count_2023"] - summary["count_2004"]
    summary["share_change_pp"] = summary["share_2023"] - summary["share_2004"]

    paired = a[["mapblklot", "landuse", "category"]].merge(
        b[["mapblklot", "landuse", "category"]], on="mapblklot",
        suffixes=("_2004", "_2023"), validate="one_to_one",
    )
    valid = paired.loc[
        ~paired["category_2004"].isin(EXCLUDED)
        & ~paired["category_2023"].isin(EXCLUDED)
    ].copy()
    valid["category_changed"] = valid["category_2004"] != valid["category_2023"]
    valid["detailed_use_changed"] = valid["landuse_2004"] != valid["landuse_2023"]
    transitions = pd.crosstab(valid["category_2004"], valid["category_2023"])
    return summary.round(2), valid, transitions, len(paired)


def match_housing(housing, land, polygons, year):
    """Parcel-ID match first; spatial fallback for unmatched records only."""
    result = housing.merge(
        land[["mapblklot", "category"]], left_on="BlockLot", right_on="mapblklot",
        how="left", validate="many_to_one", indicator=True,
    )
    exact = result["_merge"].eq("both")
    cat_col, method_col = f"landuse_{year}_category", f"landuse_{year}_match_method"
    result[cat_col] = result["category"].fillna("Unmatched")
    result[method_col] = np.where(exact, "parcel_id", "unmatched")

    can_locate = (
        ~exact & result["latitude"].between(-90, 90)
        & result["longitude"].between(-180, 180)
    )
    candidates = result.loc[can_locate]
    if not candidates.empty:
        pts = gpd.GeoDataFrame(
            candidates[["housing_row_id"]].copy(),
            geometry=gpd.points_from_xy(candidates["longitude"], candidates["latitude"]),
            crs="EPSG:4326",
        ).to_crs(polygons.crs)
        hits = gpd.sjoin(pts, polygons, how="inner", predicate="within")
        if not hits.empty:
            stats = hits.groupby("housing_row_id")["category"].agg(["size", "nunique"])
            # 2004 has multiple polygons per parcel; accept only if categories agree.
            accepted = (stats["nunique"] == 1) if year == 2004 else (stats["size"] == 1)
            ids = stats.index[accepted]
            chosen = (hits.loc[hits["housing_row_id"].isin(ids)]
                      .drop_duplicates("housing_row_id").set_index("housing_row_id"))
            result.loc[chosen.index, cat_col] = chosen["category"].to_numpy()
            result.loc[chosen.index, method_col] = "spatial"
            result.loc[stats.index[~accepted], method_col] = "ambiguous_spatial"

    assert len(result) == len(housing)
    assert result["Net Units Completed"].sum() == housing["Net Units Completed"].sum()
    return result.drop(columns=["mapblklot", "category", "_merge"])



def compare_full_period(housing, matched_2004, matched_2023):
    """Compare the SAME 2005–2025 housing records using both land-use snapshots.

    2004/2023 are land-use snapshot years, NOT housing-growth periods.
    There is only one housing completion period: 2005–2025.
    """
    paired = housing[[
        "housing_row_id", "BPA", "BlockLot", "First Completion Date", "Net Units Completed"
    ]].merge(
        matched_2004[["housing_row_id", "landuse_2004_category"]],
        on="housing_row_id", validate="one_to_one",
    ).merge(
        matched_2023[["housing_row_id", "landuse_2023_category"]],
        on="housing_row_id", validate="one_to_one",
    )
    assert len(paired) == len(housing)

    paired = paired.loc[
        ~paired["landuse_2004_category"].isin(EXCLUDED)
        & ~paired["landuse_2023_category"].isin(EXCLUDED)
    ].copy()

    dates = pd.to_datetime(
        paired["First Completion Date"].astype("string").str.strip(),
        format="mixed", errors="coerce",
    )
    valid_dates = dates.notna() & dates.dt.year.between(2005, 2025)
    if (~valid_dates).any():
        problematic = paired.loc[~valid_dates]
        print(f"WARNING: {len(problematic):,} records "
              f"({problematic['Net Units Completed'].sum():,.0f} net units) "
              "excluded because completion dates are invalid or outside 2005–2025.")
        paired = paired.loc[valid_dates].copy()
    if paired.empty:
        raise ValueError("No records in common sample with valid 2005–2025 completion dates")

    # Same housing projects and net units, once under each snapshot's categories.
    blocks = []
    for year in (2004, 2023):
        tmp = paired[["Net Units Completed", f"landuse_{year}_category"]].copy()
        tmp.columns = ["net_units", "category"]
        tmp["period"] = "2005-2025"
        tmp["snapshot"] = year
        tmp["additions"] = tmp["net_units"].clip(lower=0)
        tmp["losses"] = -tmp["net_units"].clip(upper=0)
        blocks.append(tmp)

    summary = pd.concat(blocks, ignore_index=True).groupby(
        ["period", "snapshot", "category"], as_index=False
    ).agg(
        records=("net_units", "size"),
        additions=("additions", "sum"),
        losses=("losses", "sum"),
        net_units=("net_units", "sum"),
    )
    common_net = paired["Net Units Completed"].sum()
    city_net = housing["Net Units Completed"].sum()
    summary["share_common_sample_pct"] = (summary["net_units"] / common_net * 100).round(2)
    summary["share_citywide_pct"] = (summary["net_units"] / city_net * 100).round(2)

    for year in (2004, 2023):
        rows = summary.loc[summary["snapshot"] == year]
        assert rows["records"].sum() == len(paired)
        assert rows["net_units"].sum() == common_net
        assert (rows["additions"] - rows["losses"] == rows["net_units"]).all()
    return paired, summary


def main():
    # Check every input before processing or writing any results.
    required = (
        "planning_landuse.shp", "planning_landuse.shx",
        "planning_landuse.dbf", "planning_landuse.prj",
        "land_use_2023.csv", "housing_production_located_recovered.csv",
    )
    missing = [str(INPUT / name) for name in required if not (INPUT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    old = gpd.read_file(INPUT / "planning_landuse.shp")  # EPSG:2227
    new_raw = pd.read_csv(
        INPUT / "land_use_2023.csv", usecols=["mapblklot", "landuse", "the_geom"],
        dtype={"mapblklot": "string"}, low_memory=False,
    )
    land04 = clean_landuse(old, 2004, "MAPBLKLOT", "LANDUSE")
    land23 = clean_landuse(new_raw, 2023, "mapblklot", "landuse")

    comp, matched, matrix, matched_count = compare_landuse(land04, land23)
    save(comp, "landuse_comparison_2004_2023.csv", index=True)
    save(matrix, "landuse_transitions_2004_2023.csv", index=True)
    print(f"Land use: {len(land04):,} parcels (2004), {len(land23):,} (2023)")
    print(f"Same parcel IDs: {matched_count:,}; known in both: {len(matched):,}")
    print(f"Changed broad categories: {matched['category_changed'].sum():,}")
    print("\nLand-use shares and changes:\n", comp.to_string())

    housing = pd.read_csv(
        INPUT / "housing_production_located_recovered.csv",
        dtype={"BPA": "string", "BlockLot": "string"}, low_memory=False,
    ).reset_index(drop=True)
    housing["housing_row_id"] = np.arange(len(housing))
    housing["BlockLot"] = housing["BlockLot"].str.strip()
    for col in ("latitude", "longitude", "Net Units Completed"):
        housing[col] = pd.to_numeric(housing[col], errors="coerce")
    if housing["Net Units Completed"].isna().any():
        raise ValueError("Housing Production has missing net housing units")
    city_total = housing["Net Units Completed"].sum()
    print(f"\nHousing Production: {len(housing):,} records, {city_total:,.0f} net units")

    old_poly = prepare_polygons(old, "MAPBLKLOT", "LANDUSE")
    new_poly = gpd.GeoDataFrame(
        new_raw[["mapblklot", "landuse"]].copy(),
        geometry=gpd.GeoSeries.from_wkt(new_raw["the_geom"], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    new_poly = prepare_polygons(new_poly, "mapblklot", "landuse")

    joined = {}
    for year, land, polygons in ((2004, land04, old_poly), (2023, land23, new_poly)):
        result = match_housing(housing, land, polygons, year)
        joined[year] = result
        cat_col = f"landuse_{year}_category"
        known = ~result[cat_col].isin(EXCLUDED)
        print(f"\n{year} match methods:\n"
              f"{result[f'landuse_{year}_match_method'].value_counts().to_string()}")
        print(f"Known categories: {known.sum():,}/{len(result):,} records; "
              f"{result.loc[known, 'Net Units Completed'].sum():,.0f}/"
              f"{city_total:,.0f} net units")

    common, full_period = compare_full_period(housing, joined[2004], joined[2023])
    save(full_period, "housing_growth_2005_2025_common_sample.csv")
    print(f"\nCommon sample (2005–2025): {len(common):,} records, "
          f"{common['Net Units Completed'].sum():,.0f} net units")
    print("\n=== Housing growth for ONE period: 2005–2025 ===")
    print(full_period.pivot_table(
        index="category", columns="snapshot", values="net_units",
        aggfunc="sum", fill_value=0,
    ).to_string())
    print(f"\nOnly three output CSVs were written to: {OUTPUT}")


if __name__ == "__main__":
    main()
