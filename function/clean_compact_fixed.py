#!/usr/bin/env python3
"""Subaspect 4: compare 2004/2023 land-use snapshots and 2005–2025 housing production.

Run from anywhere: python clean.py
Inputs (relative to this script): raw/planning_landuse.shp,
raw/land_use_2023.csv, data/processed/housing_production_located_recovered.csv.
Outputs: data/processed/*.csv. Requires pandas, geopandas, shapely, numpy.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

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
    df.to_csv(OUT / name, index=index)


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


def summarize_housing(data, year):
    cat_col = f"landuse_{year}_category"
    df = data[[cat_col, "Net Units Completed"]].copy()
    df["additions"] = df["Net Units Completed"].clip(lower=0)
    df["losses"] = -df["Net Units Completed"].clip(upper=0)
    summary = df.groupby(cat_col).agg(
        records=("Net Units Completed", "size"),
        additions=("additions", "sum"), losses=("losses", "sum"),
        net_units=("Net Units Completed", "sum"),
    ).reset_index().rename(columns={cat_col: "category"})
    summary["landuse_year"] = year
    summary["share_citywide_pct"] = (
        summary["net_units"] / data["Net Units Completed"].sum() * 100
    ).round(2)
    return summary


def compare_periods(housing, matched_2004, matched_2023):
    """Same records classified in both snapshots; compare completion periods."""
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
    # Different rows may use ISO timestamps, ISO dates, or US-style dates.
    # pandas 2+ may infer one format from the first row and reject the rest.
    dates = pd.to_datetime(
        paired["First Completion Date"].astype("string").str.strip(),
        format="mixed", errors="coerce"
    )
    valid_dates = dates.notna() & dates.dt.year.between(2005, 2025)
    if (~valid_dates).any():
        problematic = paired.loc[~valid_dates].copy()
        problematic["date_issue"] = np.where(
            dates.loc[~valid_dates].isna(), "unparseable_or_missing", "outside_2005_2025"
        )
        save(problematic, "housing_period_date_issues.csv")
        print(f"\nWARNING: {len(problematic):,} common-sample records have invalid "
              "or out-of-period completion dates, totaling "
              f"{problematic['Net Units Completed'].sum():,.0f} net units. "
              "Excluded from period comparisons ONLY; see "
              "data/processed/housing_period_date_issues.csv.")
        print("Sample problematic dates:",
              problematic["First Completion Date"].head(10).tolist())
        paired = paired.loc[valid_dates].copy()
        dates = dates.loc[valid_dates]
    if paired.empty:
        raise ValueError("No common-sample housing records with valid 2005–2025 dates")
    paired["period"] = np.where(dates.dt.year <= 2014, "2005-2014", "2015-2025")
    blocks = []
    for year in (2004, 2023):
        df = paired[["period", "Net Units Completed", f"landuse_{year}_category"]].copy()
        df.columns = ["period", "net_units", "category"]
        df["snapshot"] = year
        blocks.append(df)
    result = pd.concat(blocks, ignore_index=True).groupby(
        ["period", "snapshot", "category"], as_index=False
    ).agg(records=("net_units", "size"), net_units=("net_units", "sum"))
    result["years"] = result["period"].map({"2005-2014": 10, "2015-2025": 11})
    result["net_units_per_year"] = (result["net_units"] / result["years"]).round(2)
    assert all(result.loc[result["snapshot"] == y, "net_units"].sum() == paired["Net Units Completed"].sum()
               for y in (2004, 2023))
    return paired, result


def main():
    old = gpd.read_file(RAW / "planning_landuse.shp")  # EPSG:2227
    new_raw = pd.read_csv(
        RAW / "land_use_2023.csv", usecols=["mapblklot", "landuse", "the_geom"],
        dtype={"mapblklot": "string"}, low_memory=False,
    )
    land04 = clean_landuse(old, 2004, "MAPBLKLOT", "LANDUSE")
    land23 = clean_landuse(new_raw, 2023, "mapblklot", "landuse")
    save(land04, "landuse_2004_clean.csv")
    save(land23, "landuse_2023_clean.csv")

    comp, matched, matrix, matched_count = compare_landuse(land04, land23)
    save(comp, "landuse_comparison_2004_2023.csv", index=True)
    save(matched, "landuse_matched_parcels.csv")
    save(matrix, "landuse_transitions_2004_2023.csv", index=True)
    print(f"Land use: {len(land04):,} parcels (2004), {len(land23):,} (2023)")
    print(f"Same parcel IDs: {matched_count:,}; known in both: {len(matched):,}")
    print(f"Changed broad categories: {matched['category_changed'].sum():,}")
    print("\nLand-use shares and changes:\n", comp.to_string())

    housing = pd.read_csv(
        OUT / "housing_production_located_recovered.csv",
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
    summaries = []
    for year, land, polygons in ((2004, land04, old_poly), (2023, land23, new_poly)):
        result = match_housing(housing, land, polygons, year)
        joined[year] = result
        save(result, f"housing_landuse_{year}_matches.csv")
        summary = summarize_housing(result, year)
        assert len(result) == len(housing) and summary["net_units"].sum() == city_total
        summaries.append(summary)
        cat = f"landuse_{year}_category"
        known = ~result[cat].isin(EXCLUDED)
        print(f"\n=== {year} Housing Production / Land Use ===")
        print("Match methods:\n", result[f"landuse_{year}_match_method"].value_counts().to_string())
        print(f"Known category: {known.sum():,}/{len(result):,} records; "
              f"{result.loc[known, 'Net Units Completed'].sum():,.0f}/{city_total:,.0f} net units")
        print(summary.to_string(index=False))
    save(pd.concat(summaries, ignore_index=True), "housing_growth_by_landuse_2004_2023.csv")

    common, periods = compare_periods(housing, joined[2004], joined[2023])
    save(common, "housing_landuse_common_sample.csv")
    save(periods, "housing_growth_by_period_and_landuse.csv")
    print(f"\nCommon sample: {len(common):,} records, "
          f"{common['Net Units Completed'].sum():,.0f} net units")
    print("\nNet units by completion period and snapshot:\n",
          periods.pivot_table(index=["period", "category"], columns="snapshot",
                              values="net_units", aggfunc="sum", fill_value=0).to_string())
    print(f"\nSaved CSV files in: {OUT}")


if __name__ == "__main__":
    main()
