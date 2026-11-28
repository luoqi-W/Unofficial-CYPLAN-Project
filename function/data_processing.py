"""Load Census data, classify tracts and identify comparable areas."""

import pandas as pd
import geopandas as gpd


def load_census_tracts(path):
    # Use the population, housing counts and land area from each Census year.
    tracts = gpd.read_file(path)
    tracts = tracts.rename(columns={"NAME": "tract_name", "POP100": "population",
                                    "HU100": "housing_units", "AREALAND": "land_area_m2"})
    tracts = tracts[["GEOID", "tract_name", "population", "housing_units", "land_area_m2", "geometry"]]
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    for column in ["population", "housing_units", "land_area_m2"]:
        tracts[column] = pd.to_numeric(tracts[column], errors="coerce")

    if tracts["GEOID"].duplicated().any():
        raise ValueError("Duplicate Census tract GEOIDs.")
    if tracts[["population", "housing_units", "land_area_m2"]].isna().any().any():
        raise ValueError("Missing Census population, housing or land area.")
    if (tracts[["population", "housing_units", "land_area_m2"]] < 0).any().any():
        raise ValueError("Negative values in Census stock or area data.")

    tracts["land_area_km2"] = tracts["land_area_m2"] / 1000000
    return tracts.to_crs(epsg=4326)


def attach_display_geometry(tracts, path):
    # Use the same year's shoreline-clipped cartographic shapes for maps only.
    display_tracts = gpd.read_file(path)[["GEOID", "geometry"]].to_crs(epsg=4326)
    return display_tracts.merge(tracts.drop(columns="geometry"), on="GEOID", how="left", validate="one_to_one")


def select_sf_tracts(table):
    # Keep San Francisco County census tracts (state 06, county 075).
    table = table[table["GEO_ID"].str.startswith("1400000US06075", na=False)].copy()
    table["GEOID"] = table["GEO_ID"].str[-11:]
    return table


def clean_income_data(income):
    # B19013 is median household income. Keep usable estimates only.
    income = select_sf_tracts(income)
    income["median_income"] = pd.to_numeric(income["B19013_001E"].astype(str).str.replace(",", "").str.replace("+", "", regex=False), errors="coerce")
    income = income[["GEOID", "median_income"]].drop_duplicates()
    income = income[income["median_income"] > 0].copy()
    return income


def clean_income_2010(income):
    # The original 2010 download has one column per tract instead of one row.
    # Read the Census tract number and convert the export to the usual table format.
    rows = []
    for column in income.columns:
        if column.startswith("Census Tract ") and column.endswith("!!Estimate"):
            tract = column.split(",")[0].replace("Census Tract ", "")
            tract_code = f"{round(float(tract) * 100):06d}"
            rows.append(["1400000US06075" + tract_code, income[column].iloc[0]])
    income = pd.DataFrame(rows, columns=["GEO_ID", "B19013_001E"])
    return clean_income_data(income)


def clean_tenure_data(tenure):
    # B25003: renter-occupied units / all occupied housing units.
    tenure = select_sf_tracts(tenure)
    tenure["total"] = pd.to_numeric(tenure["B25003_001E"], errors="coerce")
    tenure["renter"] = pd.to_numeric(tenure["B25003_003E"], errors="coerce")
    tenure = tenure[["GEOID", "total", "renter"]].drop_duplicates()
    tenure = tenure[(tenure["total"] > 0) & tenure["renter"].between(0, tenure["total"])].copy()
    tenure["renter_share"] = tenure["renter"] / tenure["total"]
    return tenure[["GEOID", "renter_share"]]


def classify_tracts(tracts, income, tenure):
    # Keep tracts with both ACS measures when constructing the groups.
    groups = income.merge(tenure, on="GEOID", how="inner", validate="one_to_one")
    groups = groups[groups["GEOID"].isin(tracts["GEOID"])].copy()
    groups["income_group"] = pd.qcut(groups["median_income"], q=3, labels=["low", "middle", "high"])
    groups["renter_group"] = pd.qcut(groups["renter_share"], q=3, labels=["low_renter", "middle_renter", "high_renter"])

    # Keep unclassified tracts instead of dropping their housing and population.
    tracts = tracts.merge(groups, on="GEOID", how="left", validate="one_to_one")
    for column in ["income_group", "renter_group"]:
        tracts[column] = tracts[column].astype(object).fillna("unclassified")
    return tracts


def load_tract_relationships(path):
    # Official Census 2010–2020 tract relationship file (California).
    relationships = pd.read_csv(path, sep="|", dtype={"GEOID_TRACT_10": str, "GEOID_TRACT_20": str})
    relationships = relationships.rename(columns={"GEOID_TRACT_10": "GEOID_2010", "GEOID_TRACT_20": "GEOID_2020"})
    sf = relationships["GEOID_2010"].str.startswith("06075", na=False) & relationships["GEOID_2020"].str.startswith("06075", na=False)
    relationships = relationships[sf & (relationships["AREALAND_PART"] > 0)].copy()
    return relationships[["GEOID_2010", "GEOID_2020", "AREALAND_TRACT_10", "AREALAND_TRACT_20", "AREALAND_PART"]]


def match_comparable_tracts(relationships, min_overlap=0.99):
    # Keep an entire 2020 tract if at least 99% of its land is inside a 2010 tract.
    # This allows a 2010 tract to have multiple complete successors in 2020.
    relationships = relationships.copy()
    relationships["share_2020"] = relationships["AREALAND_PART"] / relationships["AREALAND_TRACT_20"]
    matches = relationships[relationships["share_2020"] >= min_overlap].copy()

    # Check how much of each 2010 tract is covered by its candidate successors.
    coverage = matches.groupby("GEOID_2010", as_index=False).agg(
        overlap_land_m2=("AREALAND_PART", "sum"),
        old_land_m2=("AREALAND_TRACT_10", "first"))
    coverage["share_2010"] = coverage["overlap_land_m2"] / coverage["old_land_m2"]
    coverage = coverage[coverage["share_2010"] >= min_overlap]
    matches = matches.merge(coverage[["GEOID_2010", "share_2010"]], on="GEOID_2010", how="inner")

    # No 2020 tract may contribute its counts to more than one 2010 area.
    if matches["GEOID_2020"].duplicated().any():
        raise ValueError("Ambiguous tract relationship.")
    return matches[["GEOID_2010", "GEOID_2020", "share_2010", "share_2020"]]


def build_growth_table(tracts_2010, tracts_2020, matches):
    # Sum whole successor tracts. Do not allocate or interpolate counts by area.
    before = tracts_2010.rename(columns={"GEOID": "GEOID_2010", "housing_units": "housing_2010",
                                         "population": "population_2010", "land_area_km2": "land_area_2010_km2"})
    after = tracts_2020[["GEOID", "housing_units", "population"]].rename(columns={"GEOID": "GEOID_2020",
                                                                        "housing_units": "housing_2020", "population": "population_2020"})
    after = matches.merge(after, on="GEOID_2020", how="left", validate="one_to_one")
    if after[["housing_2020", "population_2020"]].isna().any().any():
        raise ValueError("A matched 2020 tract is missing from the Census counts.")
    after = after.groupby("GEOID_2010", as_index=False).agg(
        housing_2020=("housing_2020", "sum"),
        population_2020=("population_2020", "sum"),
        tracts_2020=("GEOID_2020", "size"),
        share_2010=("share_2010", "first"),
        min_share_2020=("share_2020", "min"))
    master = before.merge(after, on="GEOID_2010", how="inner", validate="one_to_one")
    if len(master) != len(after):
        raise ValueError("A matched 2010 tract is missing from the Census counts.")
    return master[["GEOID_2010", "tract_name", "housing_2010", "housing_2020", "population_2010", "population_2020",
                   "land_area_2010_km2", "median_income", "renter_share", "income_group", "renter_group",
                   "tracts_2020", "share_2010", "min_share_2020", "geometry"]]
