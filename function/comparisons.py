"""Housing growth, area normalization and population normalization."""

import numpy as np
import pandas as pd


def area_normalization(df, housing_column, area_column, result_column):
    # Housing units per square kilometer of land. Exclude invalid denominators.
    df = df.copy()
    valid = (df[area_column] > 0) & np.isfinite(df[area_column])
    df[result_column] = np.nan
    df.loc[valid, result_column] = df.loc[valid, housing_column] / df.loc[valid, area_column]
    return df


def population_normalization(df, housing_column, population_column, result_column):
    # Housing units per 1,000 residents.
    df = df.copy()
    valid = (df[population_column] > 0) & np.isfinite(df[population_column])
    df[result_column] = np.nan
    df.loc[valid, result_column] = df.loc[valid, housing_column] / df.loc[valid, population_column] * 1000
    return df


def net_housing_change(housing_2010, housing_2020):
    # Net change in the original housing counts.
    return housing_2020 - housing_2010


def net_housing_growth_rate(housing_2010, housing_2020):
    # Percentage change relative to the initial housing stock.
    if housing_2010 <= 0:
        return np.nan
    return (net_housing_change(housing_2010, housing_2020) / housing_2010) * 100


def calculate_original_growth(df):
    # Apply the same functions to tract counts and group totals.
    df = df.copy()
    df["net_housing_change"] = net_housing_change(df["housing_2010"], df["housing_2020"])
    df["housing_growth_rate"] = df.apply(
        lambda row: net_housing_growth_rate(row["housing_2010"], row["housing_2020"]), axis=1)
    return df


def calculate_housing_growth(df):
    # Compare housing stock at the two dates, keeping signed changes.
    df = calculate_original_growth(df)

    # The 2010 land area and population are reference denominators for growth.
    df = area_normalization(df, "net_housing_change", "land_area_2010_km2", "net_units_per_km2")
    df = population_normalization(df, "net_housing_change", "population_2010", "net_units_per_1000_initial_residents")

    # Housing availability per resident at each date uses that year's population.
    df = population_normalization(df, "housing_2010", "population_2010", "housing_per_1000_2010")
    df = population_normalization(df, "housing_2020", "population_2020", "housing_per_1000_2020")
    df["housing_per_1000_growth_rate"] = np.nan
    valid = df["housing_per_1000_2010"] > 0
    df.loc[valid, "housing_per_1000_growth_rate"] = (df.loc[valid, "housing_per_1000_2020"] - df.loc[valid, "housing_per_1000_2010"]) / df.loc[valid, "housing_per_1000_2010"] * 100
    return df


def compare_housing_growth(df, group_column):
    # Keep 2010 groups fixed. Compare original housing counts, not per-capita counts.
    summary = df.groupby(group_column, as_index=False).agg(
        tracts_2010=("GEOID_2010", "size"),
        tracts_2020=("tracts_2020", "sum"),
        housing_2010=("housing_2010", "sum"),
        housing_2020=("housing_2020", "sum"))
    return calculate_original_growth(summary)


def compare_snapshot_groups(df, group_column, year):
    # Describe each year's own groups using original housing counts.
    summary = df.groupby(group_column, as_index=False).agg(
        tracts=("GEOID", "size"),
        housing_units=("housing_units", "sum"))
    summary["year"] = year
    return summary


def compare_citywide(tracts_2010, tracts_2020):
    # The county is the reference; no tract matching is necessary for county totals.
    summary = pd.DataFrame({"area": ["San Francisco County"],
                            "housing_2010": [tracts_2010["housing_units"].sum()],
                            "housing_2020": [tracts_2020["housing_units"].sum()],
                            "population_2010": [tracts_2010["population"].sum()],
                            "population_2020": [tracts_2020["population"].sum()],
                            "land_area_2010_km2": [tracts_2010["land_area_km2"].sum()]})
    return calculate_housing_growth(summary)


def summarize_coverage(master, tracts_2010, tracts_2020):
    # Report coverage so results from the matched areas are not called citywide.
    summary = pd.DataFrame({"year": [2010, 2020],
                            "total_tracts": [len(tracts_2010), len(tracts_2020)],
                            "included_tracts": [len(master), master["tracts_2020"].sum()],
                            "total_housing": [tracts_2010["housing_units"].sum(), tracts_2020["housing_units"].sum()],
                            "included_housing": [master["housing_2010"].sum(), master["housing_2020"].sum()],
                            "total_population": [tracts_2010["population"].sum(), tracts_2020["population"].sum()],
                            "included_population": [master["population_2010"].sum(), master["population_2020"].sum()]})
    summary["housing_coverage_pct"] = summary["included_housing"] / summary["total_housing"] * 100
    summary["population_coverage_pct"] = summary["included_population"] / summary["total_population"] * 100
    return summary


def rank_tracts(df, column, n=10):
    # Return the largest and smallest valid values, without separate rank columns.
    columns = ["GEOID_2010", "tract_name", "housing_2010", "housing_2020", "net_housing_change", column]
    valid = df[df[column].notna()]
    top = valid.nlargest(n, column)[columns]
    bottom = valid.nsmallest(n, column)[columns]
    return top, bottom


def explain_map_comparison(tracts_2010, tracts_2020, column, denominator, label):
    # County ratios use total counts; invalid tract ratios are reported separately.
    for year, tracts in [(2010, tracts_2010), (2020, tracts_2020)]:
        valid = tracts[tracts[column].notna()]
        highest = valid.loc[valid[column].idxmax()]
        scale = 1000 if denominator == "population" else 1
        total = tracts["housing_units"].sum() / tracts[denominator].sum() * scale
        print(f"{year}: county ratio = {total:,.1f} {label}; "
              f"highest tract = {highest['tract_name']} ({highest[column]:,.1f}); "
              f"{tracts[column].isna().sum()} undefined tract ratios.")


def explain_growth_map(master, column, label):
    # Describe the observed range without treating ratios as causal evidence.
    valid = master[master[column].notna()]
    highest = valid.loc[valid[column].idxmax()]
    lowest = valid.loc[valid[column].idxmin()]
    print(f"Highest: {highest['tract_name']} ({highest[column]:,.1f} {label}).")
    print(f"Lowest: {lowest['tract_name']} ({lowest[column]:,.1f} {label}).")
    print(f"Positive: {(valid[column] > 0).sum()}; negative: {(valid[column] < 0).sum()}; "
          f"zero: {(valid[column] == 0).sum()}; undefined: {master[column].isna().sum()}.")


def explain_group_comparison(summary, group_column):
    # Identify the groups with the largest and smallest observed changes.
    groups = summary[summary[group_column] != "unclassified"]
    for column, label, suffix in [("net_housing_change", "net change", " units"),
                                  ("housing_growth_rate", "growth rate", "%")]:
        valid = groups.dropna(subset=[column])
        high = valid.loc[valid[column].idxmax()]
        low = valid.loc[valid[column].idxmin()]
        print(f"{label.capitalize()}: highest = {high[group_column]} ({high[column]:,.2f}{suffix}); "
              f"lowest = {low[group_column]} ({low[column]:,.2f}{suffix}).")


def explain_snapshot_comparison(summary, group_column):
    # These are separate cross-sections; group members can change between years.
    for year, table in summary.groupby("year"):
        table = table[table[group_column] != "unclassified"]
        highest = table.loc[table["housing_units"].idxmax()]
        print(f"{year}: {highest[group_column]} has the largest housing stock "
              f"({highest['housing_units']:,.0f} units).")
