#!/usr/bin/env python3
""" Assign parcel-centroid coordinates to Housing Production records.

Matches use Parcel IDs only. First, join against individual parcels. 
Then, for still-unmatched records, look up IDs inside multi-parcel
lists. Housing rows and net unit counts are
preserved, including records without a location.
"""

import argparse
import re

import pandas as pd

# Examples of accepted multi-parcel fields:
#   Complete:             '3791002', '3791001', '3790001'
#   Trailing comma:       '7148054', '7148060',
#   Partial last ID:      '5054A114', '5054A130', '5054ACOM
# Only the fully enclosed IDs (e.g., 5054A114, 5054A130) are used.
COMPLETE_LIST = re.compile(r"'[A-Za-z0-9]+'(?:\s*,\s*'[A-Za-z0-9]+')+")
TRAILING_COMMA = re.compile(COMPLETE_LIST.pattern + r",\s*")
TRAILING_PARTIAL_ID = re.compile(COMPLETE_LIST.pattern + r",\s*'[A-Za-z0-9]+")
QUOTED_ID = re.compile(r"'([A-Za-z0-9]+)'")


def parse_parcel_list(raw):
    """Return (complete IDs, source status).

    Only accept a malformed list when every *preceding* ID is fully quoted
    and the malformed piece is confined to the very end of the cell.
    """
    raw = str(raw).strip()

    if COMPLETE_LIST.fullmatch(raw):
        return QUOTED_ID.findall(raw), "complete"

    if TRAILING_COMMA.fullmatch(raw) or TRAILING_PARTIAL_ID.fullmatch(raw):
        return QUOTED_ID.findall(raw), "partial_source_id_list"

    return [], "invalid"


def make_multi_parcel_lookup(multi_parcels):
    """Create a unique Parcel ID -> group centroid lookup for safe candidates.

    The resulting location is a *group* centroid, not the precise centroid of
    an individual parcel. No residential_units values are copied or divided.
    IDs that imply multiple different locations are rejected.
    """
    rows = []
    unsafe_ids = set()
    invalid_groups = 0
    partial_groups = 0

    for row in multi_parcels.itertuples(index=False):
        raw = str(row.mapblklot).strip()
        parcel_ids, source_status = parse_parcel_list(raw)

        if (
            not parcel_ids
            or pd.isna(row.centroid_latitude)
            or pd.isna(row.centroid_longitude)
        ):
            invalid_groups += 1
            unsafe_ids.update(QUOTED_ID.findall(raw))
            continue

        if source_status == "partial_source_id_list":
            partial_groups += 1

        for parcel_id in set(parcel_ids):
            rows.append(
                (parcel_id, row.centroid_latitude, row.centroid_longitude, source_status)
            )

    candidates = pd.DataFrame(
        rows,
        columns=["parcel_id", "latitude", "longitude", "source_status"],
    )

    # A Parcel ID pointing to two different group centroids is ambiguous.
    locations = candidates[["parcel_id", "latitude", "longitude"]].drop_duplicates()
    conflicting_ids = set(
        locations.loc[locations.duplicated("parcel_id", keep=False), "parcel_id"]
    )
    unsafe_ids.update(conflicting_ids)

    # If the same location appears in complete and partial lists, prefer the
    # complete-list attribution for the match_note.
    candidates = candidates.loc[~candidates["parcel_id"].isin(unsafe_ids)].copy()
    candidates["priority"] = candidates["source_status"].ne("complete").astype(int)
    lookup = (
        candidates.sort_values("priority")
        .drop_duplicates("parcel_id", keep="first")
        .set_index("parcel_id")[["latitude", "longitude", "source_status"]]
    )
    assert lookup.index.is_unique, "Multi-parcel lookup must have unique IDs."

    return lookup, unsafe_ids, conflicting_ids, invalid_groups, partial_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--housing", default="../raw_data/housing_production_filtered.csv")
    parser.add_argument("--land", default="../raw_data/sf_land_use_clean.csv")
    parser.add_argument("--output", default="housing_production_located.csv")
    args = parser.parse_args()

    # Keep IDs as strings (leading zeros and letters are meaningful).
    housing = pd.read_csv(args.housing, dtype={"BlockLot": "string"}, low_memory=False)
    land = pd.read_csv(args.land, dtype={"mapblklot": "string"}, low_memory=False)

    # Light checks on the other team's Housing dataset; Land Use was
    # cleaned and audited separately in Step 1.
    ids = housing["BlockLot"]
    print("=== Housing Production Checks ===")
    print("Total records:", len(housing))
    print("Missing / blank BlockLot:", (ids.isna() | ids.str.strip().eq("")).sum())
    print("BlockLot with surrounding whitespace:", ids.ne(ids.str.strip()).sum())
    print(
        "Non-alphanumeric BlockLot:",
        (~ids.dropna().str.fullmatch(r"[A-Za-z0-9]+") & ids.dropna().ne("")).sum(),
    )

    # Round 1: exact, many-to-one join against individual parcels.
    single = land.loc[
        land["geography_type"].eq("parcel"),
        ["mapblklot", "centroid_latitude", "centroid_longitude"],
    ].rename(
        columns={"centroid_latitude": "latitude", "centroid_longitude": "longitude"}
    )

    located = housing.merge(
        single,
        how="left",
        left_on="BlockLot",
        right_on="mapblklot",
        validate="many_to_one",  # Protect against duplicate Housing rows.
        indicator=True,
    )

    single_match = (
        located["_merge"].eq("both")
        & located["latitude"].notna()
        & located["longitude"].notna()
    )
    located["match_status"] = "unmatched"
    located["match_note"] = "no_parcel_match"
    located.loc[single_match, "match_status"] = "matched_single"
    located.loc[single_match, "match_note"] = "parcel_centroid"

    # Round 2: use the fully quoted IDs from multi-parcel records, even when
    # the list ends in a trailing comma or a truncated last ID.
    multi = land.loc[
        land["geography_type"].eq("multiple_parcels"),
        ["mapblklot", "centroid_latitude", "centroid_longitude"],
    ]
    lookup, unsafe_ids, conflicting_ids, invalid_groups, partial_groups = (
        make_multi_parcel_lookup(multi)
    )

    remaining = located["match_status"].eq("unmatched")
    multi_match = (
        remaining
        & located["BlockLot"].isin(lookup.index)
        & ~located["BlockLot"].isin(unsafe_ids)
    )

    located.loc[multi_match, "latitude"] = located.loc[multi_match, "BlockLot"].map(
        lookup["latitude"]
    )
    located.loc[multi_match, "longitude"] = located.loc[multi_match, "BlockLot"].map(
        lookup["longitude"]
    )
    located.loc[multi_match, "match_status"] = "matched_multi_centroid"

    partial_ids = set(lookup.index[lookup["source_status"] == "partial_source_id_list"])
    matched_from_partial = multi_match & located["BlockLot"].isin(partial_ids)
    located.loc[multi_match, "match_note"] = "shared_group_centroid_not_individual_parcel"
    located.loc[matched_from_partial, "match_note"] = (
        "shared_group_centroid;partial_source_id_list"
    )

    # Keep unlocatable Housing records; report ambiguous group IDs.
    still_unmatched = located["match_status"].eq("unmatched")
    located.loc[
        still_unmatched & located["BlockLot"].isin(unsafe_ids), "match_note"
    ] = "unsafe_multi_parcel_list_or_conflict"

    # Joining should never change the original permits or housing counts.
    assert len(located) == len(housing), "Join changed the number of Housing records."
    assert located["Net Units Completed"].equals(housing["Net Units Completed"]), (
        "Join changed Net Units Completed values or row order."
    )

    print("\n=== Step 2 Location Assignment ===")
    print("Matched by single parcel:", single_match.sum())
    print("Matched by multi-parcel group centroid:", multi_match.sum())
    print("  Of which from partial source ID lists:", matched_from_partial.sum())
    print("Unmatched:", still_unmatched.sum())
    print("Total location coverage:", f"{(~still_unmatched).mean():.2%}")
    print("Multi-parcel groups with incomplete final token:", partial_groups)
    print("Multi-parcel groups skipped as invalid:", invalid_groups)
    print("Conflicting multi-parcel IDs excluded:", len(conflicting_ids))

    located = located.drop(columns=["mapblklot", "_merge"])
    located.to_csv(args.output, index=False)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
