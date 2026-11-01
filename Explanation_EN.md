# English explanation: SF Housing research, Steps 1–8

## Choose the working directory

Run `notebooks/SF_Housing_Combined_Local.ipynb` from the `notebooks` directory; its paths begin with `../data/`. Run `SF_Housing_Combined_Root.ipynb` from the project root, beside `data`; its paths begin with `data/`. The calculations are identical. Choose one version rather than running both in sequence.

All imports appear at the top. Later steps use variables created by earlier steps. Neither notebook reads a file from `data/processed`: all outputs can be rebuilt with the processed and visualization directories initially empty. Raw inputs are in `data/raw`, CSV outputs go directly to `data/processed`, and PNG files go directly to `visualizations`.

## Step 1: Clean and filter the raw datasets

### Housing Production

The raw Housing Production CSV supplies the permit ID, address, parcel ID, first completion date and signed net completed-unit change. Read identifiers as text to preserve leading zeros and letters.

Convert the first completion date to a date and keep `2005-01-01 <= date < 2026-01-01`. Exclude only missing or zero net changes. Keep negative changes: a record with +10 adds ten net units, while a record with −2 removes two net units. Both belong in net additions.

Rename the six fields expected by the teammate workflow, such as `bpa` to `BPA` and `blocklot` to `BlockLot`. The filtered table is named **`housing_production_filtered`**. Its data source remains Housing Production.

Save `housing_production_filtered.csv`. The study sample contains 5,981 records, 60,900 positive net changes, 2,918 units of losses, and 57,982 signed net additions.

### Land Use: the relevant part of the teammate's cleaner

Read the raw Land Use CSV, preserve `mapblklot` as text, and flag missing parcel IDs and residential-unit counts. Missing IDs cannot support a location join. Follow the original zero-fill rule for missing unit values while retaining the missing-value flag. Remove thousands separators with classroom string operations and convert the unit values to integers.

Retain the ten parcel-conflict decisions supplied in `clean_sf_land_use.py`, then deduplicate parcel IDs. These are the uploaded script's decisions; this rewrite does not claim to have independently reverified the administrative evidence in its comments. Keep the fields used by this project and save `sf_land_use_clean.csv`.

Land Use residential units are a current snapshot, not dated 2008 or 2005 stock. Because location matching uses the published centroid fields, the analysis does not repair unused Land Use polygons or export an additional GeoPackage.

## Step 2: Give every housing record a location

### Individual-parcel matching from `tabular_join.py`

Select Land Use rows marked `parcel`. Left-merge Housing Production `BlockLot` with Land Use `mapblklot` and attach the published centroid coordinates. The left merge keeps every housing record; unmatched records temporarily retain missing coordinates.

Attach locations only. Do not add Land Use residential stock to permit-level net changes.

### Multi-parcel matching with the teammate's rules

The source has 77 multi-parcel entries. Each may list several parcel IDs but supplies one shared centroid. Try these member IDs only for records not already located through an individual parcel.

Replace the original regex implementation with classroom indexing, lists, loops and conditions while preserving its acceptance rules. Use fully quoted IDs only. Require at least two complete IDs. If malformed content is confined to the end, use the preceding complete IDs but never the incomplete fragment. Reject IDs from invalid lists or conflicting coordinates. Prefer complete-list attribution when the same coordinate is repeated.

Save `housing_production_located.csv`, keeping labels for individual centroids, shared group centroids and unmatched records. A shared centroid is not an exact member-parcel location.

### Generate address keys inside the notebook

The notebook no longer reads a prepared address-key file. It derives comparison fields directly from this run's `located` table:

1. Uppercase addresses, trim their ends and collapse repeated whitespace.
2. Replace whole-word AVENUE/AVE with AV and STREET with ST, retaining punctuation, house numbers and unit information.
3. Extract the assessor block from the first four parcel-ID digits and an optional following uppercase letter.
4. Combine the normalized address and block into `address_block_key`.

A single `normalize_address()` function applies identical rules to housing, DBI address candidates and Census responses. It uses string operations, loops and conditions from Labs 3–4 and the function-definition pattern shown in Lab 6. It uses no regex or `.map()`.

Save `address_keys.csv`. This file is now a reproducible notebook output, not a required pre-existing input.

### Expand the raw DBI JSON

Read the downloaded JSON with the Lab 6 `urlopen → read → json.loads` pattern. The `file:` prefix means a local file, so no internet request is made. Iterate through all records and place permit ID, block, lot, primary-address flag, coordinates and the original point-source label into a DataFrame. Save `dbi_coordinates.csv`.

First flatten all 7,421 candidate records without choosing a location. Then select complete primary-address coordinates. Deduplicate identical permit/coordinate pairs, count distinct locations per permit and accept only one unique location. Left-merge candidates back by BPA and fill missing housing coordinates only.

### Read the original recorded-parcel CSV

The recorded-parcel source is already a flat CSV, so read `parcel_dated.csv` directly from raw. Match exact historical parcel IDs and accept one unique published centroid. Preserve its active flag and reference-centroid label. Do not guess a successor parcel or arbitrarily choose among different locations.

### Build address candidates from raw JSON

Read every saved raw address record. Combine street number, name and suffix, normalize the street address with the same function, and combine it with the source assessor block. Save `address_candidates.csv`.

Fill a still-missing housing location only when the normalized address and block agree exactly and candidates identify one unique coordinate. This is not fuzzy address matching.

### Expand the saved Census responses

For every request and returned candidate, retain the requested address, full matched address, street portion before the first comma, city, number of matches and coordinates. Normalize both street addresses identically and save `census_candidates.csv`.

Accept only a unique San Francisco match with an identical normalized street address. Preserve the approximate address-range label; these are not surveyed building coordinates.

The recovery priority remains Land Use, DBI primary address, recorded parcel, exact address within the same block, and Census address range. Existing coordinates are never replaced. Save `housing_production_located_recovered.csv` and `location_source_summary.csv`.

## Step 3: Assign housing points to 2020 census tracts

Preserve the method in the teammate's `tract_aggregation.py`: create EPSG:4326 housing points, use `gpd.sjoin(..., how="left", predicate="within")` against the tract polygons, and merge each GEOID back into the complete housing table by permit ID.

Read the official GeoJSON downloaded directly from the same DataSF 2020 tract dataset. It replaces the previously assistant-converted boundary file. Rename only its native `geoid/name` fields to the workflow's `GEOID/NAME`.

Keep any permit without a valid tract as an unassigned row rather than inventing a neighborhood. Save `housing_production_with_tract.csv`. The join summary reports both record counts and the net units represented by records at the Land Use-only and recovered stages.

All 5,981 study records receive a tract in this dataset. The 2,006 records originally unlocated through Land Use represent 39,174 net units, which the supplementary sources recover for tract analysis.

## Step 4: Establish a complete tract table and correct the baseline

The former proposal treated the current Land Use download as 2008 stock and subtracted 2005–2008 changes to reconstruct 2005 stock. That is unsupported: a database's original baseline year does not make its current public snapshot a historical inventory.

Instead, start with all 244 census tracts. Aggregate assigned positive changes, the absolute value of negative changes, signed net additions and production-record counts by GEOID. Left-merge these summaries into the complete tract layer.

Fill zero counts for tracts without assigned records. Keep geographically unassigned housing in the detailed permit table, not a fictitious tract. Do not create an unverified `baseline_2005_stock` denominator.

## Step 5: Calculate and map net housing additions

For each tract, net additions equal positive net changes minus the absolute value of losses. Save `tract_housing_production.csv` and draw `historical_net_additions_map.png`.

The citywide denominator is the full selected source sample, including any records lacking tract or ACS classification. This run contains 57,982 net units, all assigned to tracts.

This is a completed net-housing count, not a permit count or a growth rate normalized by historical stock. The inclusive 2005–2025 window does not justify a percentage-growth formula without a reliable stock denominator.

## Step 6: Classify tracts using 2020–2024 ACS conditions

Read the raw 2024 ACS five-year B19013 and B25003 files, select San Francisco tract rows and construct text GEOIDs. The release describes **2020–2024**, not a single year or 2005 neighborhood conditions.

Convert negative income special codes to missing. The income code 250001 means $250,000 or more; retain its top-code flag and use 250000 as the grouping lower bound. Renter share is renter-occupied units divided by all occupied units, excluding vacant units. Invalid counts and zero denominators do not produce a share.

Merge ACS into the housing tract table by GEOID; no spatial join is needed. Keep all 244 tracts. Use the same 234 tracts with valid income and renter share for both classifications. Calculate each measure's one-third and two-thirds quantiles and assign lower/middle/higher groups using `.loc[]`. Each dimension has 78 tracts per group.

The other ten known tracts are `unclassified` because their common ACS measures are incomplete. This differs from housing without a known tract. Save `tract_acs_groups.csv`, `acs_group_cutoffs.csv` and `acs_group_maps.png`.

These are relative local groupings, not official income-eligibility categories or neighborhood identities assumed constant throughout 2005–2025.

## Step 7: Compare historical housing production across groups

For income and renter-share groups separately, calculate total net additions, their share of the full-source citywide total, median tract net additions and mean tract net additions. Totals describe production concentration; medians describe a typical tract.

Keep ACS-unclassified and geographically unassigned housing separate. The latter is zero here. The ten ACS-excluded tracts contribute 1,571 net units, which remain in citywide accounting.

Save `group_housing_comparisons.csv` and the total/median small-multiple charts. Do not compare against an unsupported 2005 stock share or describe these counts as growth rates or causal effects.

## Step 8: Compare the adopted FZP indicators geographically

### Inputs, locations, density and use amendments

Read the archived adopted density, use-district amendment and height GIS files. They were previously verified in the publisher's Adopted December 2025 map group and archived on October 28, 2026. Preserve the source's November/Proposed service and field names.

Use the density layer's 154,464 distinct parcel IDs as the mapped universe. Project geometry to EPSG:2227, calculate centroids, reproject to the tract CRS and apply a within spatial join. A cross-tract parcel is assigned by its centroid. Keep unmatched parcels. Merge superseded/approved use-district attributes by parcel ID.

Use `Form-based Proposed` as the new form-based base-density indicator; do not treat `Form-based Existing` as new change. Presence in the dedicated use-amendment layer defines that indicator. The unit is a parcel ID, not a dwelling, unique building or area; different IDs may share a footprint.

### Strict height comparison

Use one height feature per parcel ID, Not Split, matching old-height labels, a readable simple old-height code and nonmissing base/local numeric heights. Parse simple codes with classroom loops while retaining the previous selection rule.

BaseHeight is the adopted base scenario, not the pre-plan control. NEW_HEIGHT_NUM is the Local Program scenario and is not an unconditional entitlement for every project. Subtract the readable old control from each adopted scenario and retain negative differences.

### Aggregation and group comparisons

Aggregate indicators by tract and merge them into the one-row-per-tract housing/ACS master. Density and use-amendment coverage divide designated parcel IDs by mapped parcel IDs; height-increase coverage uses only strictly comparable IDs. Leave zero-denominator percentages missing rather than interpreting unavailable data as 0% change.

Compare income groups, renter-share groups and historical-production groups. Define the latter by net-additions quantiles within the same 234 common-valid ACS tracts, keeping ties together. Pool each group's numerators and eligible denominators before dividing. Calculate housing shares independently from tract totals once, avoiding repetition of tract housing across parcel rows.

Save `policy_parcel_attributes.csv`, `tract_policy_housing.csv`, `group_policy_comparisons.csv` and the four policy figures.

The results describe where these adopted-map indicators are concentrated, those locations' historical production, and their recent community characteristics. They are not additional-unit capacity estimates. Projects completed after 2026 may have earlier approvals, so a completion-date filter cannot establish FZP effects or calculate extra legal capacity.

## Outputs and reproducibility

All 18 CSV files in `data/processed`, including the address-key and three expanded-JSON tables, are generated in the notebook. The eight PNG figures are saved directly to `visualizations`.

Both path versions were run independently without pre-existing processed results and produced matching outputs. The three original teammate files are unchanged. The notebooks contain no external script calls, test assertions or hidden prepared-table inputs.

Official raw tract GeoJSON: [DataSF Census 2020: Tracts for San Francisco](https://data.sf.gov/d/tmph-tgz9). The other archived raw sources remain Housing Production, Land Use, DBI, Recorded Parcels, Census Geocoder, ACS and the adopted GIS layers.
