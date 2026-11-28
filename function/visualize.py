"""Figures for the 2010–2020 housing analysis."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter


def plot_housing_map(tracts_2010, tracts_2020, column, label, filename, title="San Francisco housing density: 2010 vs. 2020"):
    # Use each year's original geometry and the same color breaks for both maps.
    values = pd.concat([tracts_2010[column], tracts_2020[column]]).dropna()
    positive = values[values > 0]
    breaks = [0] + positive.quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()
    breaks = sorted(set(breaks))
    cmap = plt.get_cmap("OrRd", len(breaks) - 1)
    norm = BoundaryNorm(breaks, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, tracts, year in [(axes[0], tracts_2010, 2010), (axes[1], tracts_2020, 2020)]:
        tracts.plot(ax=ax, column=column, cmap=cmap, norm=norm, edgecolor="gray", linewidth=0.3,
                    missing_kwds={"color": "lightgray"})
        ax.set_xlim(-122.52, -122.35)
        ax.set_ylim(37.70, 37.82)
        ax.set_title(str(year))
        ax.set_axis_off()

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                            fraction=0.06, pad=0.06, ticks=breaks, format=StrMethodFormatter("{x:,.0f}"))
    colorbar.set_label(label)
    fig.suptitle(title, fontsize=16)
    fig.text(0.5, 0.02, "Each year's shoreline-clipped tract map; common color breaks; mainland view. Gray = no valid denominator.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.90, bottom=0.17, wspace=0.08)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_group_comparison(income, renter, column, title, ylabel, filename, decimals=1):
    # Compare fixed 2010 income and renter-share groups using the same y scale.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, table, field, order, name in [
        (axes[0], income, "income_group", ["low", "middle", "high"], "By income"),
        (axes[1], renter, "renter_group", ["low_renter", "middle_renter", "high_renter"], "By renter share")]:
        table = table.set_index(field).reindex(order)
        bars = ax.bar(["Low", "Middle", "High"], table[column], color="#4982a7")
        for bar, value in zip(bars, table[column]):
            if pd.notna(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.{decimals}f}", ha="center", va="bottom" if value >= 0 else "top")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.025, "2010–2020 comparable areas; groups defined by 2006–2010 ACS. Changes and rates use original group housing totals. Unclassified areas are in the tables only.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_snapshot_comparison(income, renter, column, ylabel, filename):
    # Compare native-year snapshots; the groups do not contain the same areas.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, table, field, order, title in [
        (axes[0], income, "income_group", ["low", "middle", "high"], "By income"),
        (axes[1], renter, "renter_group", ["low_renter", "middle_renter", "high_renter"], "By renter share")]:
        table = table.pivot(index=field, columns="year", values=column).reindex(order)
        table.index = ["Low", "Middle", "High"]
        table.plot(kind="bar", ax=ax, color=["#6a9bc2", "#c96a40"], rot=0)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original housing stock by year-specific ACS groups", fontsize=15)
    fig.text(0.5, 0.025, "Each year uses its own boundaries and ACS groups. These are snapshots, not growth rates. Unclassified areas are in the tables only.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_growth_map(tracts_2010, master, filename, column="net_units_per_km2",
                    title="Net housing change per land km², 2010–2020",
                    label="Net units / 2010 land km²"):
    # Use original 2010 polygons only to display the comparable-area change.
    change = master[["GEOID_2010", column]].rename(columns={"GEOID_2010": "GEOID"})
    mapped = tracts_2010.merge(change, on="GEOID", how="left")
    values = mapped[column]
    breaks = sorted(set([0] + values[values > 0].quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()))
    colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
    mapped["color"] = "#d7d7d7"
    mapped.loc[values == 0, "color"] = "#ffffff"
    mapped.loc[values < 0, "color"] = "#3182bd"
    legend = [Patch(facecolor="#d7d7d7", label="No comparable data / undefined ratio"),
              Patch(facecolor="#ffffff", edgecolor="gray", label="Zero"),
              Patch(facecolor="#3182bd", label="Net loss")]
    for i in range(len(breaks) - 1):
        selected = (values > breaks[i]) & (values <= breaks[i + 1])
        mapped.loc[selected, "color"] = colors[i]
        legend.append(Patch(facecolor=colors[i], label=f">{breaks[i]:,.0f}–{breaks[i + 1]:,.0f}"))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mapped.plot(ax=ax, color=mapped["color"], edgecolor="gray", linewidth=0.3)
    ax.set_xlim(-122.52, -122.35)
    ax.set_ylim(37.70, 37.82)
    ax.set_axis_off()
    ax.set_title(title, fontsize=15)
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, title=label, frameon=False)
    fig.text(0.5, 0.015, "Comparable 2010 areas; full 2020 successor counts; positive values in five quantile groups; blue = net loss.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_growth_distribution(master, column, xlabel, filename):
    # Keep the full distribution and preserve negative growth.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(master[column].dropna(), bins=25, color="#4982a7", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of comparable 2010 areas")
    ax.set_title("Distribution of housing growth, 2010–2020")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def plot_top_tracts(top, bottom, column, xlabel, filename):
    # Show both groups on one scale. Keep the signed values and common zero line.
    table = pd.concat([top, bottom]).drop_duplicates(subset="GEOID_2010").sort_values(column)
    colors = []
    for value in table[column]:
        if value >= 0:
            colors.append("#4982a7")
        else:
            colors.append("#c96a40")
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(table["tract_name"], table[column], color=colors)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.margins(x=0.14)
    ax.set_xlabel(xlabel)
    ax.set_title("Highest and lowest housing growth rates, 2010–2020", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02, "Comparable areas with positive initial housing stock. Small initial stocks can produce extreme rates.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()
