"""Approximate starting Elo ratings for national teams (cold-start priors).

These are ballpark World-Football-Elo-style values for the strongest sides as of
the 2024-25 international window. They are only *priors*: once training runs, the
Elo update in features.py overwrites them from actual results, so exact accuracy
matters less than getting teams roughly ordered instead of all tied at 1500.

Teams not listed fall back to config.ELO_START (1500), which is about right for a
mid/low-tier national side on this scale.
"""
from __future__ import annotations

import config

# Approximate ratings; higher = stronger. Ordered by strength for readability.
INITIAL_ELO: dict[str, float] = {
    "Argentina": 2100,
    "France": 2060,
    "Spain": 2050,
    "Brazil": 2030,
    "England": 1990,
    "Portugal": 1985,
    "Netherlands": 1975,
    "Belgium": 1945,
    "Italy": 1940,
    "Germany": 1930,
    "Croatia": 1900,
    "Colombia": 1890,
    "Uruguay": 1885,
    "Morocco": 1875,
    "Switzerland": 1860,
    "Denmark": 1855,
    "Japan": 1850,
    "USA": 1830,
    "United States": 1830,
    "Mexico": 1825,
    "Senegal": 1820,
    "Ecuador": 1815,
    "Austria": 1810,
    "Ukraine": 1800,
    "Sweden": 1795,
    "Peru": 1790,
    "South Korea": 1790,
    "Korea Republic": 1790,
    "Serbia": 1785,
    "Iran": 1785,
    "Poland": 1780,
    "Nigeria": 1775,
    "Chile": 1770,
    "Wales": 1765,
    "Egypt": 1760,
    "Algeria": 1760,
    "Australia": 1755,
    "Turkey": 1755,
    "Ivory Coast": 1750,
    "Cameroon": 1745,
    "Tunisia": 1740,
    "Czech Republic": 1740,
    "Scotland": 1735,
    "Hungary": 1730,
    "Norway": 1730,
    "Ghana": 1725,
    "Canada": 1720,
    "Greece": 1715,
    "Paraguay": 1710,
    "Romania": 1705,
    "Costa Rica": 1690,
    "Saudi Arabia": 1685,
    "Qatar": 1680,
    "Mali": 1680,
    "Venezuela": 1675,
    "Panama": 1660,
}


def seed_for(name: str | None) -> float:
    """Prior Elo for a team name, or the flat default if unknown."""
    if not name:
        return config.ELO_START
    return INITIAL_ELO.get(name, config.ELO_START)
