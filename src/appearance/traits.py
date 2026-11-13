"""Appearance attribute definitions, grounded in forensic DNA phenotyping.

Each trait carries its categories, a display palette (for the ones rendered on
the face), how it is shown (rendered vs reported), and calibration strengths
that set how predictable it is in the synthetic data. The strengths are tuned so
the synthetic prototype's per-trait accuracy mirrors the real literature
(HIrisPlex-S and facial-feature GWAS): pigmentation extremes are easy, the
intermediate eye class and the blonde/brown hair split are genuinely hard, and
hair type and balding are weak. This calibration is the anti-gimmick safeguard.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trait:
    key: str
    name: str
    classes: list[str]
    render: str                 # "skin" | "iris" | "hair" | "freckles" | "report"
    snp_strength: float         # weight of the genetic (panel) component
    anc_strength: float         # weight of the ancestry component (confounder)
    age_strength: float = 0.0   # weight of age (for balding)
    n_causal: int = 12          # causal appearance SNPs for this trait
    palette: dict[str, str] = field(default_factory=dict)  # class -> hex colour


# Display palettes for rendered traits (hex). Reported traits use swatches too.
TRAITS: dict[str, Trait] = {
    "eye_colour": Trait(
        key="eye_colour", name="Eye colour",
        classes=["blue", "intermediate", "brown"],
        render="iris", snp_strength=0.80, anc_strength=0.55, n_causal=10,
        palette={"blue": "#5a8bb0", "intermediate": "#6f7a4e", "brown": "#5a3b22"},
    ),
    "skin_tone": Trait(
        key="skin_tone", name="Skin tone",
        classes=["very_pale", "pale", "medium", "olive", "brown", "dark"],
        render="skin", snp_strength=0.70, anc_strength=0.85, n_causal=14,
        palette={"very_pale": "#f3d9c3", "pale": "#e7bfa3", "medium": "#d29c7b",
                 "olive": "#b67c54", "brown": "#8a5a38", "dark": "#5c3a24"},
    ),
    "hair_colour": Trait(
        key="hair_colour", name="Hair colour",
        classes=["blonde", "brown", "black", "red"],
        render="hair", snp_strength=0.60, anc_strength=0.55, n_causal=12,
        palette={"blonde": "#caa472", "brown": "#6b4630", "black": "#201912",
                 "red": "#9b4722"},
    ),
    "freckling": Trait(
        key="freckling", name="Freckling",
        classes=["none", "some", "many"],
        render="freckles", snp_strength=0.50, anc_strength=0.25, n_causal=8,
        palette={"none": "#00000000", "some": "#9c6b4a", "many": "#8a5a3a"},
    ),
    "hair_type": Trait(
        key="hair_type", name="Hair type",
        classes=["straight", "wavy", "curly"],
        render="report", snp_strength=0.50, anc_strength=0.55, n_causal=8,
    ),
    "baldness": Trait(
        key="baldness", name="Balding propensity",
        classes=["none", "moderate", "severe"],
        render="report", snp_strength=0.45, anc_strength=0.10, age_strength=0.65,
        n_causal=10,
    ),
}


def trait_list() -> list[Trait]:
    return list(TRAITS.values())


def trait_keys() -> list[str]:
    return list(TRAITS.keys())
