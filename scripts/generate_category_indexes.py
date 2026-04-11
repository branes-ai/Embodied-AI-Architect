#!/usr/bin/env python3
"""Generate _category.yaml index files for each populated platform category directory.

Scans all platform category directories under data/platforms/, computes shared/common
attributes across all platform YAML files in each directory, and writes a _category.yaml
summary file.
"""

import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml


PLATFORMS_DIR = Path(__file__).resolve().parent.parent / "data" / "platforms"


def load_platforms(category_dir: Path) -> list[dict]:
    """Load all platform YAML files in a category directory (excluding _category.yaml)."""
    platforms = []
    for f in sorted(category_dir.glob("*.yaml")):
        if f.name == "_category.yaml":
            continue
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            if data and isinstance(data, dict):
                platforms.append(data)
        except Exception as e:
            print(f"  WARNING: failed to parse {f.name}: {e}")
    return platforms


def safe_val(v):
    """Extract a numeric value from either a scalar or a dict with min/max/typical."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return v.get("typical") or v.get("max") or v.get("min")
    return None


def compute_range(platforms: list[dict], attr: str) -> dict | None:
    """Compute min/max/typical across all platforms for a given attribute."""
    mins, maxes, typicals = [], [], []
    for p in platforms:
        attrs = p.get("attributes", {})
        val = attrs.get(attr)
        if val is None:
            continue
        if isinstance(val, dict):
            if val.get("min") is not None:
                mins.append(val["min"])
            if val.get("max") is not None:
                maxes.append(val["max"])
            if val.get("typical") is not None:
                typicals.append(val["typical"])
        elif isinstance(val, (int, float)):
            mins.append(val)
            maxes.append(val)
            typicals.append(val)

    if not mins and not maxes and not typicals:
        return None

    result = {}
    if mins:
        result["min"] = round(statistics.mean(mins), 1)
    if maxes:
        result["max"] = round(statistics.mean(maxes), 1)
    if typicals:
        result["typical"] = round(statistics.mean(typicals), 1)
    return result if result else None


def most_common(values: list[str]) -> str | None:
    """Return the most common non-None value."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def collect_environments(platforms: list[dict]) -> list[str]:
    """Collect unique environment values across platforms."""
    envs = set()
    for p in platforms:
        cls = p.get("classification", {})
        env = cls.get("environment", [])
        if isinstance(env, list):
            envs.update(env)
        elif isinstance(env, str):
            envs.add(env)
    return sorted(envs)


def collect_detection_classes(platforms: list[dict]) -> list[str]:
    """Collect union of detection_classes from implications.perception."""
    classes = set()
    for p in platforms:
        impl = p.get("implications", {})
        perc = impl.get("perception", {})
        det = perc.get("detection_classes", [])
        if isinstance(det, list):
            classes.update(d for d in det if d)
    return sorted(classes)


def generate_category_yaml(category_dir: Path) -> bool:
    """Generate _category.yaml for a single category directory. Returns True if written."""
    platforms = load_platforms(category_dir)
    if not platforms:
        return False

    category_name = category_dir.name
    n = len(platforms)

    # Pick best description: longest one or first available
    descriptions = [p.get("description", "") for p in platforms if p.get("description")]
    description = max(descriptions, key=len) if descriptions else f"{category_name} platforms"
    # Shorten to a category-level description
    category_desc = f"Shared defaults for {n} {category_name.replace('_', ' ')} platform(s)"

    # default_attributes
    default_attributes = {}
    for attr in ("power_watts", "weight_kg", "cost_usd", "latency_ms"):
        r = compute_range(platforms, attr)
        if r:
            default_attributes[attr] = r

    # common_classification
    locomotions = [
        p.get("classification", {}).get("locomotion")
        for p in platforms
        if p.get("classification", {}).get("locomotion")
    ]
    common_classification = {}
    mc = most_common(locomotions)
    if mc:
        common_classification["locomotion"] = mc
    envs = collect_environments(platforms)
    if envs:
        common_classification["environment"] = envs

    # common_perception_tasks
    detection_classes = collect_detection_classes(platforms)

    # typical_constraints
    typical_constraints = {}
    power_range = compute_range(platforms, "power_watts")
    if power_range and "typical" in power_range:
        typical_constraints["max_power_watts"] = power_range["typical"]
    latency_range = compute_range(platforms, "latency_ms")
    if latency_range and "typical" in latency_range:
        typical_constraints["max_latency_ms"] = latency_range["typical"]
    cost_range = compute_range(platforms, "cost_usd")
    if cost_range and "typical" in cost_range:
        typical_constraints["max_cost_usd"] = cost_range["typical"]

    # Build the document
    doc = {
        "category": category_name,
        "description": category_desc,
        "platform_count": n,
    }
    if default_attributes:
        doc["default_attributes"] = default_attributes
    if common_classification:
        doc["common_classification"] = common_classification
    if detection_classes:
        doc["common_perception_tasks"] = detection_classes
    if typical_constraints:
        doc["typical_constraints"] = typical_constraints

    # Write with a header comment
    out_path = category_dir / "_category.yaml"
    header = (
        f"# Category: {category_name}\n"
        f"# Auto-generated index — shared defaults for all {category_name} platforms\n\n"
    )
    with open(out_path, "w") as fh:
        fh.write(header)
        yaml.dump(
            doc,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    return True


def main():
    if not PLATFORMS_DIR.is_dir():
        print(f"ERROR: platforms directory not found: {PLATFORMS_DIR}")
        sys.exit(1)

    generated = 0
    skipped = 0

    for category_dir in sorted(PLATFORMS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        name = category_dir.name
        # Skip configurations directory (no platform files)
        if name == "configurations":
            print(f"  SKIP: {name} (excluded)")
            skipped += 1
            continue

        print(f"  Processing: {name} ... ", end="")
        if generate_category_yaml(category_dir):
            generated += 1
            print("OK")
        else:
            skipped += 1
            print("skipped (no platform files)")

    print(f"\nDone: {generated} _category.yaml files generated, {skipped} directories skipped.")


if __name__ == "__main__":
    main()
