"""Experience retrieval specialist agent.

Searches the ExperienceCache for similar past design episodes,
computes similarity, and warm-starts the current design from prior
experience when a close match is found.

Usage:
    from embodied_ai_architect.graphs.experience_specialist import experience_retriever
"""

from __future__ import annotations

import logging
from typing import Any

from embodied_ai_architect.graphs.experience import DesignEpisode, ExperienceCache
from embodied_ai_architect.graphs.soc_state import SoCDesignState, get_constraints
from embodied_ai_architect.graphs.task_graph import TaskNode

logger = logging.getLogger(__name__)

# Minimum similarity threshold for warm-starting from prior experience
SIMILARITY_THRESHOLD = 0.5


def compute_similarity(
    current_use_case: str,
    current_platform: str,
    current_constraints: dict[str, Any],
    past_episode: DesignEpisode,
) -> float:
    """Compute similarity between current problem and a past episode.

    Similarity is based on:
    - Use case match (40% weight)
    - Platform match (20% weight)
    - Constraint overlap (40% weight)

    Args:
        current_use_case: Current design use case.
        current_platform: Current platform.
        current_constraints: Current constraints dict.
        past_episode: Past design episode to compare against.

    Returns:
        Similarity score from 0.0 to 1.0.
    """
    score = 0.0

    # Use case match (0.4)
    if current_use_case and past_episode.use_case:
        if current_use_case == past_episode.use_case:
            score += 0.4
        elif _use_case_category(current_use_case) == _use_case_category(past_episode.use_case):
            score += 0.2

    # Platform match (0.2)
    if current_platform and past_episode.platform:
        if current_platform == past_episode.platform:
            score += 0.2
        elif _platform_family(current_platform) == _platform_family(past_episode.platform):
            score += 0.1

    # Constraint overlap (0.4)
    if current_constraints and past_episode.constraints:
        current_keys = {k for k, v in current_constraints.items() if v is not None}
        past_keys = {k for k, v in past_episode.constraints.items() if v is not None}
        if current_keys or past_keys:
            overlap = len(current_keys & past_keys)
            union = len(current_keys | past_keys)
            jaccard = overlap / union if union > 0 else 0.0
            score += 0.4 * jaccard

            # Bonus for similar constraint values
            common = current_keys & past_keys
            if common:
                close_count = 0
                for k in common:
                    cv = current_constraints.get(k)
                    pv = past_episode.constraints.get(k)
                    if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                        if pv != 0:
                            ratio = abs(cv - pv) / abs(pv)
                            if ratio < 0.25:  # within 25%
                                close_count += 1
                if close_count > 0:
                    score = min(1.0, score + 0.1 * (close_count / len(common)))

    return round(min(1.0, score), 3)


def _use_case_category(use_case: str) -> str:
    """Map use case to a broader category."""
    aerial = {"delivery_drone", "agricultural_drone", "inspection_drone", "drone"}
    ground = {"warehouse_amr", "amr", "quadruped", "quadruped_robot", "mobile_robot"}
    medical = {"surgical_robot", "medical_device", "patient_monitor"}
    if use_case in aerial:
        return "aerial"
    if use_case in ground:
        return "ground"
    if use_case in medical:
        return "medical"
    return use_case


def _platform_family(platform: str) -> str:
    """Map platform to a broader family."""
    aerial = {"drone", "vtol", "fixed_wing"}
    ground = {"amr", "quadruped", "biped", "wheeled"}
    if platform in aerial:
        return "aerial"
    if platform in ground:
        return "ground"
    return platform


def experience_retriever(task: TaskNode, state: SoCDesignState) -> dict[str, Any]:
    """Specialist agent: search past experience and warm-start current design.

    Searches the ExperienceCache for episodes similar to the current design
    problem. If a close match is found (similarity > threshold), adapts the
    prior hardware candidates as warm-start suggestions.

    Writes to state: prior_experience, hardware_candidates (warm-start)
    """
    use_case = state.get("use_case", "")
    platform = state.get("platform", "")
    constraints = state.get("constraints", {})

    # Use in-memory cache if provided in state, otherwise default
    cache_path = state.get("_experience_cache_path", None)
    try:
        cache = ExperienceCache(db_path=cache_path)
    except Exception as e:
        logger.warning("Could not open experience cache: %s", e)
        return {
            "summary": "Experience cache unavailable",
            "prior_experience": {"found": False, "reason": str(e)},
            "_state_updates": {
                "prior_experience": {"found": False, "reason": str(e)},
            },
        }

    # Search for similar episodes
    try:
        similar = cache.search_similar(use_case=use_case, platform=platform, limit=5)
    except Exception as e:
        logger.warning("Experience search failed: %s", e)
        similar = []

    if not similar:
        return {
            "summary": "No prior experience found",
            "prior_experience": {"found": False, "matches": []},
            "_state_updates": {
                "prior_experience": {"found": False, "matches": []},
            },
        }

    # Score each match
    scored_matches = []
    for episode in similar:
        sim = compute_similarity(use_case, platform, constraints, episode)
        scored_matches.append(
            {
                "episode_id": episode.episode_id,
                "use_case": episode.use_case,
                "platform": episode.platform,
                "similarity": sim,
                "outcome_score": episode.outcome_score,
                "hardware_selected": episode.hardware_selected,
                "architecture_chosen": episode.architecture_chosen,
                "iterations_used": episode.iterations_used,
                "lessons_learned": episode.lessons_learned,
            }
        )

    scored_matches.sort(key=lambda m: m["similarity"], reverse=True)

    state_updates: dict[str, Any] = {}
    best_match = scored_matches[0] if scored_matches else None

    # Warm-start hardware candidates if best match is above threshold
    warm_started = False
    if best_match and best_match["similarity"] >= SIMILARITY_THRESHOLD:
        # Load full episode for warm-start data
        full_episode = cache.load(best_match["episode_id"])
        if full_episode and full_episode.hardware_selected:
            warm_started = True
            # Boost the previously-selected hardware in candidates
            existing_candidates = state.get("hardware_candidates", [])
            if existing_candidates:
                for c in existing_candidates:
                    if c.get("name") == full_episode.hardware_selected:
                        c["score"] = min(100, c.get("score", 50) + 15)
                        c["warm_start_from"] = full_episode.episode_id
                state_updates["hardware_candidates"] = existing_candidates

    experience_result = {
        "found": True,
        "matches": scored_matches,
        "best_similarity": best_match["similarity"] if best_match else 0.0,
        "warm_started": warm_started,
    }
    state_updates["prior_experience"] = experience_result

    match_count = len(scored_matches)
    above_threshold = sum(1 for m in scored_matches if m["similarity"] >= SIMILARITY_THRESHOLD)
    summary = f"Found {match_count} similar episodes ({above_threshold} above threshold)"
    if warm_started:
        summary += f", warm-started from {best_match['episode_id']}"

    return {
        "summary": summary,
        "prior_experience": experience_result,
        "_state_updates": state_updates,
    }
