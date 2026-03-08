"""Spatial fusion and result reconciliation logic for UI grounding.

Manage Non-Maximum Suppression (NMS), geometric validation of visual hits,
and the strategic merging of visual and OCR candidates.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_solution.constants import (
    FINAL_DEDUP_RADIUS_FACTOR,
    FUSION_DISTANCE_FACTOR,
    GEOM_BASE_SCORE_WEIGHT,
    GEOM_RATIO_BONUS_WEIGHT,
    MAX_TEMPLATE_HITS,
    MIN_DEDUPE_RADIUS_FACTOR,
    NMS_RADIUS_FACTOR,
)
from cv_solution.models import Candidate, DetectionMethod

if TYPE_CHECKING:
    from pathlib import Path

    from cv_solution.models import GroundingConfig


class FusionProcessor:
    """Handle spatial reconciliation, NMS, and multi-modal result fusion.

    This processor refines raw detections by suppressing overlapping matches
    and merging visual evidence with OCR data to increase confidence scores.
    """

    def __init__(self, config: GroundingConfig) -> None:
        """Initialize the fusion processor with grounding configurations.

        Args:
            config: Configuration object containing thresholds and weights.

        """
        self.config = config

    def apply_nms(
        self,
        candidates: list[Candidate],
        template_size: int,
    ) -> list[Candidate]:
        """Apply non-maximum suppression to remove overlapping template hits.

        Args:
            candidates: List of raw candidate detections.
            template_size: Base size of the icon for radius calculation.

        Returns:
            A filtered list of candidates where redundant overlaps are removed.

        """
        if not candidates:
            return []

        # Sort by score descending and cap results to MAX_TEMPLATE_HITS
        sorted_hits = sorted(candidates, key=lambda x: x.score, reverse=True)[
            :MAX_TEMPLATE_HITS
        ]
        kept_candidates: list[Candidate] = []
        distance_threshold_sq = (template_size * NMS_RADIUS_FACTOR) ** 2

        for c in sorted_hits:
            if not any(
                ((c.x - a.x) ** 2 + (c.y - a.y) ** 2) < distance_threshold_sq
                for a in kept_candidates
            ):
                kept_candidates.append(c)
        return kept_candidates

    def validate_geometry(
        self,
        icon_path: Path,
        hits: list[Candidate],
    ) -> list[Candidate]:
        """Filter candidates that deviate from the template aspect ratio.

        Args:
            icon_path: Path to the original template image.
            hits: List of candidates to validate.

        Returns:
            Updated candidates with modified scores based on geometric match.

        """
        template_img = cv2.imread(str(icon_path))
        if template_img is None or not hits:
            return hits

        template_height, template_width = template_img.shape[:2]
        target_ratio = template_width / template_height
        updated_hits = []

        for hit in hits:
            if hit.bbox:
                _, _, w_hit, h_hit = hit.bbox
                current_ratio = w_hit / h_hit

                # Calculate deviation: 1.0 is a perfect aspect ratio match
                deviation = min(target_ratio, current_ratio) / max(
                    target_ratio,
                    current_ratio,
                )

                # Adjusted score = original * (Base Weight + Bonus * Match)
                new_score = hit.score * (
                    GEOM_BASE_SCORE_WEIGHT + (GEOM_RATIO_BONUS_WEIGHT * deviation)
                )

                updated_hits.append(
                    replace(
                        hit,
                        geometry_score=float(deviation),
                        score=new_score,
                    ),
                )
            else:
                updated_hits.append(hit)

        return updated_hits

    def fuse_and_filter(
        self,
        template_hits: list[Candidate],
        ocr_hits: list[Candidate],
        template_size: int,
    ) -> list[Candidate]:
        """Fuse visual template matches with OCR text hits via spatial proximity.

        Args:
            template_hits: Candidates found via visual matching.
            ocr_hits: Candidates found via OCR.
            template_size: Base size used to determine the search radius.

        Returns:
            A unified list of candidates, merged where proximity allows,
            using a 2.0-tiered scoring system (Visual + OCR).

        """
        final_list: list[Candidate] = []
        matched_ocr_indices = set[int]()
        matched_template_indices = set[int]()

        # 1. Fusion Pass: Pair visual hits with proximal OCR hits
        for t_idx, t_hit in enumerate(template_hits):
            potential_ocr_matches = [
                (o_idx, o_hit)
                for o_idx, o_hit in enumerate(ocr_hits)
                if o_idx not in matched_ocr_indices
                and self._euclidean_distance(t_hit, o_hit)
                < template_size * FUSION_DISTANCE_FACTOR
            ]

            if potential_ocr_matches:
                matched_template_indices.add(t_idx)
                # Select the strongest OCR match within range
                _, best_o_hit = max(potential_ocr_matches, key=lambda x: x[1].score)

                for o_idx, _ in potential_ocr_matches:
                    matched_ocr_indices.add(o_idx)

                # Merge 'extra' metadata from both sources
                merged_extra = t_hit.extra.copy()
                merged_extra.update(best_o_hit.extra)

                # Add explicit audit metadata
                merged_extra.update(
                    {
                        "visual_source": t_hit.method,
                        "text_source": best_o_hit.method,
                        "matched_text": (
                            best_o_hit.extra.get("text")
                            or best_o_hit.extra.get("recovered_text")
                        ),
                        "raw_visual_score": t_hit.score,
                        "raw_text_score": best_o_hit.score,
                        "visual_bbox": t_hit.bbox,
                        "text_bbox": best_o_hit.bbox,
                    },
                )

                # Tiered Score Logic: Summation (Range 0.0 to 2.0)
                # Ensures FUSED results always outrank single-source hits.
                fused_score = t_hit.score + best_o_hit.score

                final_list.append(
                    Candidate(
                        x=t_hit.x,
                        y=t_hit.y,
                        score=min(2.0, fused_score),
                        method=DetectionMethod.FUSED,
                        img_score=t_hit.score,
                        txt_score=best_o_hit.score,
                        extra=merged_extra,
                        bbox=t_hit.bbox or best_o_hit.bbox,
                    ),
                )

        # 2. Leftovers Pass: Include non-merged hits
        final_list.extend(
            [
                t
                for i, t in enumerate(template_hits)
                if i not in matched_template_indices
            ],
        )
        final_list.extend(
            [o for j, o in enumerate(ocr_hits) if j not in matched_ocr_indices],
        )

        # 3. Final Spatial Deduplication
        deduped_results: list[Candidate] = []
        dedupe_radius = template_size * max(
            FINAL_DEDUP_RADIUS_FACTOR,
            MIN_DEDUPE_RADIUS_FACTOR,
        )

        # Sorting by score ensures FUSED candidates (>1.0) suppress
        # single-source candidates (<1.0) within the same radius.
        for c in sorted(final_list, key=lambda x: x.score, reverse=True):
            if not any(
                self._euclidean_distance(c, existing) < dedupe_radius
                for existing in deduped_results
            ):
                deduped_results.append(c)

        # Filter by threshold (threshold applies to both 1.0 and 2.0 scales)
        return [cand for cand in deduped_results if cand.score >= self.config.threshold]

    @staticmethod
    def _euclidean_distance(a: Candidate, b: Candidate) -> float:
        """Calculate the Euclidean distance between two candidate centers.

        Args:
            a: The first candidate.
            b: The second candidate.

        Returns:
            The distance as a float.

        """
        return float(np.hypot(a.x - b.x, a.y - b.y))
