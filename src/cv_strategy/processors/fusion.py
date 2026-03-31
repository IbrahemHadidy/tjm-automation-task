"""Spatial fusion and result reconciliation logic for UI grounding.

Manage Non-Maximum Suppression (NMS), geometric validation of visual hits,
and the strategic merging of visual and OCR candidates.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_strategy.constants import (
    FINAL_DEDUP_RADIUS_FACTOR,
    FUSION_DISTANCE_FACTOR,
    FUSION_SCORE_BONUS,
    GEOM_BASE_SCORE_WEIGHT,
    GEOM_RATIO_BONUS_WEIGHT,
    MAX_TEMPLATE_HITS,
    MIN_DEDUPE_RADIUS_FACTOR,
    NMS_RADIUS_FACTOR,
)
from cv_strategy.models import Candidate, DetectionMethod

if TYPE_CHECKING:
    from pathlib import Path

    from cv_strategy.models import GroundingConfig


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

    def apply_smart_nms(
        self,
        candidates: list[Candidate],
        default_size: int = 32,
    ) -> list[Candidate]:
        """Apply scale-aware non-maximum suppression.

        Unlike standard NMS, this calculates a unique suppression radius for
        each candidate based on its detected 'base_size'. This allows small
        toolbar icons and large desktop icons to coexist without
        accidentally suppressing each other.

        Args:
            candidates: List of raw detections from all scale passes.
            default_size: Fallback pixel width if a candidate lacks scale metadata.

        Returns:
            Deduplicated list of candidates.

        """
        if not candidates:
            return []

        # Sort by score descending
        sorted_hits = sorted(candidates, key=lambda x: x.score, reverse=True)[
            :MAX_TEMPLATE_HITS
        ]
        kept_candidates: list[Candidate] = []

        for c in sorted_hits:
            # Use the specific size this icon was found with, or fallback
            local_size = c.extra.get("base_size", default_size)
            threshold_sq = (local_size * NMS_RADIUS_FACTOR) ** 2

            if not any(
                ((c.x - a.x) ** 2 + (c.y - a.y) ** 2) < threshold_sq
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
        default_size: int = 32,
    ) -> list[Candidate]:
        """Fuse visual template matches with OCR text hits via spatial proximity.

        This method performs multi-modal reconciliation by pairing visual anchors
        with nearby OCR results. It uses 'Local Scaling', meaning the search
        radius for a match is derived from each candidate's specific detected
        base size rather than a global constant.

        Args:
            template_hits: Candidates found via visual matching (carrying 'base_size').
            ocr_hits: Candidates found via global or recovery OCR sweeps.
            default_size: Fallback pixel width used if a template hit lacks
                specific scale metadata. Defaults to 32.

        Returns:
            A unified list of candidates, merged where proximity allows.
            Fused results use a tiered scoring system (0.0 to 2.0) to ensure
            multi-modal evidence always outranks single-source hits.

        """
        final_list: list[Candidate] = []
        matched_ocr_indices = set[int]()
        matched_template_indices = set[int]()

        # 1. Fusion Pass
        for t_idx, t_hit in enumerate(template_hits):
            # Determine the ruler for THIS specific template hit
            local_size = t_hit.extra.get("base_size", default_size)
            fusion_radius = local_size * FUSION_DISTANCE_FACTOR

            potential_ocr_matches = [
                (o_idx, o_hit)
                for o_idx, o_hit in enumerate(ocr_hits)
                if o_idx not in matched_ocr_indices
                and self._euclidean_distance(t_hit, o_hit) < fusion_radius
            ]

            if potential_ocr_matches:
                matched_template_indices.add(t_idx)
                _, best_o_hit = max(potential_ocr_matches, key=lambda x: x[1].score)

                for o_idx, _ in potential_ocr_matches:
                    matched_ocr_indices.add(o_idx)

                # Merge Logic (Keep the base_size in the fused candidate)
                merged_extra = t_hit.extra.copy()
                merged_extra.update(best_o_hit.extra)

                fused_score = min(
                    2.0,
                    t_hit.score + best_o_hit.score + FUSION_SCORE_BONUS,
                )

                final_list.append(
                    replace(
                        t_hit,
                        score=fused_score,
                        method=DetectionMethod.FUSED,
                        img_score=t_hit.score,
                        txt_score=best_o_hit.score,
                        extra=merged_extra,
                        bbox=t_hit.bbox or best_o_hit.bbox,
                    ),
                )

        # 2. Add Leftovers
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

        # 3. Final Spatial Deduplication using local candidate sizes
        deduped_results: list[Candidate] = []
        for c in sorted(final_list, key=lambda x: x.score, reverse=True):
            local_size = c.extra.get("base_size", default_size)
            dedupe_radius = local_size * max(
                FINAL_DEDUP_RADIUS_FACTOR,
                MIN_DEDUPE_RADIUS_FACTOR,
            )

            if not any(
                self._euclidean_distance(c, existing) < dedupe_radius
                for existing in deduped_results
            ):
                deduped_results.append(c)

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
