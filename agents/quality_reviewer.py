import logging
from models.video_plan import VideoPlan, QualityReview
from agents.base_agent import BaseAgent
from protocols.llm_protocol import ILLMService
from langfuse import observe, get_client

logger = logging.getLogger(__name__)

_ROUND_FOCUS = {
    1: "hook quality — will viewers stop scrolling in the first 3 seconds?",
    2: "content value, storytelling flow, and watch-till-end retention",
    3: "YouTube algorithm factors: searchability, trending potential, CTA, viral hooks",
}

_TOOLS = [
    {
        "name": "submit_review",
        "description": "Submit the quality review with scores and an optional revised plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "hook_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "content_value_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "retention_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "youtube_optimization_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "weaknesses": {"type": "array", "items": {"type": "string"}},
                "specific_improvements": {"type": "array", "items": {"type": "string"}},
                "reviewer_notes": {"type": "string"},
                "approved": {
                    "type": "boolean",
                    "description": "True if score >= 75 and ready to produce",
                },
                "revised_plan": {
                    "type": "object",
                    "description": "Full revised VideoPlan if score < 75. Must include all fields.",
                    "nullable": True,
                },
            },
            "required": [
                "overall_score", "hook_score", "content_value_score",
                "retention_score", "youtube_optimization_score",
                "strengths", "weaknesses", "specific_improvements",
                "reviewer_notes", "approved",
            ],
        },
    }
]


class QualityReviewAgent(BaseAgent):
    def __init__(self, llm_service: ILLMService):
        super().__init__(llm_service, "QualityReviewer")
        self._review_data: dict = {}

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "submit_review":
            self._review_data = inputs
            return {"status": "ok"}
        return {"error": f"Unknown tool: {name}"}

    @observe()
    def review(
        self,
        script: str,
        plan: VideoPlan,
        round_number: int,
        prev_reviews: list[QualityReview] | None = None,
    ) -> QualityReview:
        self._review_data = {}

        focus = _ROUND_FOCUS[round_number]
        prev = ""
        if prev_reviews:
            prev = "\n\nPrevious rounds:\n" + "\n".join(
                f"  Round {r.round_number}: {r.overall_score}/100 — {r.reviewer_notes}"
                for r in prev_reviews
            )

        system = f"""You are a ruthless YouTube growth expert who has analysed 10,000+ viral videos.

REVIEW ROUND {round_number}/3 — Focus: {focus}

Score guide:
  90–100 → Viral potential, will definitely perform
  75–89  → Above-average, likely to do well
  60–74  → Decent, needs improvement
  <60    → Significant issues, must revise

If overall_score < 75 you MUST provide a complete revised_plan.
The revised plan must preserve these fields exactly:
  video_type: {plan.video_type}
  visual_style: {plan.visual_style}
  video_format: {plan.video_format}
  target_audience: {plan.target_audience}

Be specific in your improvements. Call submit_review to return your assessment.{prev}"""

        plan_json = plan.model_dump_json(indent=2)
        messages = [{
            "role": "user",
            "content": f"Original script:\n{script}\n\nCurrent video plan:\n{plan_json}",
        }]
        self._run_tool_loop(messages, system, _TOOLS, max_tokens=6000)

        if not self._review_data:
            raise RuntimeError(f"QualityReviewAgent: no review returned for round {round_number}")

        review_data = dict(self._review_data)

        # Merge preserved fields into revised_plan if provided
        if review_data.get("revised_plan"):
            rp = review_data["revised_plan"]

            # Model sometimes returns revised_plan as a JSON string instead of a dict
            if isinstance(rp, str):
                import json as _json
                try:
                    rp = _json.loads(rp)
                    review_data["revised_plan"] = rp
                except Exception:
                    review_data.pop("revised_plan", None)
                    rp = None

            if isinstance(rp, dict):
                # Always force-set preserved fields — prevents enum repr strings
                # (e.g. 'VideoType.EDUCATIONAL') that the model sometimes returns
                for field in ("video_type", "visual_style", "video_format", "target_audience"):
                    val = getattr(plan, field)
                    rp[field] = val.value if hasattr(val, "value") else val
                if "scenes" in rp and "total_duration_seconds" not in rp:
                    rp["total_duration_seconds"] = sum(
                        s.get("duration_seconds", 5) for s in rp["scenes"]
                    )
                for field in ("title", "description", "hook", "background_music_style",
                              "call_to_action", "thumbnail_concept"):
                    if field not in rp:
                        rp[field] = getattr(plan, field)

        try:
            review = QualityReview(round_number=round_number, **review_data)
        except Exception as exc:
            logger.warning("QualityReview parse failed (round %d): %s", round_number, exc)
            review_data.pop("revised_plan", None)
            review = QualityReview(round_number=round_number, **review_data)

        get_client().update_current_span(
            metadata={
                "round": round_number,
                "focus": _ROUND_FOCUS[round_number],
                "overall_score": review.overall_score,
                "hook_score": review.hook_score,
                "content_value_score": review.content_value_score,
                "retention_score": review.retention_score,
                "youtube_optimization_score": review.youtube_optimization_score,
                "plan_revised": review.revised_plan is not None,
            },
        )
        get_client().score_current_span(
            name=f"quality_score_round_{round_number}",
            value=review.overall_score / 100,
            comment=review.reviewer_notes,
        )
        return review

    @observe(name="quality-review-full")
    def run_full_review(
        self, script: str, plan: VideoPlan
    ) -> tuple[VideoPlan, list[QualityReview]]:
        """Run all 3 review rounds, revising plan as needed. Returns (final_plan, reviews)."""
        history: list[QualityReview] = []  # local — no instance bleed between pipeline runs
        current = plan

        for rnd in range(1, 4):
            print(f"\n  [Review {rnd}/3] Focus: {_ROUND_FOCUS[rnd]}")
            review = self.review(script, current, rnd, prev_reviews=history)
            history.append(review)

            bar = "#" * (review.overall_score // 10) + "-" * (10 - review.overall_score // 10)
            print(f"  Score: {review.overall_score}/100 [{bar}]  Approved: {review.approved}")
            print(f"  + {' | '.join(review.strengths[:2])}")
            print(f"  - {' | '.join(review.weaknesses[:2])}")

            if review.revised_plan:
                current = review.revised_plan
                print("  ~ Plan revised")

        final_score = history[-1].overall_score
        print(f"\n  Final score after 3 reviews: {final_score}/100")
        if final_score < 60:
            print("  ! Score below 60 -- proceeding with best available plan")

        get_client().score_current_span(
            name="final_quality_score",
            value=final_score / 100,
            comment=f"Scores per round: {[r.overall_score for r in history]}",
        )
        return current, history
