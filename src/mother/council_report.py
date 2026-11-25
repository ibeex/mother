"""Reader-facing council comparison; the session trace remains lossless and separate."""

from __future__ import annotations

import re
from collections import Counter

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mother.council import CouncilPeerReview, CouncilResult
from mother.widgets import Response


def _review_note(review: CouncilPeerReview, label: str) -> str:
    """Quote only clearly labeled reviewer comments, never the ballot itself."""
    body = re.split(r"FINAL RANKING:", review.text, maxsplit=1, flags=re.I)[0]
    match = re.search(
        rf"^[ \t]*(?:#{{1,6}}[ \t]+|[-*][ \t]+)?\**{re.escape(label)}\**"
        + r"(?:[:—-]\**[ \t]*(?:\n[ \t]*)?|[ \t]*\n[ \t]*)(\S[^\n]*)",
        body,
        re.I | re.M,
    )
    if not match:
        return ""
    note = match.group(1).strip()
    if not note or re.match(r"(?:[#* -]*Response [A-Z]\b|\d+\.)", note, re.I):
        return ""
    return re.split(r'(?<=[.!?])\s+(?=[A-Z“"])', note, maxsplit=1)[0]


def _named_feedback(text: str, result: CouncilResult) -> str:
    """Resolve explicit labels without replacing articles or terms like 'C compiler'."""

    def name(match: re.Match[str]) -> str:
        letter = match.group(1).upper()
        return result.label_to_model.get(f"Response {letter}", match.group())

    text = re.sub(r"\bResponse\s+([A-Z]+)\b", name, text, flags=re.I)
    # Reviewers also write comparisons like 'more concise than A, but ...'.
    return re.sub(r"(?<=than )([A-Z])(?=[.,;:]| and | but |$)", name, text)


def ballot_markdown(result: CouncilResult) -> str:
    """Render the recorded order of each ballot, leaving missing places empty."""
    places = max(len(result.stage1), *(len(review.parsed_ranking) for review in result.stage2), 1)
    rows = [["Reviewer", *(f"#{i + 1}" for i in range(places))], ["---"] * (places + 1)]
    for review in result.stage2:
        names = [
            result.label_to_model.get(label, "Unknown answer") for label in review.parsed_ranking
        ]
        rows.append([review.reviewer_model_id, *names, *(["—"] * (places - len(names)))])
    return "\n".join(
        "| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |" for row in rows
    )


def _verdict(result: CouncilResult) -> str:
    rankings = result.aggregate_rankings
    if not rankings:
        return "No peer ranking available."
    best = min(rank.average_rank for rank in rankings)
    leaders = [r.label for r in rankings if r.average_rank == best]
    names = ", ".join(result.label_to_model[label] for label in leaders)
    firsts = Counter(
        review.parsed_ranking[0]
        for review in result.stage2
        if review.parsed_ranking and review.parsed_ranking[0] in result.label_to_model
    )
    tied = len(leaders) > 1
    headline = f"**{names}** tied for best average rank" if tied else f"**{names}** ranked highest"
    if len(firsts) == len(result.stage1) > 1 and set(firsts.values()) == {1}:
        return f"{headline}. Each answer received one first-place vote."
    if firsts and not tied:
        total = firsts.total()
        return f"{headline} — {firsts[leaders[0]]} of {total} first-place {'votes' if total != 1 else 'vote'}."
    return f"{headline}."


def comparison_markdown(result: CouncilResult) -> str:
    """Summarize observed ranks and attributed feedback, without another model call."""
    lines = ["### How the answers compared", ""]
    if result.used_fallback:
        lines.extend(["The judge was unavailable; the reply above is a candidate answer.", ""])
    ranks = {rank.label: rank for rank in result.aggregate_rankings}
    notes = {note.subject: note for note in result.assessments}
    headline = _verdict(result)
    if disagreement := notes.get("Disagreement"):
        headline += f" {_named_feedback(disagreement.text, result)}"
    lines.extend([headline, ""])
    if ranks:
        lines.extend(["*Average rank · lower is better*", ""])
    for candidate in sorted(
        result.stage1,
        key=lambda item: ranks[item.label].average_rank if item.label in ranks else float("inf"),
    ):
        label = candidate.label
        rank = ranks.get(label)
        average = "not ranked"
        if rank:
            average = f"{rank.average_rank:.2f}"
            if rank.rankings_count < len(result.stage2):
                average += f" ({rank.rankings_count} of {len(result.stage2)} reviews)"
        lines.append(f"- **{candidate.model_id}** · {average}")
        if assessment := notes.get(label):
            lines[-1] += f" — {_named_feedback(assessment.text, result)}"
            continue
        # Prefer a review that actually ranked this candidate first; attribute the
        # excerpt so a single reviewer's opinion is never presented as consensus.
        for review in sorted(result.stage2, key=lambda r: r.parsed_ranking[:1] != (label,)):
            note = _review_note(review, label)
            if note:
                lines[-1] += f" — {review.reviewer_model_id}: “{_named_feedback(note, result)}”"
                break
    return "\n".join(lines)


def comparison_sources_markdown(result: CouncilResult) -> str:
    """Show shared citations once; break them out only when the sources differ."""
    notes = result.assessments
    if not notes:
        return ""
    shared = set(notes[0].reviews)
    if all(set(note.reviews) == shared for note in notes[1:]):
        sources = notes[0].sources
        if len(shared) > 1 and shared == set(range(1, len(result.stage2) + 1)):
            sources = f"all {len(shared)} reviews"
        scope = "the comparison note" if len(notes) == 1 else "every comparison note"
        return f"The judge cites {sources} for {scope}."
    rows = [
        f"- {result.label_to_model.get(note.subject, note.subject)}: {note.sources}"
        for note in notes
    ]
    return "**Sources for the judge’s comparison**\n\n" + "\n".join(rows)


def build_council_report(result: CouncilResult) -> Vertical:
    """Build in-flow, keyboard-expandable details below the synthesized reply."""
    answers = [
        Collapsible(Response(candidate.text), title=candidate.model_id)
        for candidate in result.stage1
    ]
    details = [Collapsible(*answers, title=f"Individual answers ({len(answers)})")]
    reviews: list[Collapsible] = []
    for index, review in enumerate(result.stage2, start=1):
        reviews.append(
            Collapsible(
                Response(_named_feedback(review.text, result)),
                title=f"Review {index} · {review.reviewer_model_id}",
            )
        )
    if reviews:
        evidence = ballot_markdown(result)
        if sources := comparison_sources_markdown(result):
            evidence += f"\n\n{sources}"
        details.append(
            Collapsible(
                Response(evidence), *reviews, title=f"Peer reviews & ballots ({len(reviews)})"
            )
        )
    metadata = f"Judge: {result.judge_model_id}"
    if result.duration_seconds is not None:
        metadata += f" · {result.duration_seconds:.2f}s"
    if result.used_fallback:
        metadata += f" · Fallback: {result.fallback_reason or 'unknown reason'}"
    footer = Static(metadata, markup=False, classes="council-footer")
    return Vertical(
        Response(comparison_markdown(result)), *details, footer, classes="council-report"
    )
