"""Hacker News-specific fetched content cleanup."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import ParseResult

from mother.tools.cleaners.base import ContentCleaner

_HN_URL_PATTERN = r"https?://news\.ycombinator\.com"
_HN_VOTE_LINK_RE = re.compile(rf"{_HN_URL_PATTERN}/vote\?id=")
_HN_ITEM_LINK_RE = re.compile(
    rf"\[(?P<count>\d+\s+comments?)\]\({_HN_URL_PATTERN}/item\?id=\d+[^)]*\)"
)
_HN_COMMENT_HEADER_RE = re.compile(
    rf"^\[(?P<user>[^\]]+)\]\({_HN_URL_PATTERN}/user\?id=[^)]+\)"
    + rf"\[(?P<age>[^\]]+)\]\({_HN_URL_PATTERN}/item\?id=\d+[^)]*\)"
    + r"(?P<tail>[\s\S]*)$"
)
_HN_SEGMENT_SPLIT_RE = re.compile(
    rf"(?:!?\[Image \d+\]\({_HN_URL_PATTERN}/s\.gif\))?" + rf"\[\]\({_HN_URL_PATTERN}/vote\?id="
)
_HN_PLAIN_STATS_RE = re.compile(
    r"^(?P<points>\d+\s+points?)\s+by\s+(?P<submitter>\S+)\s+"
    + r"(?P<age>.+?)\s+\|\s+.*?(?P<comments>\d+\s+comments?)\s*$"
)
_HN_PLAIN_COMMENT_HEADER_RE = re.compile(r"^(?P<user>\S+)\s+(?P<age>.+?\bago)\s+\|\s+.+\[[^\]]+\]$")


def _simplify_markdown_text(value: str) -> str:
    cleaned = value
    cleaned = re.sub(r"!?\[Image \d+\]\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[\[[^\]]*\]\]\(javascript:void\(0\)\)", "", cleaned)
    cleaned = re.sub(r"\[\]\([^)]*\)", "", cleaned)

    for label in (
        "reply",
        "parent",
        "next",
        "prev",
        "root",
        "hide",
        "favorite",
        "past",
        "help",
        "login",
    ):
        cleaned = re.sub(rf"\[{label}\]\([^)]+\)", "", cleaned, flags=re.IGNORECASE)

    def _replace_markdown_link(match: re.Match[str]) -> str:
        text = match.group(1).strip()
        if not text:
            return ""
        return text

    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_markdown_link, cleaned)
    cleaned = re.sub(r"\s+\|\s+", " • ", cleaned)
    cleaned = re.sub(r"(?:\s*•\s*){2,}", " • ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _format_story_header(segment: str) -> list[str]:
    rest = segment
    lines = ["Hacker News discussion"]

    title_match = re.match(r"^\[(?P<title>[^\]]+)\]\((?P<link>[^)]+)\)(?P<tail>[\s\S]*)$", rest)
    if title_match is not None:
        lines.extend(
            [
                "",
                f"Title: {title_match.group('title')}",
                f"Article: {title_match.group('link')}",
            ]
        )
        rest = title_match.group("tail")

    site_match = re.match(r"^\s*\(\[(?P<site>[^\]]+)\]\([^)]+\)\)(?P<tail>[\s\S]*)$", rest)
    if site_match is not None:
        lines.append(f"Site: {site_match.group('site')}")
        rest = site_match.group("tail")

    stats_match = re.search(
        r"(?P<points>\d+\s+points?)\s+by\s+\[(?P<submitter>[^\]]+)\]\([^)]+\)"
        + r"\[(?P<age>[^\]]+)\]\([^)]+\)",
        rest,
    )
    if stats_match is not None:
        lines.append(f"Points: {stats_match.group('points')}")
        lines.append(f"Submitter: {stats_match.group('submitter')}")
        lines.append(f"Posted: {stats_match.group('age')}")

    comments_match = _HN_ITEM_LINK_RE.search(rest)
    if comments_match is not None:
        lines.append(f"Comments: {comments_match.group('count')}")

    extra_links = _simplify_markdown_text(rest)
    extra_links = re.sub(
        r"^.*?\d+\s+points?\s+by\s+.+?comments",
        "",
        extra_links,
        flags=re.IGNORECASE,
    ).strip()
    extra_links = re.sub(r"^•\s*", "", extra_links).strip()
    if extra_links:
        lines.extend(["", f"Links: {extra_links}"])

    return lines


def _extract_page_title(lines: Sequence[str]) -> str | None:
    for line in lines:
        if not line.startswith("Title: "):
            continue
        title = re.sub(
            r"\s+\|\s+Hacker News$",
            "",
            line.removeprefix("Title: "),
            flags=re.IGNORECASE,
        )
        normalized_title = title.strip()
        if normalized_title:
            return normalized_title
        return None
    return None


def _looks_like_story_header(segment: str) -> bool:
    return bool(
        re.match(r"^\[[^\]]+\]\([^)]+\)", segment)
        or re.search(r"\d+\s+points?\s+by\s+\[", segment)
        or _HN_ITEM_LINK_RE.search(segment)
    )


def _format_comment(segment: str, index: int) -> str | None:
    comment_match = _HN_COMMENT_HEADER_RE.match(segment)
    if comment_match is None:
        fallback = _simplify_markdown_text(segment)
        if not fallback:
            return None
        return f"{index}. {fallback}"

    body = comment_match.group("tail")
    body = re.sub(
        r"^(?:\s*\|\s*\[(?:parent|next|prev|root)\]\([^)]+\))+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"^\s*\[\[[^\]]*\]\]\(javascript:void\(0\)\)\s*", "", body, flags=re.IGNORECASE)
    body = _simplify_markdown_text(body)
    body = re.sub(r"^•\s*", "", body).strip()
    if not body:
        return None

    return f"{index}. {comment_match.group('user')} — {comment_match.group('age')}\n{body}"


def _normalize_plain_line(line: str) -> str:
    return line.replace("\u00a0", " ").strip()


def _collapse_plain_text_paragraphs(lines: Sequence[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    for raw_line in lines:
        line = _normalize_plain_line(raw_line)
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs).strip()


def _clean_plain_text_hacker_news_item_content(lines: Sequence[str]) -> str | None:
    normalized_lines = [_normalize_plain_line(line) for line in lines]
    stats_index = next(
        (index for index, line in enumerate(normalized_lines) if _HN_PLAIN_STATS_RE.match(line)),
        -1,
    )
    if stats_index < 0:
        return None

    stats_match = _HN_PLAIN_STATS_RE.match(normalized_lines[stats_index])
    if stats_match is None:
        return None

    title_line = ""
    for index in range(stats_index - 1, -1, -1):
        candidate = normalized_lines[index]
        if (
            not candidate
            or candidate.startswith("#")
            or re.match(r"^Hacker News", candidate, flags=re.IGNORECASE)
            or candidate == "help"
        ):
            continue
        title_line = candidate
        break

    output = ["Hacker News discussion"]
    if title_line:
        title_match = re.match(r"^(?P<title>.+?)\s+\((?P<site>[^()]+\.[^()]+)\)$", title_line)
        if title_match is not None:
            output.extend(
                [
                    "",
                    f"Title: {title_match.group('title')}",
                    f"Site: {title_match.group('site')}",
                ]
            )
        else:
            output.extend(["", f"Title: {title_line}"])

    output.append(f"Points: {stats_match.group('points')}")
    output.append(f"Submitter: {stats_match.group('submitter')}")
    output.append(f"Posted: {stats_match.group('age')}")
    output.append(f"Comments: {stats_match.group('comments')}")

    comments: list[str] = []
    index = stats_index + 1
    while index < len(normalized_lines):
        line = normalized_lines[index]
        comment_match = _HN_PLAIN_COMMENT_HEADER_RE.match(line)
        if comment_match is None:
            index += 1
            continue

        body_lines: list[str] = []
        index += 1
        while index < len(normalized_lines):
            current_line = normalized_lines[index]
            if not current_line:
                body_lines.append("")
                index += 1
                continue
            if current_line == "reply":
                break
            if re.match(r"^Guidelines\b", current_line, flags=re.IGNORECASE):
                index = len(normalized_lines)
                break
            if _HN_PLAIN_COMMENT_HEADER_RE.match(current_line):
                index -= 1
                break
            body_lines.append(current_line)
            index += 1

        body = _collapse_plain_text_paragraphs(body_lines)
        if body:
            comments.append(
                f"{len(comments) + 1}. {comment_match.group('user')} — {comment_match.group('age')}\n{body}"
            )

        index += 1

    if comments:
        output.extend(["", "Comments:", "", "\n\n".join(comments)])

    cleaned = "\n".join(output).strip()
    return cleaned or None


def _clean_hacker_news_item_content(body: str) -> str:
    lines = [line.rstrip() for line in body.splitlines()]
    page_title = _extract_page_title(lines)
    markdown_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "Markdown Content:"),
        -1,
    )
    markdown_lines = lines[markdown_index + 1 :] if markdown_index >= 0 else lines
    discussion_line = next((line for line in markdown_lines if _HN_VOTE_LINK_RE.search(line)), None)
    if discussion_line is not None:
        segments = [
            re.sub(r"^\d+&how=up&goto=[^)]*\)", "", segment).strip()
            for segment in _HN_SEGMENT_SPLIT_RE.split(discussion_line)
        ]
        segments = [segment for segment in segments if segment]
        if segments:
            has_story_header = _looks_like_story_header(segments[0])
            output = (
                _format_story_header(segments[0])
                if has_story_header
                else ["Hacker News discussion"]
            )
            if not has_story_header and page_title:
                output.extend(["", f"Title: {page_title}"])

            comment_segments = segments[1:] if has_story_header else segments
            comments = [
                comment
                for position, segment in enumerate(comment_segments, start=1)
                if (comment := _format_comment(segment, position)) is not None
            ]
            if comments:
                output.extend(["", "Comments:", "", "\n\n".join(comments)])

            cleaned = "\n".join(output).strip()
            if cleaned:
                return cleaned

    return _clean_plain_text_hacker_news_item_content(markdown_lines) or body


def _matches_hacker_news_item(url: ParseResult) -> bool:
    return url.hostname == "news.ycombinator.com" and url.path == "/item"


hacker_news_cleaner = ContentCleaner(
    id="hacker-news-item",
    matches=_matches_hacker_news_item,
    clean=_clean_hacker_news_item_content,
)
