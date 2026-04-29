#!/usr/bin/env python3
# Copyright (c) 2026 Aisle Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal AI Act biometric-risk POC.

This is intentionally separate from scan.py. It looks for source files that
appear to handle biometric signals, asks OpenCode with DeepSeek to plan and
apply remediation, then can open a draft GitHub PR with the generated
compliance files plus any code changes OpenCode made.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VERSION = "0.1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_VARIANT = "high"
DEFAULT_BASE_BRANCH = "main"
DEFAULT_BRANCH_PREFIX = "codex/ai-act-biometric-review"
GITHUB_API_URL = "https://api.github.com"

SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".compliance",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

AI_ACT_CONTEXT = """\
EU AI Act context for this POC:
- Article 3 defines biometric identification as automated recognition of human features to establish identity by comparing biometric data to stored biometric data.
- Article 3 defines biometric verification as one-to-one identity confirmation against previously provided biometric data.
- Article 3 defines emotion recognition as identifying or inferring emotions or intentions on the basis of biometric data.
- Article 3 defines biometric categorisation as assigning natural persons to categories on the basis of biometric data.
- Annex III lists remote biometric identification, sensitive/protected biometric categorisation, and emotion recognition as high-risk biometric AI areas where permitted by law.
- Article 5 prohibits some biometric uses, including untargeted scraping to build facial recognition databases, workplace/education emotion inference except medical or safety uses, and biometric categorisation to infer listed sensitive traits.
- Article 50 requires deployers of emotion recognition or biometric categorisation systems to inform exposed persons.

Source references:
- Article 3 definitions: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-3
- Annex III high-risk biometric areas: https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3
- Article 5 prohibited AI practices: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-5
- Article 50 transparency obligations: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50

Positive examples for this POC:
- webcam/video/image/audio/voice input used to infer emotion, stress, mood, intent, or affect;
- face embeddings, face_recognition, face-api, Rekognition, Azure Face, DeepFace, dlib, OpenCV face detection, or facial landmark code used for matching or identification;
- fingerprint, iris, retina, voiceprint, gait, liveness, or other biometric-template matching;
- age, sex, ethnicity, race, disability, health, political, religious, sexual orientation, or similar categories inferred from face, voice, gait, or other biometric data.

Negative examples that should not trigger biometric AI Act remediation by themselves:
- text-only sentiment analysis of chats, emails, support tickets, or relationship messages;
- product copy that says emotional, emotion, face a consequence, interface, or user-facing;
- one-to-one biometric login verification where the sole purpose is confirming the claimed user identity, unless other biometric categorisation or identification signals appear.
"""

BIOMETRIC_EVIDENCE_PATTERNS: Sequence[Tuple[str, str, re.Pattern[str]]] = (
    (
        "face recognition",
        "biometric identification or verification",
        re.compile(r"\bface[-_\s]?recognition\b|\bcompare_faces\b|\bface_encodings?\b", re.IGNORECASE),
    ),
    (
        "facial recognition",
        "biometric identification or verification",
        re.compile(r"\bfacial[-_\s]?recognition\b", re.IGNORECASE),
    ),
    (
        "face biometric workflow",
        "biometric identification or verification",
        re.compile(
            r"\bface[-_\s]?(?:detect(?:or|ion)?|match(?:ing)?|verification|verify|identify|identification|embedding|landmark|liveness|login|auth)\b",
            re.IGNORECASE,
        ),
    ),
    ("fingerprint", "biometric identification or verification", re.compile(r"\bfingerprints?\b", re.IGNORECASE)),
    ("iris", "biometric identification or verification", re.compile(r"\biris\b", re.IGNORECASE)),
    ("retina", "biometric identification or verification", re.compile(r"\bretina(?:l)?\b", re.IGNORECASE)),
    ("voiceprint", "biometric identification or verification", re.compile(r"\bvoice\s*prints?\b|\bvoiceprints?\b", re.IGNORECASE)),
    ("gait", "biometric identification or verification", re.compile(r"\bgait\b", re.IGNORECASE)),
    ("biometric", "biometric identification, categorisation, or verification", re.compile(r"\bbiometrics?\b", re.IGNORECASE)),
    ("Rekognition", "biometric identification or verification", re.compile(r"\brekognition\b", re.IGNORECASE)),
    ("Azure Face", "biometric identification or verification", re.compile(r"\bazure\s+face\b", re.IGNORECASE)),
    ("DeepFace", "biometric identification or verification", re.compile(r"\bdeepface\b", re.IGNORECASE)),
    ("dlib", "biometric identification or verification", re.compile(r"\bdlib\b", re.IGNORECASE)),
    ("OpenCV", "possible biometric image processing", re.compile(r"\bopencv\b|\bcv2\b|\bCascadeClassifier\b|\bdetectMultiScale\b", re.IGNORECASE)),
    ("face-api", "biometric identification or verification", re.compile(r"\bface-api(?:\.js)?\b", re.IGNORECASE)),
)

EMOTION_WITH_BIOMETRIC_INPUT_PATTERNS: Sequence[Tuple[str, str, re.Pattern[str]]] = (
    (
        "emotion recognition from biometric input",
        "emotion recognition on biometric data",
        re.compile(
            r"\b(?:emotion|affect|mood|stress|intent(?:ion)?)\b.{0,80}\b(?:face|facial|camera|webcam|video|voice|audio|biometric)\b"
            r"|\b(?:face|facial|camera|webcam|video|voice|audio|biometric)\b.{0,80}\b(?:emotion|affect|mood|stress|intent(?:ion)?)\b",
            re.IGNORECASE,
        ),
    ),
)

CONTEXTUAL_EMOTION_PATTERNS: Sequence[Tuple[str, str, re.Pattern[str]]] = (
    (
        "emotion signal near biometric processing",
        "emotion recognition on biometric data",
        re.compile(r"\bemotion(?:al)?\b|\baffect\b|\bmood\b|\bstress\b|\bintent(?:ion)?\b", re.IGNORECASE),
    ),
)

BIOMETRIC_INPUT_CONTEXT_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bwebcam\b|\bcamera\b|\bvideo\b|\baudio\b|\bvoice\b|\bfacial\b|\bbiometric\b", re.IGNORECASE),
)

REQUIRED_DOCUMENTS = [
    "AI system inventory entry",
    "EU AI Act biometric-risk classification assessment",
    "Fundamental rights impact assessment or documented rationale for why one is not required",
    "Technical documentation for model purpose, inputs, outputs, and risk controls",
    "Human oversight, logging, monitoring, and incident-response notes",
]

NO_TRIGGER_DOCUMENTS = [
    "No biometric-specific AI Act document is required by this POC result.",
    "Optionally record a short out-of-scope rationale if the repo performs text-only sentiment or relationship analysis.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_rel_path(path: Path) -> str:
    return path.as_posix()


def iter_source_files(repo_path: Path, max_chars: int = 200_000) -> Iterable[Path]:
    for root, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        root_path = Path(root)
        for filename in sorted(filenames):
            path = root_path / filename
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > max_chars:
                    continue
            except OSError:
                continue
            yield path


def make_signal_match(
    line_no: int,
    label: str,
    concept: str,
    line: str,
    basis: str,
) -> Dict[str, object]:
    return {
        "line": line_no,
        "term": label,
        "ai_act_concept": concept,
        "basis": basis,
        "snippet": line.strip()[:240],
    }


def collect_pattern_matches(
    lines: Sequence[str],
    patterns: Sequence[Tuple[str, str, re.Pattern[str]]],
    basis: str,
    max_matches: int,
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    for line_no, line in enumerate(lines, 1):
        for label, concept, pattern in patterns:
            if pattern.search(line):
                matches.append(make_signal_match(line_no, label, concept, line, basis))
                break
        if len(matches) >= max_matches:
            break
    return matches


def has_biometric_input_context(lines: Sequence[str]) -> bool:
    return any(pattern.search(line) for line in lines for pattern in BIOMETRIC_INPUT_CONTEXT_PATTERNS)


def find_biometric_signals(repo_path: Path, max_matches_per_file: int = 8) -> List[Dict[str, object]]:
    signals: List[Dict[str, object]] = []
    repo_path = repo_path.resolve()

    for path in iter_source_files(repo_path):
        try:
            text = read_text(path)
        except OSError:
            continue

        lines = text.splitlines()
        direct_matches = collect_pattern_matches(
            lines,
            BIOMETRIC_EVIDENCE_PATTERNS,
            "direct biometric-data processing signal",
            max_matches_per_file,
        )
        emotion_biometric_matches = collect_pattern_matches(
            lines,
            EMOTION_WITH_BIOMETRIC_INPUT_PATTERNS,
            "emotion inference tied to biometric input on the same line",
            max_matches_per_file,
        )
        matches = direct_matches + emotion_biometric_matches

        if direct_matches and len(matches) < max_matches_per_file:
            contextual_matches = collect_pattern_matches(
                lines,
                CONTEXTUAL_EMOTION_PATTERNS,
                "emotion-related term appears in a file with biometric-data processing",
                max_matches_per_file - len(matches),
            )
            matches.extend(contextual_matches)

        if not direct_matches and not emotion_biometric_matches and has_biometric_input_context(lines):
            contextual_matches = collect_pattern_matches(
                lines,
                CONTEXTUAL_EMOTION_PATTERNS,
                "emotion-related term appears in a file with camera, video, audio, voice, facial, or biometric input context",
                max_matches_per_file,
            )
            matches.extend(contextual_matches)

        matches = matches[:max_matches_per_file]

        if matches:
            rel = path.relative_to(repo_path)
            signals.append({"file": normalize_rel_path(rel), "matches": matches})

    return signals


def infer_required_documents(signals: List[Dict[str, object]]) -> List[str]:
    if signals:
        return REQUIRED_DOCUMENTS
    return NO_TRIGGER_DOCUMENTS


def infer_suspected_triggers(signals: List[Dict[str, object]]) -> List[str]:
    if not signals:
        return []

    triggers = set()
    for item in signals:
        for match in item["matches"]:
            concept = str(match.get("ai_act_concept", ""))
            if "emotion recognition" in concept:
                triggers.add("emotion recognition on biometric data")
            elif "categorisation" in concept or "categorization" in concept:
                triggers.add("biometric categorisation")
            elif "verification" in concept:
                triggers.add("biometric identification or verification review")
            else:
                triggers.add("biometric identification or categorisation review")

    return sorted(triggers)


def build_report(repo_path: Path, signals: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "schema_version": VERSION,
        "generated_at": utc_now(),
        "target": repo_path.resolve().name,
        "risk_domain": "eu_ai_act_biometric",
        "status": "possible_trigger_found" if signals else "no_prefilter_signals",
        "manual_review_required": True,
        "summary": {
            "files_with_biometric_signals": len(signals),
            "total_signal_lines": sum(len(item["matches"]) for item in signals),
        },
        "suspected_ai_act_triggers": infer_suspected_triggers(signals),
        "required_documents": infer_required_documents(signals),
        "signals": signals,
        "ai_act_context": AI_ACT_CONTEXT,
        "disclaimer": (
            "This POC reports possible compliance triggers from source-code signals. "
            "It is not a legal conclusion and requires review by qualified counsel or a DPO."
        ),
    }


def render_report_markdown(report: Dict[str, object]) -> str:
    summary = report["summary"]
    signals = report["signals"]
    lines = [
        "# AI Act biometric-risk report",
        "",
        f"- Target: `{report['target']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Files with signals: `{summary['files_with_biometric_signals']}`",
        f"- Signal lines: `{summary['total_signal_lines']}`",
        "",
        "## Suspected triggers",
        "",
    ]
    for trigger in report["suspected_ai_act_triggers"]:
        lines.append(f"- {trigger}")
    if not report["suspected_ai_act_triggers"]:
        lines.append("- No biometric AI Act trigger found by this POC.")
    lines.extend(["", "## Required documents", ""])
    for doc in report["required_documents"]:
        lines.append(f"- {doc}")
    lines.extend(["", "## Classification context", "", AI_ACT_CONTEXT.strip(), ""])
    lines.extend(["", "## Evidence", ""])
    if not signals:
        lines.append("No biometric prefilter signals were found. Text-only emotional or sentiment analysis is outside this biometric POC unless it is tied to biometric data such as face, voice, video, or other biometric inputs.")
    for item in signals:
        lines.append(f"### `{item['file']}`")
        lines.append("")
        for match in item["matches"]:
            concept = match.get("ai_act_concept", "possible AI Act biometric signal")
            basis = match.get("basis", "source-code signal")
            lines.append(
                f"- Line {match['line']} `{match['term']}` ({concept}; {basis}): `{match['snippet']}`"
            )
        lines.append("")
    lines.extend(["## Review note", "", str(report["disclaimer"]), ""])
    return "\n".join(lines)


def render_required_actions(report: Dict[str, object]) -> str:
    lines = [
        "# Required AI Act follow-up actions",
        "",
        "This file was generated by the AI Act biometric-risk POC.",
        "",
        "## Immediate actions",
        "",
    ]
    for doc in report["required_documents"]:
        if report["signals"]:
            lines.append(f"- Create or update: {doc}.")
        else:
            lines.append(f"- {doc}")
    lines.extend(
        [
            "- Confirm whether any flagged processing identifies people, categorizes people, or infers emotion from biometric data.",
            "- Do not classify text-only sentiment or relationship analysis as biometric emotion recognition unless it uses biometric input.",
            "- Confirm whether the system is used in a prohibited, high-risk, limited-risk, or out-of-scope context.",
            "- Record the reviewer, date, decision, and rationale before merging related code.",
            "",
            "## Manual review warning",
            "",
            "This POC is intentionally conservative. A human compliance owner must decide whether the flagged code actually falls within the EU AI Act biometric provisions.",
            "",
        ]
    )
    return "\n".join(lines)


def build_planning_prompt(report: Dict[str, object]) -> str:
    signals_json = json.dumps(report["signals"], indent=2)
    return f"""\
You are OpenCode running in a repository. Do not edit files in this step.

We are building a compliance POC for EU AI Act biometric risk.

Use this classification context:

```text
{AI_ACT_CONTEXT.strip()}
```

The scanner found these possible biometric source-code signals:

```json
{signals_json}
```

Create a concise, decision-complete remediation plan for this repository.

The plan must:
- start by saying whether the evidence is likely in scope or likely out of scope for this biometric POC;
- identify whether the code may involve biometric identification, biometric categorisation, biometric verification, or emotion recognition on biometric data;
- explicitly explain if a finding is only text sentiment, relationship analysis, or product copy and therefore not biometric emotion recognition by itself;
- list the minimum code or documentation changes needed for a POC only when the evidence is plausibly in scope;
- avoid broad refactors and preserve existing application behavior where possible;
- include manual legal/compliance review steps and the exact open questions a reviewer must answer.

Use these examples to calibrate:
- In scope: `face_recognition.compare_faces(frame, known_faces)` because it compares face biometric data to templates.
- In scope: `emotionModel.predict(webcamFrame)` because emotion is inferred from image/video biometric input.
- In scope: `voiceStressClassifier(audioStream)` when used to infer mood or intent from voice characteristics.
- Needs careful review: `biometricLogin.verify(userId, fingerprint)` may be one-to-one verification and can be outside Annex III high-risk if the sole purpose is confirming the claimed identity.
- Out of scope for this biometric POC: `analyze WhatsApp messages for emotional dynamics` because the input is text, not biometric data.
- Out of scope for this biometric POC: strings such as `face the consequences`, `interface`, or `emotional intimacy`.

Return Markdown only.
"""


def build_implementation_prompt(plan_path: Path, report_path: Path) -> str:
    return f"""\
You are OpenCode running in this repository. Implement the remediation plan at:

{plan_path.as_posix()}

Use the risk evidence at:

{report_path.as_posix()}

Rules:
- Make the smallest useful POC changes.
- Do not commit, branch, push, or open a pull request.
- Do not delete or rename files.
- Do not run destructive commands.
- Preserve existing behavior unless the remediation plan explicitly requires a small guard, notice, or review workflow.
- If the report says no biometric prefilter signals were found, do not invent code changes; update only compliance notes if needed.
- Do not treat text-only sentiment analysis, relationship analysis, or product copy as biometric emotion recognition unless biometric input such as face, voice, video, audio, gait, fingerprint, iris, or retina is involved.
- If the repository shape is unclear, add explicit compliance TODOs or metadata instead of inventing a large framework-specific integration.

After editing files, print a short summary of changed files and why.
"""


def build_opencode_command(
    model: str,
    variant: Optional[str],
    prompt: str,
    dangerously_skip_permissions: bool = False,
    executable: str = "opencode",
) -> List[str]:
    command = [executable, "run", "--model", model]
    if variant:
        command.extend(["--variant", variant])
    if dangerously_skip_permissions:
        command.append("--dangerously-skip-permissions")
    command.append(prompt)
    return command


def resolve_opencode_executable() -> Optional[str]:
    found = shutil.which("opencode")
    if found:
        return found

    windows_home = Path.home() / ".opencode" / "bin" / "opencode.exe"
    if windows_home.exists():
        return str(windows_home)

    return None


def run_opencode(
    repo_path: Path,
    model: str,
    variant: Optional[str],
    prompt: str,
    dangerously_skip_permissions: bool = False,
) -> str:
    executable = resolve_opencode_executable()
    if executable is None:
        raise RuntimeError("opencode was not found in PATH or ~/.opencode/bin.")

    command = build_opencode_command(
        model,
        variant,
        prompt,
        dangerously_skip_permissions,
        executable=executable,
    )
    result = subprocess.run(
        command,
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "opencode failed with exit code "
            f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    output = result.stdout.strip()
    if result.stderr.strip():
        output += "\n\n## OpenCode stderr\n\n```text\n" + result.stderr.strip() + "\n```"
    return output.strip()


def mock_opencode_output(kind: str) -> str:
    if kind == "plan":
        return """\
# OpenCode remediation plan

- Confirm whether the flagged code performs biometric identification, biometric categorization, or emotion recognition.
- Add a human compliance review gate before production use.
- Add an AI system inventory entry and required AI Act documentation placeholders.
- Keep source changes minimal for this POC.
"""
    return """\
Mock implementation mode: no source files were edited. In a real run, OpenCode would implement the remediation plan here.
"""


def validate_runtime_requirements(args: argparse.Namespace) -> None:
    if not args.mock_opencode:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is required unless --mock-opencode is used.")
        if resolve_opencode_executable() is None:
            raise RuntimeError("opencode was not found in PATH or ~/.opencode/bin. Install and configure OpenCode first.")

    if args.create_pr and not args.dry_run:
        if not args.github_repo:
            raise RuntimeError("--github-repo owner/name is required with --create-pr.")
        if not github_token():
            raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required with --create-pr.")


def run_git(repo_path: Path, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def is_git_repo_root(path: Path) -> bool:
    return (path / ".git").exists()


def ensure_clean_worktree(repo_path: Path) -> None:
    if not is_git_repo_root(repo_path):
        raise RuntimeError(f"{repo_path} is not a git repository root.")
    result = run_git(repo_path, ["status", "--porcelain"], check=True)
    if result.stdout.strip():
        raise RuntimeError(
            "Target repository has existing changes. Start from a clean worktree before creating a PR."
        )


def parse_porcelain_status(output: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for raw_line in output.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path = raw_line[3:].strip()
        entries.append((status, path))
    return entries


def get_changed_files(repo_path: Path) -> List[str]:
    result = run_git(repo_path, ["status", "--porcelain", "--untracked-files=all"], check=True)
    changed: List[str] = []
    unsupported: List[str] = []

    for status, path in parse_porcelain_status(result.stdout):
        if " -> " in path:
            unsupported.append(f"{status} {path}")
            continue
        if "D" in status or "R" in status or "C" in status:
            unsupported.append(f"{status} {path}")
            continue
        if status == "??" or "A" in status or "M" in status:
            changed.append(path)
        else:
            unsupported.append(f"{status} {path}")

    if unsupported:
        raise RuntimeError(
            "Only added and modified files are supported for PR upload. Unsupported changes:\n"
            + "\n".join(unsupported)
        )
    return sorted(set(changed))


def list_generated_files(compliance_dir: Path, repo_path: Path) -> List[str]:
    if not compliance_dir.exists():
        return []
    files = []
    for path in sorted(compliance_dir.rglob("*")):
        if path.is_file():
            files.append(normalize_rel_path(path.relative_to(repo_path)))
    return files


def build_pr_body(
    report: Dict[str, object],
    changed_files: Sequence[str],
    plan_path: str,
) -> str:
    signal_files = [str(item["file"]) for item in report["signals"]]
    lines = [
        "## Summary",
        "",
        "This draft PR was generated by the AI Act biometric-risk POC.",
        "",
        "## Detected biometric trigger",
        "",
    ]
    if signal_files:
        lines.append("Possible biometric AI Act signals were found in:")
        lines.append("")
        for file_path in signal_files:
            lines.append(f"- `{file_path}`")
    else:
        lines.append("No biometric prefilter signals were found, but compliance artifacts were generated for review.")

    lines.extend(
        [
            "",
            "## Generated artifacts and code changes",
            "",
            f"- OpenCode remediation plan: `{plan_path}`",
        ]
    )
    for file_path in changed_files:
        lines.append(f"- `{file_path}`")

    lines.extend(
        [
            "",
            "## Manual review warning",
            "",
            "This is a POC. The output is not a legal conclusion. A qualified reviewer must confirm AI Act classification, required documentation, and whether the changes are appropriate before merge.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_compliance_files(
    repo_path: Path,
    report: Dict[str, object],
    plan_markdown: Optional[str] = None,
    pr_body: Optional[str] = None,
) -> Dict[str, Path]:
    compliance_dir = repo_path / ".compliance" / "ai-act"
    paths = {
        "json_report": compliance_dir / "biometric-risk-report.json",
        "markdown_report": compliance_dir / "biometric-risk-report.md",
        "required_actions": compliance_dir / "required-actions.md",
        "plan": compliance_dir / "opencode-remediation-plan.md",
        "pr_body": compliance_dir / "pr-body.md",
    }
    write_text(paths["json_report"], json.dumps(report, indent=2) + "\n")
    write_text(paths["markdown_report"], render_report_markdown(report))
    write_text(paths["required_actions"], render_required_actions(report))
    if plan_markdown is not None:
        write_text(paths["plan"], plan_markdown.rstrip() + "\n")
    if pr_body is not None:
        write_text(paths["pr_body"], pr_body.rstrip() + "\n")
    return paths


def github_token() -> Optional[str]:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def github_request(
    method: str,
    path: str,
    token: str,
    payload: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{GITHUB_API_URL}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "nano-analyzer-ai-act-poc",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {body}") from exc


def make_github_blob_payload(content: bytes) -> Dict[str, str]:
    return {
        "content": base64.b64encode(content).decode("ascii"),
        "encoding": "base64",
    }


def make_tree_entry(path: str, blob_sha: str) -> Dict[str, str]:
    return {
        "path": path.replace("\\", "/"),
        "mode": "100644",
        "type": "blob",
        "sha": blob_sha,
    }


def make_pr_payload(title: str, head: str, base: str, body: str) -> Dict[str, object]:
    return {
        "title": title,
        "head": head,
        "base": base,
        "body": body,
        "draft": True,
    }


def validate_repo_full_name(repo_full_name: str) -> None:
    if not re.match(r"^[^/\s]+/[^/\s]+$", repo_full_name):
        raise RuntimeError("--github-repo must be in owner/name format.")


def create_draft_pr_from_changes(
    repo_path: Path,
    repo_full_name: str,
    base_branch: str,
    branch_name: str,
    changed_files: Sequence[str],
    pr_body: str,
    token: str,
) -> Dict[str, object]:
    validate_repo_full_name(repo_full_name)
    if not changed_files:
        raise RuntimeError("No changed files to upload.")

    owner_repo_path = f"/repos/{repo_full_name}"
    base_ref = github_request("GET", f"{owner_repo_path}/git/ref/heads/{base_branch}", token)
    base_sha = str(base_ref["object"]["sha"])
    base_commit = github_request("GET", f"{owner_repo_path}/git/commits/{base_sha}", token)
    base_tree_sha = str(base_commit["tree"]["sha"])

    github_request(
        "POST",
        f"{owner_repo_path}/git/refs",
        token,
        {"ref": f"refs/heads/{branch_name}", "sha": base_sha},
    )

    tree = []
    for rel_path in changed_files:
        file_path = repo_path / rel_path
        if not file_path.is_file():
            raise RuntimeError(f"Changed path is not a file: {rel_path}")
        blob = github_request(
            "POST",
            f"{owner_repo_path}/git/blobs",
            token,
            make_github_blob_payload(file_path.read_bytes()),
        )
        tree.append(make_tree_entry(rel_path, str(blob["sha"])))

    new_tree = github_request(
        "POST",
        f"{owner_repo_path}/git/trees",
        token,
        {"base_tree": base_tree_sha, "tree": tree},
    )
    commit = github_request(
        "POST",
        f"{owner_repo_path}/git/commits",
        token,
        {
            "message": "Add AI Act biometric-risk POC output",
            "tree": str(new_tree["sha"]),
            "parents": [base_sha],
        },
    )
    github_request(
        "PATCH",
        f"{owner_repo_path}/git/refs/heads/{branch_name}",
        token,
        {"sha": str(commit["sha"])},
    )
    return github_request(
        "POST",
        f"{owner_repo_path}/pulls",
        token,
        make_pr_payload(
            "AI Act biometric risk review required",
            branch_name,
            base_branch,
            pr_body,
        ),
    )


def run_poc(args: argparse.Namespace) -> int:
    repo_path = Path(args.path).resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        raise RuntimeError(f"Target repo path does not exist or is not a directory: {repo_path}")

    validate_runtime_requirements(args)

    if args.create_pr and not args.dry_run:
        ensure_clean_worktree(repo_path)

    print(f"Scanning biometric AI Act signals in {repo_path}")
    signals = find_biometric_signals(repo_path)
    report = build_report(repo_path, signals)
    paths = generate_compliance_files(repo_path, report)
    print(f"Generated risk report: {paths['markdown_report']}")

    planning_prompt = build_planning_prompt(report)
    if args.mock_opencode:
        plan_markdown = mock_opencode_output("plan")
    else:
        print("Calling OpenCode planning pass...")
        plan_markdown = run_opencode(repo_path, args.model, args.variant, planning_prompt)
    paths = generate_compliance_files(repo_path, report, plan_markdown=plan_markdown)
    print(f"Generated OpenCode plan: {paths['plan']}")

    implementation_prompt = build_implementation_prompt(
        paths["plan"].relative_to(repo_path),
        paths["markdown_report"].relative_to(repo_path),
    )
    if args.mock_opencode:
        implementation_output = mock_opencode_output("implementation")
    else:
        print("Calling OpenCode implementation pass...")
        implementation_output = run_opencode(
            repo_path,
            args.model,
            args.variant,
            implementation_prompt,
            dangerously_skip_permissions=True,
        )
    if implementation_output:
        print(implementation_output.splitlines()[0])

    if is_git_repo_root(repo_path):
        changed_files = get_changed_files(repo_path)
    else:
        changed_files = list_generated_files(repo_path / ".compliance" / "ai-act", repo_path)

    plan_rel_path = normalize_rel_path(paths["plan"].relative_to(repo_path))
    pr_body = build_pr_body(report, changed_files, plan_rel_path)
    paths = generate_compliance_files(
        repo_path,
        report,
        plan_markdown=plan_markdown,
        pr_body=pr_body,
    )

    if is_git_repo_root(repo_path):
        changed_files = get_changed_files(repo_path)
    else:
        changed_files = list_generated_files(repo_path / ".compliance" / "ai-act", repo_path)

    print("Changed files:")
    for file_path in changed_files:
        print(f"  - {file_path}")

    if args.dry_run or not args.create_pr:
        print("Dry run or --create-pr not set; skipping GitHub PR creation.")
        return 0

    branch_name = f"{args.branch_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    token = github_token()
    if token is None:
        raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required.")

    print(f"Creating draft PR in {args.github_repo} from {branch_name}...")
    pr = create_draft_pr_from_changes(
        repo_path=repo_path,
        repo_full_name=args.github_repo,
        base_branch=args.base_branch,
        branch_name=branch_name,
        changed_files=changed_files,
        pr_body=pr_body,
        token=token,
    )
    print(f"Draft PR: {pr.get('html_url')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_act_poc",
        description="OpenCode-driven EU AI Act biometric-risk POC.",
    )
    parser.add_argument("path", help="Target repository root to scan and edit")
    parser.add_argument("--github-repo", default=None, help="GitHub repository in owner/name format")
    parser.add_argument("--base-branch", default=DEFAULT_BASE_BRANCH, help="Base branch for the generated PR")
    parser.add_argument("--branch-prefix", default=DEFAULT_BRANCH_PREFIX, help="Generated PR branch prefix")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenCode model (default: {DEFAULT_MODEL})")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help=f"OpenCode model variant/reasoning effort (default: {DEFAULT_VARIANT})")
    parser.add_argument("--create-pr", action="store_true", help="Create a draft GitHub PR")
    parser.add_argument("--dry-run", action="store_true", help="Generate files locally but skip GitHub PR creation")
    parser.add_argument("--mock-opencode", action="store_true", help="Skip OpenCode calls and use deterministic mock output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_poc(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
