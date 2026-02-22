"""Consolidate study artifacts from multiple outputs* folders into outputs_master."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida historico e banco de parametros em uma pasta unica para estudo continuo."
    )
    parser.add_argument(
        "--target",
        default="outputs_master",
        help="Pasta de destino consolidada (default: outputs_master).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Pastas origem. Se vazio, detecta automaticamente outputs* exceto o target.",
    )
    parser.add_argument(
        "--copy-manifests",
        action="store_true",
        help="Copia run_manifest_*.json para a pasta alvo (sem sobrescrever).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = consolidate_outputs(
        target=(Path.cwd() / args.target).resolve(),
        sources=[Path(s).resolve() for s in args.sources] if args.sources else None,
        copy_manifests=bool(args.copy_manifests),
    )
    if int(report.get("sources_count", 0)) == 0:
        print("[INFO] Nenhuma pasta origem encontrada para consolidar.")
        return
    report_file = Path(str(report["report_file"]))

    print("[DONE] Consolidacao finalizada.")
    print(f"[INFO] target: {report['target']}")
    print(f"[INFO] merged best_history files: {report['merged_best_history_files']}")
    print(f"[INFO] merged strategy_history files: {report['merged_strategy_history_files']}")
    print(f"[INFO] merged params_bank files: {report['merged_params_bank_files']}")
    if args.copy_manifests:
        print(f"[INFO] copied manifests: {report['copied_manifests']}")
    print(f"[INFO] report: {report_file}")


def consolidate_outputs(
    target: Path,
    sources: list[Path] | None = None,
    copy_manifests: bool = False,
) -> dict[str, Any]:
    """Merge multiple outputs* folders into one canonical outputs folder."""
    root = Path.cwd()
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    resolved_sources = _resolve_sources(root=root, target=target, sources=sources)
    if not resolved_sources:
        report = {
            "target": str(target),
            "sources": [],
            "sources_count": 0,
            "merged_best_history_files": 0,
            "merged_strategy_history_files": 0,
            "merged_params_bank_files": 0,
            "copied_manifests": 0,
        }
        report_file = target / "_study_setup_report.json"
        report["report_file"] = str(report_file)
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    merged_histories = 0
    merged_strategy_histories = 0
    merged_banks = 0
    copied_manifests = 0

    for src in resolved_sources:
        for history_file in src.glob("**/best_history_*.csv"):
            rel = history_file.relative_to(src)
            dst = target / rel
            _merge_best_history(history_file, dst)
            merged_histories += 1

        for strategy_history_file in src.glob("**/strategy_history_*.csv"):
            rel = strategy_history_file.relative_to(src)
            dst = target / rel
            _merge_strategy_history(strategy_history_file, dst)
            merged_strategy_histories += 1

        for bank_file in src.glob("**/params_bank_*.jsonl"):
            rel = bank_file.relative_to(src)
            dst = target / rel
            _merge_params_bank(bank_file, dst)
            merged_banks += 1

        if copy_manifests:
            for manifest in src.glob("**/run_manifest_*.json"):
                rel = manifest.relative_to(src)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(manifest.read_bytes())
                    copied_manifests += 1

    report = {
        "target": str(target),
        "sources": [str(x) for x in resolved_sources],
        "sources_count": len(resolved_sources),
        "merged_best_history_files": merged_histories,
        "merged_strategy_history_files": merged_strategy_histories,
        "merged_params_bank_files": merged_banks,
        "copied_manifests": copied_manifests,
    }
    report_file = target / "_study_setup_report.json"
    report["report_file"] = str(report_file)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _resolve_sources(root: Path, target: Path, sources: list[Path] | None) -> list[Path]:
    if sources:
        candidates = [Path(s).resolve() for s in sources]
    else:
        candidates = [
            p.resolve()
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith("outputs") and p.resolve() != target
        ]
    return [p for p in candidates if p.exists() and p.is_dir() and p.resolve() != target]


def _merge_best_history(src: Path, dst: Path) -> None:
    src_df = _read_csv_safe(src)
    if src_df.empty:
        return
    dst_df = _read_csv_safe(dst) if dst.exists() else pd.DataFrame()
    out = pd.concat([dst_df, src_df], ignore_index=True)
    if out.empty:
        return

    if "created_at_utc" in out.columns:
        out["created_at_utc"] = pd.to_datetime(out["created_at_utc"], errors="coerce", utc=True, format="mixed")

    dedupe_cols = [c for c in ["run_tag", "timeframe"] if c in out.columns]
    if dedupe_cols:
        sort_cols = [c for c in ["created_at_utc"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=False, na_position="last")
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")

    if "created_at_utc" in out.columns:
        out = out.sort_values("created_at_utc", ascending=False, na_position="last")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)


def _merge_strategy_history(src: Path, dst: Path) -> None:
    src_df = _read_csv_safe(src)
    if src_df.empty:
        return
    dst_df = _read_csv_safe(dst) if dst.exists() else pd.DataFrame()
    out = pd.concat([dst_df, src_df], ignore_index=True)
    if out.empty:
        return
    if "created_at_utc" in out.columns:
        out["created_at_utc"] = pd.to_datetime(out["created_at_utc"], errors="coerce", utc=True, format="mixed")
    dedupe_cols = [c for c in ["run_tag", "timeframe", "strategy"] if c in out.columns]
    if dedupe_cols:
        sort_cols = [c for c in ["created_at_utc", "rank_in_run"] if c in out.columns]
        if sort_cols:
            ascending = [False] + [True] * (len(sort_cols) - 1)
            out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    if "created_at_utc" in out.columns:
        sort_cols = [c for c in ["created_at_utc", "rank_in_run"] if c in out.columns]
        ascending = [False] + [True] * (len(sort_cols) - 1)
        out = out.sort_values(sort_cols, ascending=ascending, na_position="last")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)


def _merge_params_bank(src: Path, dst: Path) -> None:
    best_by_key: dict[str, dict[str, Any]] = {}
    for file in [dst, src]:
        if not file.exists():
            continue
        for payload in _iter_jsonl(file):
            params = payload.get("params")
            if not isinstance(params, dict):
                continue
            key = json.dumps(params, sort_keys=True, ensure_ascii=False)
            score = _to_float(payload.get("test_score", 0.0))
            prev = best_by_key.get(key)
            if prev is None or score > _to_float(prev.get("test_score", 0.0)):
                best_by_key[key] = payload

    if not best_by_key:
        return

    ranked = sorted(best_by_key.values(), key=lambda p: _to_float(p.get("test_score", 0.0)), reverse=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for item in ranked:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    out.append(payload)
    except OSError:
        return []
    return out


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
