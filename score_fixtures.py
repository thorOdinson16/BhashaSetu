"""Score all fixtures against ground truth using the 7-check JTBD rubric."""

import json
import re
import asyncio
from pathlib import Path
from pipeline import Pipeline

FIXTURES_DIR = Path("fixtures")
GROUND_TRUTH = json.loads((FIXTURES_DIR / "ground_truth.json").read_text())

p = Pipeline()


def fuzzy_match(gt_val: str | None, ent_val: str | None, field: str) -> bool:
    """Fuzzy match entity values accounting for format differences."""
    if gt_val is None:
        return ent_val is None or str(ent_val).strip() in ("", "None", "—")

    ent_val = str(ent_val).strip()
    gt_val = str(gt_val).strip()

    if field == "phone":
        # Normalize: strip +91, spaces, dashes
        gt = re.sub(r"[\s\-\(\)\+91]", "", gt_val)
        ent = re.sub(r"[\s\-\(\)\+91]", "", ent_val)
        return gt in ent or ent in gt

    if field == "time":
        # Normalize time formats
        gt = gt_val.lower().replace(".", ":").replace(" ", "")
        ent = ent_val.lower().replace(".", ":").replace(" ", "")
        return gt == ent or gt in ent or ent in gt

    if field == "address":
        # Substring match, case insensitive
        return gt_val.lower() in ent_val.lower() or any(
            word.lower() in ent_val.lower()
            for word in gt_val.split()
            if len(word) > 2
        )

    # Default: contains match
    return gt_val.lower() in ent_val.lower()


async def score_all():
    total_score = 0
    total_possible = 28  # 4 scenarios × 7 checks

    for scenario_name, scenario in GROUND_TRUTH.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} — {scenario['description']}")
        print(f"{'='*60}")

        # Collect all entities from all turns in this scenario
        all_entities = {}

        for turn in scenario["turns"]:
            filename = turn["file"]
            lang = turn["lang"]
            filepath = FIXTURES_DIR / filename
            if not filepath.exists():
                print(f"  ❌ File not found: {filename}")
                continue

            raw = filepath.read_bytes()
            target = "hi-IN" if lang == "kn-IN" else "kn-IN"

            collected = []

            async def on_entities(entities, transcript, translated):
                collected.append(entities)

            result = await p.process_utterance(raw, lang, target, on_entities=on_entities)

            for _ in range(30):
                if collected:
                    break
                await asyncio.sleep(0.5)

            entities = collected[0] if collected else {}
            timing = result.get("timing", {})

            # Merge entities from this turn
            for field in ["date", "time", "name", "phone", "address", "amount"]:
                val = entities.get(field)
                if val and str(val).strip() not in ("", "None", "null"):
                    all_entities[field] = str(val).strip()

            # Track correction across turns
            if entities.get("correction_detected"):
                all_entities["correction_detected"] = True
                all_entities["correction_old_value"] = entities.get("correction_old_value")
                all_entities["correction_new_value"] = entities.get("correction_new_value")

            print(f"  {filename} ({turn['speaker']}): "
                  f"date={entities.get('date')} time={entities.get('time')} "
                  f"addr={entities.get('address')} phone={entities.get('phone')} "
                  f"amt={entities.get('amount')} corr={entities.get('correction_detected')} "
                  f"| {timing.get('total','?')}s")

        # Score against ground truth — use union of expected values from all turns
        checks = {}
        expected = {}
        expected_correction = False
        for turn in scenario["turns"]:
            exp = turn["expected"]
            for field in ["date", "time", "name", "phone", "address", "amount"]:
                val = exp.get(field)
                if val is not None and str(val).strip() not in ("", "None"):
                    expected[field] = val
            if exp.get("correction_detected"):
                expected_correction = True

        for field in ["date", "time", "name", "phone", "address", "amount"]:
            gt_val = expected.get(field)
            ent_val = all_entities.get(field)
            checks[field] = fuzzy_match(gt_val, ent_val, field)

        checks["correction"] = expected_correction == bool(all_entities.get("correction_detected"))

        sc_score = sum(1 for v in checks.values() if v)
        total_score += sc_score

        status = "✅" if sc_score >= 6 else "⚠️" if sc_score >= 4 else "❌"
        failed = [k for k, v in checks.items() if not v]
        print(f"  {status} SCENARIO SCORE: {sc_score}/7 | Failed: {failed}")

    pct = round(total_score / total_possible * 100) if total_possible else 0
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_score}/{total_possible} ({pct}%)")
    if pct >= 85:
        print("🎯 JTBD L5 threshold reached! (85%+)")
    elif pct >= 70:
        print("📊 JTBD L4 threshold reached! (70%+)")
    else:
        print("⚠️ Below L4 threshold")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(score_all())
