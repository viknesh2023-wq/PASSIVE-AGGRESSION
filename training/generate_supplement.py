"""
PassiveAggression Detector 2.0 — Supplementary Dataset Generator
training/generate_supplement.py

Generates additional records to bring the merged dataset closer to 10,000.
Focuses on expanding ambiguous and non_passive_aggressive examples
with more diverse templates.

Reads existing data/synthetic/pa_synthetic_merged.csv to avoid duplicates.
Outputs: data/synthetic/supplement_SYNPA_*.csv
"""

from __future__ import annotations
import csv
import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("supplement")

SEED = 9999

GENERATION_SOURCE = "synthetic"
REVIEW_STATUS     = "synthetic_unreviewed"
GENERATOR_MODEL   = "controlled_template_variation_v2.2_supplement"


@dataclass
class SynRecord:
    id:                  str
    text:                str
    primary_label:       str
    threat_label:        str
    post_text:           Optional[str]
    conversation_id:     str
    turn_index:          int
    parent_comment_id:   Optional[str]
    parent_comment:      Optional[str]
    thread_context:      Optional[str]
    target_message_id:   str
    tactic_labels:       str
    sarcasm_label:       str
    sentiment_label:     str
    generation_source:   str
    generator_model:     str
    generation_batch:    str
    review_status:       str
    quality_flags:       str


# ── Expanded topic slots (more variation) ────────────────────────────────────────
TOPICS2 = [
    ("instagram", "photo share",       "the selfie",          "the photoshoot"),
    ("instagram", "reel post",         "the reel",            "the edit"),
    ("instagram", "dm conversation",   "the DM",              "the reply"),
    ("youtube",   "comment reply",     "the comment",         "the response"),
    ("youtube",   "shorts video",      "the Short",           "the video"),
    ("reddit",    "r/relationship",    "the post",            "the situation"),
    ("reddit",    "r/gaming",          "the build",           "the game"),
    ("twitter",   "subtweet moment",   "that tweet",          "the subtweet"),
    ("twitter",   "poll vote",         "the poll",            "the result"),
    ("tiktok",    "sound use",         "the audio",           "the TikTok"),
    ("tiktok",    "stitched video",    "the stitch",          "the original"),
    ("discord",   "vc session",        "the VC",              "the session"),
    ("discord",   "bot command",       "the bot",             "the server"),
    ("groupchat", "family group",      "the message",         "the conversation"),
    ("groupchat", "school project",    "the assignment",      "the group work"),
    ("groupchat", "work slack",        "the message",         "the thread"),
]

# ── Expanded ambiguous templates ────────────────────────────────────────────────
AMBIGUOUS_EXPANDED = [
    # Social media specific terse replies
    "👀",
    "✨",
    "Noted 👀",
    "Interesting timing.",
    "Wow okay.",
    "That's a choice.",
    "All righty then.",
    "Love that for you.",
    "Sure, okay.",
    "Checks out.",
    "Makes sense.",
    "That tracks.",
    "Absolutely wild.",
    "As one does.",
    "Bold.",
    "Remarkable.",
    "Unhinged behavior.",
    "Not surprising.",
    "Say less.",
    "Say more.",
    "Oof.",
    "Yikes.",
    "Okay bestie.",
    "You do that.",
    "Big energy.",
    "Chaotic.",
    "Choices were made.",
    # Slightly more words
    "Didn't expect that from {spec} but sure.",
    "Whatever happens with {task} happens.",
    "I'll just be over here thinking about {spec}.",
    "Not gonna comment on {task}.",
    "Logging on just to see {spec} posted. A day.",
    "I have many thoughts about {task} but zero words.",
    "The fact that {spec} exists is a lot.",
    "Processing {task} in real time.",
    "Something about {spec} is giving me pause.",
    "I'll wait and see what happens with {task}.",
    "The {spec} energy today is something.",
    "Choosing not to speak on {task}.",
    "Sitting with the feelings about {spec} quietly.",
    "The audacity of {task}. The absolute audacity.",
    "I'm not ready to talk about {spec} yet.",
    "Whatever {task} is, it's certainly that.",
    "The {spec} feedback is real and raw.",
    "Not commenting on {task}. Not today.",
    "{spec} dropped and I have three thoughts and none of them are calm.",
    "Choosing peace regarding {task}.",
    "The {spec} reveal has me speechless.",
    "I've seen {task} and I have opinions I'm keeping to myself.",
    "Manifesting a different outcome for {spec}.",
    "Praying for {task} at this point.",
    "The nerve of {spec}. Genuinely.",
    "Screaming internally about {task}.",
    "I'm not saying anything about {spec} that my lawyer hasn't approved.",
    "Unbothered. Moisturized. Minding my business about {task}.",
    "Did we all just witness {spec} happen in real time?",
    "The timeline is a lot and {task} is not helping.",
    "Woke up to {spec} existing. Having a normal one.",
    "Living, laughing, logging off after seeing {task}.",
    "The {spec} lore thickens.",
    "Oh so {task} is happening. Cool. Fine. Whatever.",
    "Making note of {spec} for future reference.",
    "Archiving the {task} discourse for historians.",
    "Not surprised by {spec}. Just… not surprised.",
    "The {task} decision was made and we are all just witnessing.",
    "Filing {spec} under 'things that happened'.",
    "{task} having a moment. We are all just watching.",
    "Observing {spec} from a safe distance.",
    "Thought about {task}. Moving on.",
    "No comment on {spec}. Plenty of feelings.",
    "I'll be processing {task} for the rest of the week.",
    "Neutral face about {spec} but internally screaming.",
    "Commenting to say I have nothing to say about {task}.",
    "The {spec} conversation is above my pay grade.",
    "Replying to say I'm actively choosing not to engage with {task}.",
]

# ── More non-PA templates ────────────────────────────────────────────────────────
NON_PA_EXPANDED = [
    # Warm social engagement
    ("none", "non_sarcastic", "positive", [
        "This {spec} is really bringing the community together 💯",
        "Can't stop thinking about how good {task} turned out.",
        "Really proud of everyone who worked on {spec}.",
        "The quality of {task} has improved so much lately.",
        "Just came here to say {spec} made my day.",
        "The effort that went into {task} is so visible. Well done.",
        "Sharing {spec} with everyone I know — it's that good.",
        "The way {task} came together is genuinely inspiring.",
        "{spec} reminder that good things do happen 🙌",
        "The passion behind {task} shows in every detail.",
        "Dropped everything to see {spec} and it was worth it.",
        "The depth of {task} caught me completely off guard.",
        "People sleeping on {spec} really need to pay attention.",
        "The clarity in {task} is something else — so well executed.",
        "Coming back to appreciate {spec} for the hundredth time.",
        "Nobody told me {task} would be this good. My jaw dropped.",
        "Sharing {spec} feels like giving a gift.",
        "The response to {task} has been overwhelmingly positive and deserved.",
        "{spec} is proof that real effort resonates.",
        "So glad I took the time to actually engage with {task}.",
    ]),
    # Direct constructive disagreement
    ("none", "non_sarcastic", "negative", [
        "Respectfully disagree with the framing in {spec}.",
        "I think {task} needs a rethink on the core premise.",
        "The conclusion in {spec} doesn't hold up under scrutiny.",
        "I'd challenge the assumption in {task} — here's why.",
        "The evidence for {spec} seems cherry-picked.",
        "{task} glosses over some important nuances.",
        "I can see the intent with {spec} but the execution misses.",
        "The data in {task} doesn't support the claim being made.",
        "Pointing out that {spec} contradicts what was said last week.",
        "{task} would have landed better with a different structure.",
        "The critique of {spec} is fair but the alternative isn't clear.",
        "I don't think {task} is ready for public release.",
        "The assumptions in {spec} haven't been tested rigorously.",
        "{task} is a good start but needs significant work still.",
        "Calling out the inaccuracy in {spec} — this needs correction.",
        "The framing of {task} is misleading even if unintentionally.",
        "{spec} presents a false dichotomy that should be addressed.",
        "The tone in {task} is off for the context it's being shared in.",
        "I'm questioning the methodology behind {spec}.",
        "{task} deserves better sourcing before being circulated.",
    ]),
    # Neutral informational
    ("none", "non_sarcastic", "neutral", [
        "Dropping the link to {spec} here as requested.",
        "{task} has been updated. Check the pinned post for details.",
        "Just a reminder that {spec} closes for submissions tomorrow.",
        "New version of {task} is in the shared folder.",
        "Quick note: {spec} was moved to the other channel.",
        "For those who missed it — {task} recap is in the thread.",
        "Confirming that {spec} has been reviewed and approved.",
        "{task} is scheduled for next Tuesday at 7 PM UTC.",
        "The notes from {spec} discussion are now in the doc.",
        "Replying to confirm I received {task} — thanks!",
        "Just posted {spec} in the original thread for reference.",
        "The final version of {task} is ready for download.",
        "{spec} template has been updated per the feedback.",
        "Heads up that {task} is running about 20 minutes behind.",
        "All materials for {spec} are in the pinned folder.",
        "FYI: {task} will be archived at end of month.",
        "If anyone needs access to {spec}, message me directly.",
        "Sharing {task} summary here for the group.",
        "Quick heads up about the {spec} schedule change.",
        "{task} has been assigned. Check your notifications.",
    ]),
    # Friendly banter (sarcastic but benign)
    ("sarcasm", "sarcastic", "positive", [
        "Oh great, {spec} is trending. My entire week is derailed 😂",
        "The absolute timing of {task} dropping on a Friday 💀",
        "Thanks for making {spec} so compelling. Now I can't sleep. Rude.",
        "I cannot believe I stayed up for {task}. Completely worth it.",
        "The {spec} algorithm really said 'let's ruin them' and succeeded.",
        "My plans this weekend: none, because {task} exists now.",
        "The way {spec} just ended my productivity permanently. Love it.",
        "Oh no {task} is actually good. There goes my social life.",
        "The {spec} fandom really said 'rest? never'. And I agree now.",
        "Convinced {task} was designed specifically to destroy my sleep schedule.",
        "I was doing so well until {spec} dropped. RIP to my to-do list.",
        "The audacity of {task} being this good on a Monday morning.",
        "Three hours into {spec} and I have no regrets and also no time.",
        "Watching {task} when I should be sleeping. Character development.",
        "Tell me why {spec} is ruining my week and I'm grateful for it.",
        "The creators of {task} really said 'consequences? never heard of them' 😭",
        "{spec} has no right being this absorbing. None.",
        "Starting {task} 'just for five minutes' at 11pm. A tale as old as time.",
        "The {spec} update dropped and I am simply not the same person anymore.",
        "Somebody stop {task} from taking over my entire personality please.",
    ]),
]


def fill2(template: str, topic: tuple) -> str:
    platform, domain, spec, task = topic
    return (template
            .replace("{spec}", spec)
            .replace("{task}", task)
            .replace("{topic}", domain)
            .replace("{platform}", platform))


def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_existing_hashes(merged_path: Path) -> tuple[set, set]:
    seen_hashes = set()
    seen_norms  = set()
    if not merged_path.exists():
        return seen_hashes, seen_norms
    with open(merged_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "")
            if text:
                md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
                seen_hashes.add(md5)
                norm = normalize_for_dedup(text)
                if len(norm) > 10:
                    seen_norms.add(norm)
    logger.info("Loaded %d existing hashes from %s", len(seen_hashes), merged_path)
    return seen_hashes, seen_norms


def main():
    data_dir    = Path("data/synthetic")
    reports_dir = Path("reports")
    merged_path = data_dir / "pa_synthetic_merged.csv"

    # Load existing hashes
    seen_hashes, seen_norms = load_existing_hashes(merged_path)

    # Count existing records per label
    existing_counts: dict[str, int] = {}
    existing_total = 0
    if merged_path.exists():
        with open(merged_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("primary_label", "")
                existing_counts[lbl] = existing_counts.get(lbl, 0) + 1
                existing_total += 1

    logger.info("Existing: %d records — %s", existing_total, existing_counts)

    # Target: bring total to at least 9,000
    TARGET_TOTAL = 9_000
    needed = max(0, TARGET_TOTAL - existing_total)
    if needed == 0:
        logger.info("Already at target. No supplement needed.")
        return

    # Determine per-label supplement needs
    # Target equal thirds: 3000 each
    per_label_target = TARGET_TOTAL // 3
    supp_pa   = max(0, per_label_target - existing_counts.get("passive_aggressive", 0))
    supp_npa  = max(0, per_label_target - existing_counts.get("non_passive_aggressive", 0))
    supp_amb  = max(0, TARGET_TOTAL - existing_total - supp_pa - supp_npa)
    logger.info("Supplement targets: PA=%d, Non-PA=%d, Ambiguous=%d", supp_pa, supp_npa, supp_amb)

    rng = random.Random(SEED)

    # Find next available ID
    next_id = existing_total + 1
    if merged_path.exists():
        with open(merged_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_id = 0
            for row in reader:
                rid = row.get("id", "SYNPA-00000")
                try:
                    num = int(rid.replace("SYNPA-", ""))
                    max_id = max(max_id, num)
                except ValueError:
                    pass
        next_id = max_id + 1
    logger.info("Starting IDs from SYNPA-%05d", next_id)

    current_idx = next_id
    supplement_records: list[dict] = []

    import unicodedata

    def make_rec(text, primary_label, threat_label, tactic, sarcasm, sentiment, batch_name):
        nonlocal current_idx
        rid = f"SYNPA-{current_idx:05d}"
        current_idx += 1
        return {
            "id": rid,
            "text": text.strip(),
            "primary_label": primary_label,
            "threat_label": threat_label,
            "post_text": "",
            "conversation_id": f"standalone_{current_idx:05d}",
            "turn_index": 0,
            "parent_comment_id": "",
            "parent_comment": "",
            "thread_context": "",
            "target_message_id": rid,
            "tactic_labels": json.dumps([tactic] if tactic else []),
            "sarcasm_label": sarcasm,
            "sentiment_label": sentiment,
            "generation_source": GENERATION_SOURCE,
            "generator_model": GENERATOR_MODEL,
            "generation_batch": batch_name,
            "review_status": REVIEW_STATUS,
            "quality_flags": "supplement",
        }

    def is_fresh(text: str) -> bool:
        if not text or len(text.strip()) < 3 or len(text) > 700:
            return False
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if md5 in seen_hashes:
            return False
        norm = normalize_for_dedup(text)
        if len(norm) > 10 and norm in seen_norms:
            return False
        # English check
        for ch in text:
            if ch.isalpha():
                name = unicodedata.name(ch, "")
                if any(name.startswith(s) for s in (
                    "CYRILLIC", "ARABIC", "DEVANAGARI", "TAMIL", "CJK", "HANGUL",
                    "HIRAGANA", "KATAKANA",
                )):
                    return False
        seen_hashes.add(md5)
        if len(norm) > 10:
            seen_norms.add(norm)
        return True

    # Generate ambiguous supplement (most needed)
    amb_generated = 0
    attempts = 0
    max_attempts = supp_amb * 30
    while amb_generated < supp_amb and attempts < max_attempts:
        attempts += 1
        topic = rng.choice(TOPICS2)
        
        # Use different variant pools to increase diversity
        pool_choice = rng.random()
        if pool_choice < 0.6:
            raw = rng.choice(AMBIGUOUS_EXPANDED)
        else:
            # Generate simple terse contextual replies
            terse_options = [
                "Okay.",
                "Sure.",
                "Fine.",
                "Interesting.",
                "Noted.",
                "K.",
                "Right.",
                "Cool.",
                "Got it.",
                "Will see.",
                "As expected.",
                "That's something.",
                "Hm.",
                "Welp.",
                "Sigh.",
            ]
            raw = rng.choice(terse_options)
        
        # Add topic-specific variation to reduce duplicates
        if "{spec}" in raw or "{task}" in raw:
            text = fill2(raw, topic)
        else:
            # Append topic context to bare terse replies
            if rng.random() < 0.5:
                context_suffix = rng.choice([
                    f" about {topic[2]}",
                    f" re: {topic[3]}",
                    f" on {topic[2]}",
                    "",
                    "",
                ])
                text = raw.rstrip(".!?") + context_suffix
                if not text.endswith((".", "!", "?", "👀", "🤔")):
                    text += rng.choice([".", "...", "!"])
            else:
                text = raw

        # Randomly vary capitalization/punctuation for diversity
        if rng.random() < 0.3 and len(text) > 3:
            suffixes = ["", " lol", " honestly", " idk", " fr", " ngl", " tbh", " smh"]
            text = text.rstrip(".!?") + rng.choice(suffixes)

        if is_fresh(text):
            threat = "no_threat"
            rec = make_rec(text, "ambiguous", threat, "", "non_sarcastic", "neutral", "supplement_amb")
            supplement_records.append(rec)
            amb_generated += 1

    logger.info("Generated %d ambiguous supplement records (%d attempts)", amb_generated, attempts)

    # Generate non-PA supplement
    npa_generated = 0
    attempts = 0
    max_attempts = supp_npa * 20
    while npa_generated < supp_npa and attempts < max_attempts:
        attempts += 1
        topic = rng.choice(TOPICS2)
        entry = rng.choice(NON_PA_EXPANDED)
        tactic_str, sarcasm, sentiment, templates = entry
        raw = rng.choice(templates)
        text = fill2(raw, topic)

        if is_fresh(text):
            threat = "no_threat"
            rec = make_rec(text, "non_passive_aggressive", threat, tactic_str, sarcasm, sentiment, "supplement_npa")
            supplement_records.append(rec)
            npa_generated += 1

    logger.info("Generated %d non-PA supplement records (%d attempts)", npa_generated, attempts)

    # Generate additional PA supplement if needed
    if supp_pa > 0:
        # Import PA patterns from main generator
        import sys
        sys.path.insert(0, str(Path(".").absolute()))
        try:
            from training.generate_synthetic_v2 import PA_PATTERNS, TOPICS, fill_slots, PLATFORM_FLAVOR
        except ImportError:
            logger.warning("Could not import PA patterns — skipping PA supplement")
            supp_pa = 0

        pa_generated = 0
        attempts = 0
        max_attempts = supp_pa * 20
        while pa_generated < supp_pa and attempts < max_attempts:
            attempts += 1
            topic = rng.choice(TOPICS)
            tactic_entry = rng.choice(PA_PATTERNS)
            tactic = tactic_entry[0]
            raw = rng.choice(tactic_entry[1])
            text = fill_slots(raw, topic)
            # Add variation
            if rng.random() < 0.2:
                text = text.rstrip(".!?") + "..."

            if is_fresh(text):
                rec = make_rec(text, "passive_aggressive", "no_threat", tactic, "non_sarcastic", "negative", "supplement_pa")
                supplement_records.append(rec)
                pa_generated += 1
        logger.info("Generated %d PA supplement records (%d attempts)", pa_generated, attempts)

    # Save supplement
    if supplement_records:
        supp_path = data_dir / "supplement_SYNPA.csv"
        field_names = list(supplement_records[0].keys())
        with open(supp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(supplement_records)
        logger.info("Saved %d supplement records → %s", len(supplement_records), supp_path)

        # Re-merge all files
        all_records: list[dict] = []
        
        # Read all batch and supplement files
        for bp in sorted(list(data_dir.glob("batch*_SYNPA_*.csv")) + list(data_dir.glob("supplement_SYNPA*.csv"))):
            with open(bp, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_records.append(row)

        merged_csv = data_dir / "pa_synthetic_merged.csv"
        merged_jsonl = data_dir / "pa_synthetic_merged.jsonl"

        field_names = list(all_records[0].keys()) if all_records else []
        with open(merged_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(all_records)

        with open(merged_jsonl, "w", encoding="utf-8") as f:
            for row in all_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        from collections import Counter
        final_labels = [r["primary_label"] for r in all_records]
        final_threats = [r["threat_label"] for r in all_records]
        logger.info("Final merged dataset: %d records", len(all_records))
        logger.info("Label distribution: %s", dict(Counter(final_labels)))
        logger.info("Threat distribution: %s", dict(Counter(final_threats)))

        # Save final report
        rep = {
            "total": len(all_records),
            "unique_ids": len(set(r["id"] for r in all_records)),
            "label_dist": dict(Counter(final_labels)),
            "threat_dist": dict(Counter(final_threats)),
        }
        with open(reports_dir / "synthetic_generation_report.json", "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
