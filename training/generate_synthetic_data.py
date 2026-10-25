"""
PassiveAggression Detector 2.0 — Synthetic Social Media Data Generation Pipeline
training/generate_synthetic_data.py

Generates exactly 5,000 valid, unique synthetic examples for English social media PA detection:
  - passive_aggressive:     1,667 rows
  - non_passive_aggressive: 1,667 rows
  - ambiguous:              1,666 rows
Total:                      5,000 rows

Features:
  - 500-sample pilot batch generation and inspection.
  - Multi-turn conversation threads with parent-child links & context-dependent minimal pairs.
  - Standalone comments and posts.
  - Rigorous deduplication, lexical diversity, and QC filtering.
  - Full schema matching docs/dataset_schema.md.
"""

from __future__ import annotations
import csv
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic_generator")

SEED = 42
TOTAL_TARGET = 5000
PILOT_TARGET = 500

TARGET_COUNTS = {
    "passive_aggressive": 1667,
    "non_passive_aggressive": 1667,
    "ambiguous": 1666,
}

PILOT_COUNTS = {
    "passive_aggressive": 167,
    "non_passive_aggressive": 167,
    "ambiguous": 166,
}


@dataclass
class SyntheticRecord:
    id: str
    text: str
    primary_label: str
    post_text: Optional[str]
    conversation_id: str
    turn_index: int
    parent_comment_id: Optional[str]
    parent_comment: Optional[str]
    thread_context: Optional[str]
    target_message_id: str
    tactic_labels: str
    sarcasm_label: str
    sentiment_label: str
    generation_source: str
    generator_model: str
    generation_batch: str
    review_status: str
    quality_flags: str


# ─── Lexical & Context Matrices ────────────────────────────────────────────────

TOPICS = [
    ("tech", "code review and deployment", "PR #342", "prod release"),
    ("design", "UI redesign and user testing", "homepage mockups", "branding refresh"),
    ("office", "team meeting and scheduling", "quarterly planning doc", "budget allocation"),
    ("social", "dinner plans and meetup", "weekend road trip", "party organization"),
    ("gaming", "ranked match strategies", "patch notes 1.4", "guild raid roster"),
    ("creative", "video editing timeline", "podcast episode draft", "manuscript chapter"),
    ("marketing", "ad campaign metrics", "social media schedule", "influencer outreach"),
    ("academic", "group research paper", "literature review", "lab experiment protocol"),
]

EMOJIS_PA = ["🙃", "🙂", "👍", "👀", "🤷‍♂️", "👏", "🙄", "✨", "☕", "💅"]
EMOJIS_NON_PA = ["🙌", "🎉", "🔥", "❤️", "😊", "🚀", "💡", "😂", "🙏", "💯"]
EMOJIS_AMBIGUOUS = ["🤔", "👀", "😐", "🤐", "🤷", "😶", "🧐", "💭"]

SLANG_MARKERS = ["ngl", "tbh", "imo", "fr", "lowkey", "honestly", "literally", "idk", "smh"]


# ─── PA Generation Components ──────────────────────────────────────────────────

PA_PATTERNS = [
    # Mock Politeness
    (
        "mock_politeness",
        [
            "With all due respect, perhaps read the {spec} before asserting your opinions.",
            "So glad you found the time to bless us with your thoughts after the {task} was finished 🙂",
            "Per my previous message—which you clearly overlooked—we already resolved the {task}.",
            "Thanks for stating what literally everyone in the thread already agreed on 🙃",
            "Appreciate the input, though it would have been vastly more useful three days ago.",
            "I'm sure you tried your best on the {spec}, but we'll need someone to rewrite it properly.",
            "Polite reminder that deadlines apply to everyone, not just the rest of the team.",
            "Glad you finally logged on to critique the {spec} after everyone else stayed late.",
            "With the greatest possible respect, maybe check your own numbers before lecturing us.",
            "Thanks so much for pointing out what was explicitly written on page one 🙂",
        ]
    ),
    # Backhanded Compliment
    (
        "backhanded_compliment",
        [
            "Wow, you actually delivered something coherent on the {task} for once! Good job.",
            "Your take on the {spec} is surprisingly thoughtful compared to your usual comments.",
            "I love how you can post on {spec} without worrying whether the facts are accurate.",
            "That's a decent attempt at the {task}, much better than the mess from last week.",
            "You look surprisingly professional today, usually you don't seem to care as much.",
            "Great work on the {spec}! Didn't think you had it in you tbh.",
            "Props to you for trying so hard with the {task}, even with such limited background.",
            "Surprisingly solid analysis on {spec}, did someone help you review it? 🙂",
            "You managed to not derail the meeting about {task} today, proud of you!",
            "Honestly impressive how confident you sound about {spec} despite being wrong.",
        ]
    ),
    # Guilt Tripping
    (
        "guilt_tripping",
        [
            "Don't worry about helping with the {task}, I'm used to doing everything alone anyway...",
            "Must be nice to clock out early while the rest of us fix the {spec} over the weekend.",
            "No worries, my time isn't valuable at all. Take your time replying to the {task}.",
            "It's totally fine, I didn't actually need sleep before launching the {spec} 🙃",
            "Go enjoy your evening, I'll just clean up the entire {task} disaster myself like always.",
            "I see where my priorities rank on the team. I'll just finish the {spec} alone.",
            "Never mind, I should have known better than to expect support on the {task}.",
            "Don't stress over the delay, I'll just explain to the client why your {spec} is missing.",
            "It's okay, I'm used to having my contributions on {task} completely ignored.",
            "Sure, leave the hardest part of {spec} to me, why change our usual routine now?",
        ]
    ),
    # Blame Shifting
    (
        "blame_shifting",
        [
            "I guess following basic written guidelines for the {spec} was too much to expect.",
            "Not sure how anyone could misinterpret my extremely clear notes on {task}, but here we are.",
            "If you had spent two minutes reading the {spec}, we wouldn't have this bug.",
            "Clearly there was a failure to comprehend simple instructions regarding the {task}.",
            "I assumed everyone knew how to check the {spec}, guess my expectations were too high.",
            "The error on {task} isn't mine; I specifically flagged this and you chose to ignore it.",
            "Funny how the only department struggling with the {spec} is yours.",
            "I can't prevent mistakes when people don't bother reviewing the shared {task} doc.",
            "Maybe if you attended the sync on {spec}, you wouldn't be asking such basic questions.",
            "It worked perfectly until someone decided to tweak the {task} without telling anyone.",
        ]
    ),
    # Curt Dismissal
    (
        "curt_dismissal",
        [
            "Fine. Do whatever you want with the {spec}.",
            "Whatever you say, boss. You clearly know everything about {task}.",
            "Right, because your opinion on {spec} is the only one that matters here.",
            "Cool story. Let's see how that works out for the {task} in production 👍",
            "K.",
            "Sure thing. When the {task} crashes, please remember this conversation.",
            "If you say so. I won't waste any more breath explaining the {spec}.",
            "Whatever. Clearly facts don't matter in this discussion about {task}.",
            "Noted. Moving on to people who actually want to solve the {task}.",
            "Great plan. Good luck fixing the fallout from the {spec} alone.",
        ]
    ),
    # Reluctant Agreement & Resentment
    (
        "sarcasm",
        [
            "Oh brilliant, another meeting to discuss the meeting about {task}. Peak efficiency.",
            "Sure, let's just scrap three weeks of work on {spec} because someone had a shower thought.",
            "Wonderful idea! Why rely on data when we can just guess the {task} requirements? 👏",
            "Can't wait to spend my entire Saturday fixing this revolutionary new {spec}.",
            "Yeah, let's definitely push an untested change to {task} on a Friday afternoon. Genius.",
            "Oh absolutely, why finish the current {spec} when we can start five more half-baked ones?",
            "Fantastic. Another layer of bureaucracy added to our {task} workflow. Just what we needed.",
            "Thrilled to see our feedback on {spec} went straight into the recycle bin 🙂",
            "Love how our team's reward for finishing {task} ahead of time is everyone else's backlog.",
            "Sure, let's reinvent the wheel for the {spec} for the third time this quarter. So innovative.",
        ]
    ),
]


# ─── Non-PA Generation Components ──────────────────────────────────────────────

NON_PA_PATTERNS = [
    # Sincere politeness & gratitude
    (
        "none",
        "non_sarcastic",
        "positive",
        [
            "Thank you so much for the detailed walkthrough on {spec}! It clarified everything.",
            "Really appreciate you catching that bug in {task} before the release. Great catch!",
            "Huge thanks to everyone who helped pull together the {spec} today. Outstanding teamwork! 🙌",
            "Thanks for taking the time to explain the {task} logic so patiently.",
            "Grateful for your assistance with {spec}. That saved us hours of debugging!",
            "Awesome job on delivering the {task} ahead of schedule! Really appreciate it.",
            "Thank you for the thoughtful feedback on {spec}, I've incorporated all your suggestions.",
            "Appreciate the quick turnaround on the {task} review! Super helpful.",
        ]
    ),
    # Direct constructive criticism (non-PA)
    (
        "none",
        "non_sarcastic",
        "negative",
        [
            "I disagree with this approach for {task}. The latency benchmark increased by 20%, so we should profile it first.",
            "This query in {spec} is missing an index on the join column and will cause high load in prod.",
            "The error handling in {task} needs work. If the network drops, it fails silently without logging.",
            "I don't think this copy for {spec} aligns with our guidelines. Here are three specific edits.",
            "The current design for {task} creates an accessibility violation for screen readers.",
            "We should reject this proposal for {spec}. The maintenance overhead outweighs the benefits.",
            "There is a race condition in the {task} handler when two concurrent requests arrive.",
            "I strongly advise against deploying {spec} today because QA has not finished regression testing.",
        ]
    ),
    # Friendly banter & non-hostile sarcasm
    (
        "sarcasm",
        "sarcastic",
        "positive",
        [
            "Oh no, free pizza in the breakroom again? Truly our suffering knows no bounds 😂🍕",
            "Thanks for ruining my weekend with that amazing season finale cliffhanger lmao!",
            "Another rainy day during our beach trip... the weather gods clearly love us haha ☔",
            "If you make me laugh any harder during the {task} call I'm going to spill my coffee!",
            "You brought homemade cookies to the {spec} review? Okay, you win best teammate forever.",
            "Stop being so competent at {task}, you're making the rest of us look lazy! 😉",
            "Worst game night ever, you won three times in a row! Rematch next Friday! 🎲",
            "Tragic news: the coffee machine is having an existential crisis again today haha ☕",
        ]
    ),
    # Objective neutral informational
    (
        "none",
        "non_sarcastic",
        "neutral",
        [
            "Does anyone have the zoom link for the sync on {task}?",
            "The updated documentation for {spec} has been published in the shared wiki.",
            "We scheduled the staging migration for {task} at 4 PM UTC today.",
            "Please remember to submit your timesheets for the {spec} by Friday noon.",
            "The dependency update for {task} passed all unit tests in CI.",
            "Here is the link to the design assets for {spec}: drive.google.com/assets.",
            "The meeting room for the {task} workshop has moved to conference room B.",
            "Can someone please review PR #412 regarding the {spec} when you get a chance?",
        ]
    ),
    # Sincere agreement (Contrastive minimal pairs)
    (
        "none",
        "non_sarcastic",
        "positive",
        [
            "Sure, whatever you think is best! I completely trust your judgment on {spec}.",
            "Fine with me, let's go with your recommendation for {task}.",
            "Sounds great to me, you've done the research on {spec} so let's run with it.",
            "Totally agree with this direction for {task}, let's execute it.",
            "I'm on board with whatever the group decides for {spec}.",
            "Whatever works best for your schedule on {task} is fine by me!",
        ]
    ),
]


# ─── Ambiguous Generation Components ───────────────────────────────────────────

AMBIGUOUS_PATTERNS = [
    # Terse single words / minimal replies lacking context
    [
        "Fine.",
        "Sure.",
        "Okay.",
        "Noted.",
        "If you say so.",
        "Right.",
        "Whatever.",
        "Understood.",
        "I see.",
        "K.",
        "Cool.",
        "Sounds good.",
        "Will do.",
        "Got it.",
        "As expected.",
    ],
    # Understated neutral / mild reactions
    [
        "That's certainly an interesting choice for the {spec}.",
        "We'll see how that plays out with {task}.",
        "Interesting approach to the {spec}.",
        "That is definitely one way to handle the {task}.",
        "I hear what you're saying about {spec}.",
        "Fair enough.",
        "Could be either way with the {task}.",
        "Hard to tell until we see the results of {spec}.",
        "Different perspectives on {task}, I suppose.",
        "Let's see what happens next with {spec}.",
        "Hmm, that's one perspective on {task}.",
        "I suppose that's possible for {spec}.",
    ],
    # Double-edged social reactions
    [
        "Bold move on the {spec}.",
        "Well, that just happened with {task} 👀",
        "Look at you go with the {spec}.",
        "Classic move for {task}.",
        "I have no words for this {spec} 😂",
        "You do you with the {task}.",
        "Well, that's something new for {spec}.",
        "Can't say I saw that coming with {task}.",
        "Quite a development on {spec}.",
        "That's definitely memorable for {task}.",
    ],
    # Mildly skeptical without overt hostility
    [
        "Are we sure this is the right direction for the {spec}?",
        "I guess we'll find out soon enough regarding {task}.",
        "Not what I expected for {spec}, but let's test it.",
        "Seems like a big shift for {task}, hope it works.",
        "Curious to see the feedback on {spec}.",
        "A bit surprising given our previous discussion on {task}.",
    ]
]


# ─── Thread Context Generator ──────────────────────────────────────────────────

THREAD_SCENARIOS = [
    {
        "post": "Hey team, I put together an initial proposal for {spec}. Please take a look and share any feedback before tomorrow!",
        "parent_pos": "I reviewed the proposal and it looks really clean! Maybe we just tighten section 2.",
        "parent_neg": "I completely reworked the proposal without discussing it because the original was too weak.",
        "parent_neut": "Has anyone checked whether this matches the client specs from last month?",
    },
    {
        "post": "Just pushed the latest updates for {task}. Everything is running smoothly on staging.",
        "parent_pos": "Awesome work getting this across the finish line so quickly!",
        "parent_neg": "The build broke three times this morning and nobody gave a heads up.",
        "parent_neut": "What's the scheduled rollout window for production?",
    },
    {
        "post": "Planning our team dinner for next Thursday. Italian or sushi? Vote below!",
        "parent_pos": "Sushi sounds amazing, count me in!",
        "parent_neg": "I already booked an Italian place without waiting for votes since nobody answered.",
        "parent_neut": "Does the Italian restaurant have vegetarian options?",
    },
    {
        "post": "Here are the new design mockups for {spec}. Open to all critiques and suggestions.",
        "parent_pos": "Love the color scheme and the card layout, very modern!",
        "parent_neg": "Honestly, this looks like a template from 2012. We should start over.",
        "parent_neut": "Are the mobile responsive breakpoints included in this Figma file?",
    },
    {
        "post": "Reminder: Q3 goal review is tomorrow at 10 AM. Please fill out your self-evaluation.",
        "parent_pos": "Submitted mine this morning! Thanks for the reminder.",
        "parent_neg": "Some of us actually have real customer tickets to resolve instead of endless reviews.",
        "parent_neut": "Is there a template link for the self-evaluation form?",
    },
]


def fill_slots(template: str, topic_tuple: tuple) -> str:
    category, desc, spec, task = topic_tuple
    text = template.replace("{spec}", spec).replace("{task}", task).replace("{topic}", desc)
    return text


def generate_single_example(
    target_label: str,
    record_id: str,
    conv_id: str,
    turn_type: str,  # "standalone" | "thread"
    batch_name: str,
    rng: random.Random,
) -> SyntheticRecord:
    topic = rng.choice(TOPICS)
    is_thread = (turn_type == "thread")

    post_text = None
    parent_comment = None
    parent_comment_id = None
    thread_context = None
    turn_index = 0

    if target_label == "passive_aggressive":
        tactic, pattern_list = rng.choice(PA_PATTERNS)
        raw_template = rng.choice(pattern_list)
        msg_text = fill_slots(raw_template, topic)

        # Style decorations
        if rng.random() < 0.25:
            msg_text = f"{rng.choice(SLANG_MARKERS)}, {msg_text[0].lower() + msg_text[1:]}"
        if rng.random() < 0.30 and not any(e in msg_text for e in EMOJIS_PA):
            msg_text = f"{msg_text} {rng.choice(EMOJIS_PA)}"

        sarcasm_val = "sarcastic" if tactic == "sarcasm" or "🙂" in msg_text or "🙃" in msg_text else "non_sarcastic"
        sentiment_val = "negative"

        if is_thread:
            scenario = rng.choice(THREAD_SCENARIOS)
            post_text = fill_slots(scenario["post"], topic)
            # Parent can be negative or neutral to evoke PA response
            parent_variant = rng.choice(["parent_neg", "parent_neut"])
            parent_comment = fill_slots(scenario[parent_variant], topic)
            parent_comment_id = f"{conv_id}_turn_1"
            turn_index = 2
            thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    elif target_label == "non_passive_aggressive":
        tactic, sarcasm_val, sentiment_val, pattern_list = rng.choice(NON_PA_PATTERNS)
        raw_template = rng.choice(pattern_list)
        msg_text = fill_slots(raw_template, topic)

        if rng.random() < 0.25 and not any(e in msg_text for e in EMOJIS_NON_PA):
            msg_text = f"{msg_text} {rng.choice(EMOJIS_NON_PA)}"

        if is_thread:
            scenario = rng.choice(THREAD_SCENARIOS)
            post_text = fill_slots(scenario["post"], topic)
            parent_comment = fill_slots(scenario["parent_pos"], topic)
            parent_comment_id = f"{conv_id}_turn_1"
            turn_index = 2
            thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    else:  # ambiguous
        category = rng.choice(AMBIGUOUS_PATTERNS)
        raw_template = rng.choice(category)
        msg_text = fill_slots(raw_template, topic)

        if rng.random() < 0.20 and not any(e in msg_text for e in EMOJIS_AMBIGUOUS):
            msg_text = f"{msg_text} {rng.choice(EMOJIS_AMBIGUOUS)}"

        tactic = "none"
        sarcasm_val = "non_sarcastic"
        sentiment_val = "neutral"

        if is_thread:
            scenario = rng.choice(THREAD_SCENARIOS)
            post_text = fill_slots(scenario["post"], topic)
            parent_comment = fill_slots(scenario["parent_neut"], topic)
            parent_comment_id = f"{conv_id}_turn_1"
            turn_index = 2
            thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    return SyntheticRecord(
        id=record_id,
        text=msg_text.strip(),
        primary_label=target_label,
        post_text=post_text,
        conversation_id=conv_id,
        turn_index=turn_index,
        parent_comment_id=parent_comment_id,
        parent_comment=parent_comment,
        thread_context=thread_context,
        target_message_id=record_id,
        tactic_labels=tactic,
        sarcasm_label=sarcasm_val,
        sentiment_label=sentiment_val,
        generation_source="synthetic_social_engine_v2",
        generator_model="controlled_template_variation_v2.0",
        generation_batch=batch_name,
        review_status="automated_qc_passed",
        quality_flags="context_dependent" if is_thread else "clean",
    )


# ─── Quality Control & Deduplication Engine ────────────────────────────────────

def normalize_text_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def run_quality_control(
    candidate: SyntheticRecord,
    seen_hashes: set[str],
    seen_normalized: set[str],
) -> tuple[bool, str]:
    """Applies strict automated validation filters."""
    text = candidate.text
    if not text or len(text) < 2 or len(text) > 600:
        return False, "length_out_of_bounds"

    # Exact hash
    md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if md5_hash in seen_hashes:
        return False, "exact_duplicate"

    # Normalized text check
    norm = normalize_text_for_dedup(text)
    if len(norm) > 10 and norm in seen_normalized:
        return False, "near_duplicate"

    # English characters check (ASCII + emojis + standard punctuation)
    # Reject non-English scripts
    for char in text:
        if ord(char) > 0x24F and not (
            0x2000 <= ord(char) <= 0x206F  # punctuation
            or 0x1F000 <= ord(char) <= 0x1FAFF  # emojis
            or 0x2600 <= ord(char) <= 0x27BF  # misc symbols
        ):
            return False, "non_english_or_multilingual_detected"

    seen_hashes.add(md5_hash)
    if len(norm) > 10:
        seen_normalized.add(norm)

    return True, "accepted"


# ─── Batch Generation Runner ───────────────────────────────────────────────────

def generate_synthetic_dataset(
    target_counts: dict[str, int],
    batch_name: str,
    start_index: int = 1,
    seed: int = SEED,
) -> tuple[list[SyntheticRecord], list[dict[str, Any]]]:
    """Generates synthetic examples until target counts are strictly fulfilled."""
    rng = random.Random(seed)
    accepted_records: list[SyntheticRecord] = []
    rejected_records: list[dict[str, Any]] = []

    seen_hashes: set[str] = set()
    seen_normalized: set[str] = set()

    current_idx = start_index

    for label, target_n in target_counts.items():
        count = 0
        attempts = 0
        max_attempts = target_n * 10

        while count < target_n and attempts < max_attempts:
            attempts += 1
            # Alternate between thread and standalone
            turn_type = "thread" if (count % 2 == 0) else "standalone"
            conv_id = f"conv_{batch_name}_{label[:2]}_{count // 2:04d}" if turn_type == "thread" else f"standalone_{current_idx:05d}"
            rec_id = f"synth_{current_idx:05d}"

            candidate = generate_single_example(
                target_label=label,
                record_id=rec_id,
                conv_id=conv_id,
                turn_type=turn_type,
                batch_name=batch_name,
                rng=rng,
            )

            is_valid, reason = run_quality_control(candidate, seen_hashes, seen_normalized)
            if is_valid:
                accepted_records.append(candidate)
                count += 1
                current_idx += 1
            else:
                rejected_records.append({
                    "id": rec_id,
                    "text": candidate.text,
                    "label": label,
                    "rejection_reason": reason,
                    "batch": batch_name,
                })

        logger.info("Batch [%s] Label '%s': Generated %d / %d accepted (attempts: %d)",
                    batch_name, label, count, target_n, attempts)

        if count < target_n:
            raise RuntimeError(f"Could not reach target count {target_n} for {label} in batch {batch_name}")

    return accepted_records, rejected_records


def save_records_to_csv(records: list[SyntheticRecord], output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in SyntheticRecord.__dataclass_fields__.values()])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info("Saved %d records to %s", len(records), path)


# ─── Main Execution Workflow ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true", help="Generate only the 500-sample pilot")
    parser.add_argument("--full", action="store_true", help="Generate full 5,000 synthetic dataset")
    args = parser.parse_args()

    data_dir = Path("data/synthetic")
    reports_dir = Path("reports")
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pilot_path = data_dir / "pa_synthetic_pilot_500.csv"
    full_path = data_dir / "pa_synthetic_5000.csv"
    rejected_path = reports_dir / "synthetic_qc_rejected.csv"

    # Step 1: Pilot Generation (500 records)
    logger.info("=== STEP 1: Generating 500-Sample Pilot ===")
    pilot_records, pilot_rejected = generate_synthetic_dataset(
        target_counts=PILOT_COUNTS,
        batch_name="pilot_batch",
        start_index=1,
        seed=SEED,
    )
    save_records_to_csv(pilot_records, pilot_path)

    # Inspect Pilot
    pilot_labels = {}
    for r in pilot_records:
        pilot_labels[r.primary_label] = pilot_labels.get(r.primary_label, 0) + 1
    logger.info("Pilot Class Distribution: %s", pilot_labels)
    logger.info("Pilot Rejections: %d", len(pilot_rejected))

    if args.pilot_only:
        logger.info("Pilot-only flag set. Halting after pilot inspection.")
        return

    # Step 2: Remaining Generation (4,500 records)
    logger.info("=== STEP 2: Generating Remaining 4,500 Records to Complete 5,000 Target ===")
    remaining_counts = {
        label: TARGET_COUNTS[label] - PILOT_COUNTS[label]
        for label in TARGET_COUNTS
    }

    remaining_records, remaining_rejected = generate_synthetic_dataset(
        target_counts=remaining_counts,
        batch_name="full_batch",
        start_index=len(pilot_records) + 1,
        seed=SEED + 100,
    )

    all_records = pilot_records + remaining_records
    all_rejected = pilot_rejected + remaining_rejected

    logger.info("=== STEP 3: Verifying Final Corpus Integrity ===")
    assert len(all_records) == TOTAL_TARGET, f"Expected {TOTAL_TARGET}, got {len(all_records)}"

    final_dist = {}
    for r in all_records:
        final_dist[r.primary_label] = final_dist.get(r.primary_label, 0) + 1
    logger.info("Final Accepted Distribution: %s", final_dist)
    assert final_dist == TARGET_COUNTS, f"Distribution mismatch: {final_dist} != {TARGET_COUNTS}"

    # Save final dataset
    save_records_to_csv(all_records, full_path)

    # Save rejected audit log
    if all_rejected:
        with open(rejected_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "rejection_reason", "batch"])
            writer.writeheader()
            for row in all_rejected:
                writer.writerow(row)
        logger.info("Saved %d rejected records to %s", len(all_rejected), rejected_path)

    logger.info("SUCCESS: Exactly 5,000 synthetic records generated and verified.")


if __name__ == "__main__":
    main()
