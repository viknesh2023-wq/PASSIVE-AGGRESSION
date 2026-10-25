"""
PassiveAggression Detector 2.0 — Synthetic Social Media Dataset Generator v2
training/generate_synthetic_v2.py

Generates exactly 10,000 records across 4 batches of 2,500:
  Batch 1: SYNPA-00001 – SYNPA-02500
  Batch 2: SYNPA-02501 – SYNPA-05000
  Batch 3: SYNPA-05001 – SYNPA-07500
  Batch 4: SYNPA-07501 – SYNPA-10000

Primary labels:
  passive_aggressive:     3,334
  non_passive_aggressive: 3,333
  ambiguous:              3,333

Threat labels (independent):
  no_threat, direct_threat, implied_threat, ambiguous

All content is English social-media style: Instagram, YouTube, Reddit, X/Twitter,
TikTok, Discord, and group-chat messages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("synth_v2")

# ── Constants ──────────────────────────────────────────────────────────────────
SEED = 2024
TOTAL_TARGET = 10_000
BATCH_SIZE   = 2_500

BATCH_CONFIGS = [
    {"name": "batch1", "start": 1,    "end": 2_500},
    {"name": "batch2", "start": 2_501, "end": 5_000},
    {"name": "batch3", "start": 5_001, "end": 7_500},
    {"name": "batch4", "start": 7_501, "end": 10_000},
]

# Per-batch class targets (for balanced distribution across all 4 batches)
# Total: PA=3334, Non-PA=3333, Ambiguous=3333
BATCH_COUNTS = [
    {"passive_aggressive": 834,  "non_passive_aggressive": 833, "ambiguous": 833},
    {"passive_aggressive": 834,  "non_passive_aggressive": 833, "ambiguous": 833},
    {"passive_aggressive": 833,  "non_passive_aggressive": 834, "ambiguous": 833},
    {"passive_aggressive": 833,  "non_passive_aggressive": 833, "ambiguous": 834},
]

GENERATION_SOURCE  = "synthetic"
REVIEW_STATUS      = "synthetic_unreviewed"
GENERATOR_MODEL    = "controlled_template_variation_v2.1_social_media"


# ── Schema ─────────────────────────────────────────────────────────────────────
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
    tactic_labels:       str   # JSON array string or empty list "[]"
    sarcasm_label:       str
    sentiment_label:     str
    generation_source:   str
    generator_model:     str
    generation_batch:    str
    review_status:       str
    quality_flags:       str


# ── Topic matrices ─────────────────────────────────────────────────────────────
# (platform, content_domain, spec_slot, task_slot)
TOPICS: list[tuple[str, str, str, str]] = [
    # Instagram
    ("instagram", "posting a photo",        "the new look",           "the post"),
    ("instagram", "story reactions",        "the story highlight",    "the caption"),
    ("instagram", "collab content",         "the collab reel",        "the brand deal"),
    ("instagram", "comment section drama",  "the comment thread",     "the photo"),
    # YouTube
    ("youtube",   "video critique",         "the new upload",         "the video"),
    ("youtube",   "channel direction",      "the series format",      "the script"),
    ("youtube",   "community post",         "the community tab",      "the announcement"),
    ("youtube",   "comment debate",         "the comment section",    "the video response"),
    # Reddit
    ("reddit",    "subreddit discussion",   "the post",               "the thread"),
    ("reddit",    "AITA thread",            "the situation",          "the judgment"),
    ("reddit",    "tech help",              "the solution",           "the issue"),
    ("reddit",    "opinion thread",         "the take",               "the argument"),
    # X/Twitter
    ("twitter",   "reply thread",           "the tweet",              "the reply"),
    ("twitter",   "quote tweet drama",      "the QT",                 "the original"),
    ("twitter",   "hot take",               "the opinion",            "the thread"),
    # TikTok
    ("tiktok",    "duet response",          "the duet",               "the video"),
    ("tiktok",    "trend participation",    "the trend",              "the sound"),
    ("tiktok",    "comment drama",          "the comment",            "the TikTok"),
    # Discord
    ("discord",   "server announcement",   "the announcement",       "the update"),
    ("discord",   "gaming strategy",       "the strat",              "the raid plan"),
    ("discord",   "creative project",      "the project",            "the submission"),
    ("discord",   "server drama",          "the situation",          "the thread"),
    # General group chat
    ("groupchat", "group project",         "the draft",              "the task"),
    ("groupchat", "event planning",        "the plan",               "the event"),
    ("groupchat", "shared decision",       "the choice",             "the vote"),
]

PLATFORMS = [t[0] for t in TOPICS]

# Platform-specific emojis/slang flavoring
PLATFORM_FLAVOR: dict[str, dict[str, Any]] = {
    "instagram": {
        "pa_emojis":    ["🙃", "✨", "💅", "🙄", "👀", "😇"],
        "non_pa_emojis":["❤️", "🔥", "😍", "🙌", "💯", "🎉"],
        "slang":        ["omg", "literally", "no cap", "slay", "it's giving"],
    },
    "youtube":   {
        "pa_emojis":    ["🙂", "👍", "☕", "🤷", "👏"],
        "non_pa_emojis":["🔥", "💯", "🚀", "😂", "❤️"],
        "slang":        ["ngl", "lowkey", "fr fr", "bestie", "period"],
    },
    "reddit":    {
        "pa_emojis":    ["🙃", "🤡", "💀", "👀", "🤷"],
        "non_pa_emojis":["🙌", "💡", "👏", "🙏", "😂"],
        "slang":        ["tbh", "imo", "idk", "ngl", "smh", "lmao"],
    },
    "twitter":   {
        "pa_emojis":    ["🙂", "👀", "😭", "💀", "😐"],
        "non_pa_emojis":["💯", "🔥", "❤️", "😂", "🙌"],
        "slang":        ["lmaooo", "bestie", "no but fr", "not me", "the way"],
    },
    "tiktok":    {
        "pa_emojis":    ["💅", "😭", "🤭", "🙃", "👏"],
        "non_pa_emojis":["🔥", "❤️", "😍", "🥺", "💯"],
        "slang":        ["it's giving", "main character", "slay", "periodt", "understood the assignment"],
    },
    "discord":   {
        "pa_emojis":    ["🙂", "💀", "🤡", "👀", "😶"],
        "non_pa_emojis":["🔥", "💯", "👌", "😂", "🙌"],
        "slang":        ["ngl", "based", "fr", "bruh", "L take", "W"],
    },
    "groupchat": {
        "pa_emojis":    ["🙃", "🙂", "👍", "👀", "🤷"],
        "non_pa_emojis":["❤️", "👍", "🙌", "😊", "💯"],
        "slang":        ["ngl", "tbh", "honestly", "idk", "lmk"],
    },
}


# ── PA Pattern Library ─────────────────────────────────────────────────────────
# Format: (tactic, list_of_templates)
# {spec} and {task} are filled from topic tuple, {platform} from topic[0]

PA_PATTERNS: list[tuple[str, list[str]]] = [
    ("fake_politeness", [
        # workplace/collab
        "With all due respect, maybe read {spec} before sharing your hot take.",
        "Thanks for eventually weighing in on {task} after the rest of us sorted it 🙂",
        "Appreciate the thought, though it would've been more useful three days ago on {spec}.",
        "So glad you found time to grace us with your opinion on {task} today.",
        "Per my earlier message — which you clearly skimmed — we already covered {spec}.",
        "Gentle reminder: deadlines apply to everyone contributing to {task}.",
        "Thanks for pointing out what was literally written in the description of {spec} 🙃",
        "I'm sure you did your best on {task}. We'll just go ahead and redo it.",
        "Of course. I'll just wait here while you decide whether {spec} matters.",
        "No worries, I love being the only one who shows up for {task} every single time.",
        # Social media flavors
        "Totally fine that you posted {spec} without crediting the original. Really.",
        "Thanks for the 'support' on {task} 🙂 really felt it.",
        "Bless your heart for thinking your take on {spec} was needed.",
        "It's so cute how you think your opinion on {task} changes anything.",
        "Appreciate you engaging with {spec}, even if you clearly didn't watch it 👍",
        "Glad you could spare five seconds to react to {task} instead of sharing it.",
        "Not that I'm keeping score on {spec} but yes, I noticed who didn't comment.",
        "Thanks for the unsolicited advice on {task} 😊 super helpful as always.",
        "How sweet that you shared {spec} without mentioning who made it.",
        "Oh of course, your one-word response on {task} was so thoughtful of you 🙂",
    ]),
    ("indirect_criticism", [
        "I just find it interesting that {spec} was posted without any context.",
        "Not gonna say anything about {task}... but some people should reflect.",
        "I guess some creators just have different standards when it comes to {spec}.",
        "It's wild how some folks will post on {task} and never respond to comments.",
        "I wonder how many people actually watch {spec} before sharing it.",
        "Some people just love sharing opinions on {task} without the receipts.",
        "Funny how the loudest voices about {spec} are usually the least informed.",
        "I'm not saying {task} was poorly thought out, but... well.",
        "Interesting approach to {spec}. Definitely an approach.",
        "Some of us actually research before posting takes on {task} 🤷",
        "Not mentioning names, but whoever ghosted {spec} knows who they are.",
        "Certain people contributing to {task} might want to check their track record.",
        "I notice the people criticizing {spec} weren't around when the work happened.",
        "I'll let {task} speak for itself. You'll all see eventually.",
        "Not pointing fingers at {spec}, just saying some patterns are hard to miss.",
    ]),
    ("backhanded_compliment", [
        "Wow, {spec} is actually decent — didn't think you had it in you tbh.",
        "You really outdid yourself on {task}! That's surprising given last time.",
        "Finally got {spec} right after a few tries. Proud of you 🙂",
        "Better late than never with {task}, I suppose.",
        "Honestly? {spec} was more coherent than I expected from you.",
        "Not bad for a first serious attempt at {task}!",
        "You managed to post {spec} without drama this time. Growth!",
        "At least {task} is an improvement on what you did before.",
        "For once, {spec} actually made a point. Well done.",
        "I'm genuinely impressed you pulled off {task} with the resources you had.",
        "Your take on {spec} is way better than your usual stuff, no offense.",
        "This {task} gave me real professional vibes — didn't see that coming.",
        "Kind of obsessed with how you overcame your usual weaknesses on {spec}.",
        "I didn't expect to like {task} but here we are. Props.",
        "You almost got {spec} perfect. So close!",
    ]),
    ("implied_blame", [
        "If someone had actually read {spec}, we wouldn't be in this situation.",
        "The issue with {task} started the moment certain people stopped paying attention.",
        "I specifically flagged {spec} and you chose to ignore it. Here we are.",
        "I can't be responsible for {task} when key people don't show up.",
        "Whoever changed {spec} last night without telling anyone, this one's on you.",
        "It's not my fault {task} flopped. I said my concerns at the start.",
        "We all saw how {spec} was going. Some of us tried to warn everyone.",
        "Funny how the problems with {task} appeared right after [certain changes] 🙃",
        "I don't know who decided to redo {spec} without asking but good luck fixing it.",
        "There's a reason I documented everything about {task} — for exactly this moment.",
        "Ask yourself who was actually present during {spec} planning. Then connect the dots.",
        "People keep acting confused about {task} like the problems weren't obvious.",
        "The comments on {spec} said it all, but nobody wanted to listen.",
        "I've been saying {task} needed fixing for weeks. Not my problem anymore.",
        "Whoever missed the {spec} check-in is now responsible for all of this.",
    ]),
    ("sarcastic_agreement", [
        "Oh sure, let's scrap everything and restart {spec}. Makes total sense 👏",
        "Oh absolutely, posting {task} at midnight with no context is peak strategy.",
        "Great idea! Let's ask everyone's opinion on {spec} and then ignore it.",
        "Can't wait to spend my weekend fixing {task}. So excited.",
        "Yeah, let's definitely go with the option nobody wanted for {spec}.",
        "Oh brilliant, another last-minute change to {task}. Peak organization.",
        "Sure, deleting {spec} and reposting without crediting anyone sounds fine.",
        "Of course, the best time to announce {task} changes is after it's done.",
        "Right, because asking people to redo {spec} on zero notice is totally fair.",
        "Wonderful plan. I'm sure {task} will go perfectly. As always.",
        "Amazing how {spec} always gets updated right before the deadline 🙂",
        "Bold strategy ignoring all the feedback on {task}. Let's see how it goes.",
        "Love how {spec} had to be redone because someone didn't check first.",
        "Oh nice, so we're just winging {task} again. Good to know.",
        "Incredible that {spec} was approved like that. Truly visionary.",
    ]),
    ("dismissiveness", [
        "Fine. Do whatever you want with {spec}.",
        "Sure. When {task} fails, don't say nobody warned you.",
        "K. I'll just watch {spec} crash from here.",
        "Right. Because your idea about {task} is obviously the best one.",
        "Whatever you say. I'm done arguing about {spec}.",
        "Cool, good luck with {task} 👍",
        "Noted. Guess {spec} doesn't matter then.",
        "If that's how you want to handle {task}, be my guest.",
        "Sure. {spec} is your call. Completely.",
        "If you say so. I've said my piece about {task}.",
        "Okay. Can we move on from {spec} now?",
        "Sounds like a plan. Let's see how {task} turns out.",
        "Fine. Not my problem what happens with {spec}.",
        "Whatever. I'm out of this {task} conversation.",
        "Great. So we're just ignoring {spec}. Cool.",
    ]),
]

# ── Non-PA Pattern Library ──────────────────────────────────────────────────────
# Format: (tactic, sarcasm_label, sentiment_label, templates)
NON_PA_PATTERNS: list[tuple[str, str, str, list[str]]] = [
    ("none", "non_sarcastic", "positive", [
        "This {spec} is genuinely amazing — thank you so much for putting it together!",
        "Really appreciate how much effort went into {task} 🙌",
        "Huge shoutout to everyone who contributed to {spec}. Outstanding work!",
        "I love the direction of {task}! Keep it up!",
        "This is exactly what I was hoping for with {spec}. Well done!",
        "Thank you for such a thorough breakdown of {task} — super helpful!",
        "Honestly one of the best takes on {spec} I've seen in a while.",
        "{task} came out even better than expected! Proud of everyone involved.",
        "Just want to say how much I appreciate the work on {spec} 💯",
        "You really knocked it out of the park with {task} this time!",
        "This whole {spec} thread restored my faith in this community.",
        "Solid collaboration on {task} — this is what teamwork looks like 🔥",
        "I'm so here for this {spec} energy. Pure quality.",
        "Genuinely excited to see where {task} goes from here!",
        "The feedback on {spec} has been so constructive. Love this community.",
    ]),
    ("none", "non_sarcastic", "negative", [
        "I disagree with this approach to {spec}. Here's why: the format doesn't scale.",
        "{task} has a real consistency problem that needs to be addressed before launch.",
        "Constructive feedback: {spec} would benefit from clearer structure in section 2.",
        "I have concerns about {task} — specifically the timeline looks unrealistic.",
        "The current version of {spec} misses several key points from our earlier discussion.",
        "I don't think {task} is ready yet. Here are three specific things that need work.",
        "{spec} needs to be revised. The conclusions don't follow from the data shown.",
        "Honest feedback: {task} felt rushed compared to your previous work.",
        "I strongly recommend against publishing {spec} until the attribution is fixed.",
        "This approach to {task} has a significant flaw that should be fixed first.",
        "{spec} has potential but the execution needs another pass.",
        "I'd push back on {task} — the framing is off and could mislead people.",
        "The argument in {spec} loses me at the third paragraph — needs clarification.",
        "I'm worried {task} doesn't address the main concern that was raised.",
        "The editing on {spec} is inconsistent. Worth another review before going live.",
    ]),
    ("none", "non_sarcastic", "neutral", [
        "Can someone share the link to {spec}? Can't find it in the channel.",
        "Reminder: {task} is scheduled for review tomorrow at 3 PM.",
        "The updated version of {spec} has been uploaded to the drive.",
        "Quick update: {task} is now live on the platform.",
        "Has anyone reviewed {spec} yet? Waiting on feedback before proceeding.",
        "The timeline for {task} has been updated. Check the pinned message.",
        "Here's a summary of {spec} for anyone who missed the earlier discussion.",
        "Following up on {task} — we need a decision by end of today.",
        "Does anyone have access to {spec}? Can't seem to pull it up.",
        "For context: {task} was originally proposed two weeks ago.",
        "Sharing {spec} here as requested. Let me know if anything is missing.",
        "The {task} report has been submitted. Waiting on approvals now.",
        "Just checking — is {spec} the latest version or should I use the older one?",
        "FYI: {task} is on hold pending review. No changes until then.",
        "Updated {spec} with the corrections from last session. See thread.",
    ]),
    ("sarcasm", "sarcastic", "positive", [
        "Oh great, {spec} broke again right before the deadline. Love that for us 😂",
        "Free time this weekend? Absolutely not — {task} needs me. As always 🫠",
        "Oh sure, let's ALL meet on the ONE day I had plans for {spec} 💀",
        "The {task} wifi cutting out mid-presentation is SO on brand lmao",
        "Truly the universe loves us — {spec} went down during peak hours again 😭",
        "Wow {task} somehow got even more chaotic. Truly thriving 🔥",
        "Coffee machine broken on the day of the big {spec}? Classic.",
        "Oh amazing, the entire {task} system crashed on the busiest day. Love it.",
        "Perfect timing on {spec} going viral right when the servers can't handle it 😂",
        "Nothing like a last-minute {task} update to keep things exciting, right? 😅",
        "The {spec} formatting breaking on every browser except mine is so relatable.",
        "Honestly the {task} outage on launch day is on-brand at this point lmao.",
        "Three hours into {spec} and the power went out. Storytelling arc of the year.",
        "Oh the {task} ended early? Don't worry we'll lose sleep over it anyway 😭",
        "The {spec} algorithm hating my content is really my villain origin story 😂",
    ]),
    ("none", "non_sarcastic", "positive", [
        # Sincere agreement / support (minimal pairs vs dismissive PA)
        "Sure, whatever you think is best for {spec} — I trust your judgment.",
        "Fine by me, let's go with your recommendation on {task}.",
        "Totally on board with the direction for {spec}!",
        "I'm flexible on {task} — whatever the group decides works for me.",
        "Your call on {spec}, you know it best.",
        "Sounds good to me — let's run with {task} as planned.",
        "I'm happy to defer to your expertise on {spec}.",
        "Whatever works best for the team on {task} is fine with me!",
        "No objections here — let's move forward with {spec}.",
        "If that's what makes {task} work best, I'm all for it.",
    ]),
    ("none", "non_sarcastic", "negative", [
        # Overt direct anger (NOT passive-aggressive — important negative example)
        "I'm genuinely frustrated with how {spec} was handled. This isn't okay.",
        "This is unacceptable. {task} was due two weeks ago and nothing was delivered.",
        "I'm angry about {spec}. Not in a subtle way — directly and openly angry.",
        "I can't believe {task} was approved without any quality check. That's on leadership.",
        "No, I'm not okay with {spec}. I'm saying that clearly and directly.",
        "I'm calling this out: {task} failed because of lack of accountability.",
        "This is a hard no from me. {spec} needs to be completely redone.",
        "I'm raising this formally. {task} standards have been consistently ignored.",
        "I'm upset and I'm saying so: {spec} was mishandled from the start.",
        "Direct feedback: {task} was not ready and should never have been published.",
    ]),
]

# ── Ambiguous Pattern Library ───────────────────────────────────────────────────
AMBIGUOUS_PATTERNS: list[list[str]] = [
    # Terse single-word replies
    ["Fine.", "Sure.", "Okay.", "Noted.", "K.", "Cool.", "Right.", "Whatever.",
     "Got it.", "Will do.", "Understood.", "Interesting.", "As expected.", "Sure thing."],
    # Understated reactions
    [
        "That's certainly an interesting take on {spec}.",
        "We'll see how {task} plays out.",
        "I hear what you're saying about {spec}.",
        "Fair enough.",
        "That is definitely one way to do {task}.",
        "Hard to tell until we see results from {spec}.",
        "Different people have different approaches to {task}.",
        "Could go either way with {spec}.",
        "Hmm. That's one perspective on {task}.",
        "I suppose that's possible with {spec}.",
    ],
    # Double-edged social reactions
    [
        "Bold move posting {spec} without warning 👀",
        "Well, that just happened with {task}.",
        "Look at you go with {spec}.",
        "Classic. Just classic {task}.",
        "I have no words for {spec} 💀",
        "You do you with {task}.",
        "Quite a development on {spec}.",
        "Can't say I saw that coming with {task}.",
        "That's definitely a choice for {spec}.",
        "Memorable. Truly memorable {task}.",
    ],
    # Mild skepticism without overt hostility
    [
        "Are we sure {spec} is the right move here?",
        "Curious to see how {task} lands.",
        "Not what I expected from {spec}, but let's wait and see.",
        "Seems like a big shift for {task}. Hope it works.",
        "A bit surprised given previous discussions about {spec}.",
        "Interesting timing on {task}.",
        "Let's see what the response to {spec} looks like.",
        "I guess {task} will tell us if this works.",
    ],
    # Social-media specific ambiguous
    [
        "Didn't expect to see {spec} today 👀",
        "So we're just doing {task} like this now?",
        "The way {spec} just dropped with no context…",
        "I'll just leave {task} right here without comment.",
        "Saying nothing about {spec}. Absolutely nothing.",
        "{task} happened and I have thoughts. Many thoughts.",
        "Logging off after seeing {spec}. Need a minute.",
        "The audacity of {task} to show up on my feed today.",
    ],
]


# ── Thread Context Generator ────────────────────────────────────────────────────
THREAD_SCENARIOS: list[dict[str, str]] = [
    {
        "post":        "Hey everyone, sharing my latest {spec} — would love honest feedback!",
        "parent_pos":  "This is really solid work. Really impressed with the execution.",
        "parent_neg":  "Honestly, this {spec} needed more polish before going public.",
        "parent_neut": "Is this the final version or are there more updates coming?",
    },
    {
        "post":        "Just posted the update for {task} — let me know what you all think!",
        "parent_pos":  "Love this! The improvement from the last version is noticeable.",
        "parent_neg":  "I feel like {task} missed the mark. The core issue wasn't addressed.",
        "parent_neut": "What tools did you use for {task}? Asking for future reference.",
    },
    {
        "post":        "PSA: {task} has been updated — please review before commenting.",
        "parent_pos":  "Already reviewed! Looks great, no issues on my end.",
        "parent_neg":  "Read it. Still has the same problems as before, unfortunately.",
        "parent_neut": "Where's the link to the updated {task}?",
    },
    {
        "post":        "Dropping my thoughts on {spec} here — constructive discussion welcome.",
        "parent_pos":  "Great take! Especially the part about the approach.",
        "parent_neg":  "I disagree with most of this. {spec} needs a more nuanced framing.",
        "parent_neut": "Can you elaborate on what you mean by the third point?",
    },
    {
        "post":        "Reminder: deadline for {task} is tomorrow. Please wrap up your parts.",
        "parent_pos":  "Already done! Sent it over this morning.",
        "parent_neg":  "I haven't started {task}. Can we push the deadline?",
        "parent_neut": "Can someone confirm what format {task} should be submitted in?",
    },
    {
        "post":        "Share your honest thoughts on {spec} — no judgment here.",
        "parent_pos":  "It's genuinely one of the best {spec} posts I've seen this month.",
        "parent_neg":  "I clicked on {spec} expecting more. Left disappointed.",
        "parent_neut": "How long did {spec} take you to put together?",
    },
]


# ── Threat Label Library ────────────────────────────────────────────────────────
# Threat labels are assigned independently of PA labels.
# The vast majority of messages are no_threat.

THREAT_MESSAGES = {
    "direct_threat": [
        "If you post {spec} again I will report your account and make sure you're banned.",
        "Touch {task} one more time and see what happens.",
        "Post that {spec} and I'll make sure everyone knows what you really did.",
        "Keep talking about {task} and I'll have your whole channel taken down.",
        "I know exactly who you are and where {spec} is hosted. Don't test me.",
        "One more comment about {task} and you'll regret it.",
        "You think deleting {spec} protects you? I saved everything.",
        "Try updating {task} without asking me again. I dare you.",
        "I will personally make sure {spec} never gets another view.",
        "Post that again and I'll flood {task} with reports until it's gone.",
    ],
    "implied_threat": [
        "It would be a shame if {spec} suddenly got mass-reported by the community.",
        "Interesting choices you're making with {task}. Hope nothing comes of it.",
        "I know people who could make {spec} disappear overnight. Just saying.",
        "Keep pushing {task} and you'll find out why I don't get pushed back.",
        "Funny how accounts get hacked when they keep posting things like {spec}.",
        "You've been warned about {task}. Don't say you weren't.",
        "Be careful with {spec}. Not everyone is as patient as I am.",
        "I'm keeping track of every comment you make about {task}.",
        "I'd be careful about who I tag in {spec} if I were you.",
        "Let's just say {task} won't be around much longer if things keep up.",
    ],
    "ambiguous_threat": [
        "You'll see what happens with {spec}. It's only a matter of time.",
        "I'm watching {task} very carefully. Act accordingly.",
        "I hope {spec} was worth it.",
        "We'll see how {task} ends. That's all I'll say.",
        "Keep laughing about {spec}. See where it gets you.",
        "I don't make threats — I make promises about {task}.",
        "Everyone who contributed to {spec} will hear from me eventually.",
        "I'm not done with {task}. Not by a long shot.",
        "Big day tomorrow regarding {spec}. Stay tuned.",
        "I have receipts on {task}. Every single one.",
    ],
}

# No-threat examples that might seem threatening but aren't
NO_THREAT_EXPRESSIONS = [
    # Figurative / hyperbole
    "I'm going to kill that song, it's been stuck in my head all week!",
    "If {spec} doesn't drop soon I will literally combust with excitement.",
    "I'm dead 💀 {task} is the funniest thing I've seen all week.",
    "This {spec} is killing me, it's so good omg.",
    "If you spoil {task} for me I will unfriend you, no joke 😂",
    "I'm gonna destroy everyone in this {task} tournament lol.",
    "If {spec} doesn't get updated soon I'm going to riot (jokingly).",
    "I could watch {task} a thousand times and never get tired 😍",
    # Strong negative but non-threatening
    "I absolutely hate how {spec} was handled.",
    "This {task} is the worst thing I've seen today.",
    "{spec} ruined my day. Genuinely.",
    "I'm so frustrated with {task} right now.",
    "Done with {spec}. Not giving it another second of my time.",
    "I'll never support anything related to {task} again.",
    "Block, report, move on from {spec}.",
]


# ── Helper Functions ────────────────────────────────────────────────────────────
def fill_slots(template: str, topic: tuple) -> str:
    platform, domain, spec, task = topic
    return (template
            .replace("{spec}", spec)
            .replace("{task}", task)
            .replace("{topic}", domain)
            .replace("{platform}", platform))


def get_platform_flavor(platform: str, category: str, rng: random.Random) -> str:
    """Return a random emoji/slang for the platform and PA/non-PA category."""
    flavor = PLATFORM_FLAVOR.get(platform, PLATFORM_FLAVOR["groupchat"])
    key = "pa_emojis" if category == "pa" else "non_pa_emojis"
    return rng.choice(flavor[key])


def assign_threat_label(
    primary_label: str,
    text: str,
    rng: random.Random,
    special_threat: Optional[str] = None,
) -> str:
    """
    Assign an independent threat label.
    - ~2% of messages get a direct_threat
    - ~3% get implied_threat
    - ~3% get ambiguous threat
    - rest are no_threat
    Threat labels are independent of PA labels.
    """
    if special_threat:
        return special_threat

    r = rng.random()
    if r < 0.02:
        return "direct_threat"
    elif r < 0.05:
        return "implied_threat"
    elif r < 0.08:
        return "ambiguous"
    else:
        return "no_threat"


def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def is_valid(candidate: SynRecord, seen_hashes: set, seen_norms: set) -> tuple[bool, str]:
    text = candidate.text
    if not text or len(text.strip()) < 5 or len(text) > 700:
        return False, "length_out_of_bounds"

    md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    if md5 in seen_hashes:
        return False, "exact_duplicate"

    norm = normalize_for_dedup(text)
    if len(norm) > 10 and norm in seen_norms:
        return False, "near_duplicate"

    # English check: reject non-Latin scripts
    for ch in text:
        cp = ord(ch)
        if ch.isalpha():
            import unicodedata
            name = unicodedata.name(ch, "")
            if any(name.startswith(script) for script in (
                "CYRILLIC", "ARABIC", "DEVANAGARI", "TAMIL", "CJK", "HANGUL",
                "HIRAGANA", "KATAKANA", "THAI", "GREEK", "HEBREW",
            )):
                return False, "non_english"

    seen_hashes.add(md5)
    if len(norm) > 10:
        seen_norms.add(norm)
    return True, "accepted"


# ── Record Generator ────────────────────────────────────────────────────────────
def generate_pa_record(
    record_id: str, conv_id: str, is_thread: bool, batch_name: str, rng: random.Random
) -> SynRecord:
    topic = rng.choice(TOPICS)
    platform = topic[0]

    tactic_entry = rng.choice(PA_PATTERNS)
    tactic = tactic_entry[0]
    raw = rng.choice(tactic_entry[1])
    text = fill_slots(raw, topic)

    # Style decoration
    if rng.random() < 0.30:
        emoji = get_platform_flavor(platform, "pa", rng)
        if emoji not in text:
            text = f"{text} {emoji}"
    if rng.random() < 0.20:
        flavor = PLATFORM_FLAVOR.get(platform, PLATFORM_FLAVOR["groupchat"])
        slang = rng.choice(flavor["slang"])
        text = f"{slang}, {text[0].lower()}{text[1:]}"

    # Ellipsis decoration (occasional)
    if rng.random() < 0.15 and not text.endswith("..."):
        text = text.rstrip(".!?") + "..."

    sarcasm_label = "sarcastic" if tactic == "sarcastic_agreement" else "non_sarcastic"
    sentiment_label = "negative"

    post_text = parent_comment = parent_comment_id = thread_context = None
    turn_index = 0

    if is_thread:
        scenario = rng.choice(THREAD_SCENARIOS)
        post_text = fill_slots(scenario["post"], topic)
        parent_variant = rng.choice(["parent_neg", "parent_neut"])
        parent_comment = fill_slots(scenario[parent_variant], topic)
        parent_comment_id = f"{conv_id}_turn_1"
        turn_index = 2
        thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    threat_label = assign_threat_label("passive_aggressive", text, rng)

    return SynRecord(
        id=record_id,
        text=text.strip(),
        primary_label="passive_aggressive",
        threat_label=threat_label,
        post_text=post_text,
        conversation_id=conv_id,
        turn_index=turn_index,
        parent_comment_id=parent_comment_id,
        parent_comment=parent_comment,
        thread_context=thread_context,
        target_message_id=record_id,
        tactic_labels=json.dumps([tactic]),
        sarcasm_label=sarcasm_label,
        sentiment_label=sentiment_label,
        generation_source=GENERATION_SOURCE,
        generator_model=GENERATOR_MODEL,
        generation_batch=batch_name,
        review_status=REVIEW_STATUS,
        quality_flags="context_dependent" if is_thread else "clean",
    )


def generate_non_pa_record(
    record_id: str, conv_id: str, is_thread: bool, batch_name: str, rng: random.Random
) -> SynRecord:
    topic = rng.choice(TOPICS)
    platform = topic[0]

    entry = rng.choice(NON_PA_PATTERNS)
    tactic, sarcasm_label, sentiment_label, templates = entry
    raw = rng.choice(templates)
    text = fill_slots(raw, topic)

    if rng.random() < 0.25:
        emoji = get_platform_flavor(platform, "non_pa", rng)
        if emoji not in text:
            text = f"{text} {emoji}"

    post_text = parent_comment = parent_comment_id = thread_context = None
    turn_index = 0

    if is_thread:
        scenario = rng.choice(THREAD_SCENARIOS)
        post_text = fill_slots(scenario["post"], topic)
        parent_comment = fill_slots(scenario["parent_pos"], topic)
        parent_comment_id = f"{conv_id}_turn_1"
        turn_index = 2
        thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    # Non-PA messages can still be threatening (independent labels)
    threat_label = assign_threat_label("non_passive_aggressive", text, rng)

    return SynRecord(
        id=record_id,
        text=text.strip(),
        primary_label="non_passive_aggressive",
        threat_label=threat_label,
        post_text=post_text,
        conversation_id=conv_id,
        turn_index=turn_index,
        parent_comment_id=parent_comment_id,
        parent_comment=parent_comment,
        thread_context=thread_context,
        target_message_id=record_id,
        tactic_labels=json.dumps([]),
        sarcasm_label=sarcasm_label,
        sentiment_label=sentiment_label,
        generation_source=GENERATION_SOURCE,
        generator_model=GENERATOR_MODEL,
        generation_batch=batch_name,
        review_status=REVIEW_STATUS,
        quality_flags="context_dependent" if is_thread else "clean",
    )


def generate_ambiguous_record(
    record_id: str, conv_id: str, is_thread: bool, batch_name: str, rng: random.Random
) -> SynRecord:
    topic = rng.choice(TOPICS)
    platform = topic[0]

    category = rng.choice(AMBIGUOUS_PATTERNS)
    raw = rng.choice(category)
    text = fill_slots(raw, topic)

    if rng.random() < 0.20:
        ambiguous_emojis = ["🤔", "👀", "😐", "🤷", "😶", "🧐", "💭"]
        emoji = rng.choice(ambiguous_emojis)
        if emoji not in text:
            text = f"{text} {emoji}"

    post_text = parent_comment = parent_comment_id = thread_context = None
    turn_index = 0

    if is_thread:
        scenario = rng.choice(THREAD_SCENARIOS)
        post_text = fill_slots(scenario["post"], topic)
        parent_comment = fill_slots(scenario["parent_neut"], topic)
        parent_comment_id = f"{conv_id}_turn_1"
        turn_index = 2
        thread_context = f"[Turn 0 OP]: {post_text} | [Turn 1 UserA]: {parent_comment}"

    threat_label = assign_threat_label("ambiguous", text, rng)

    return SynRecord(
        id=record_id,
        text=text.strip(),
        primary_label="ambiguous",
        threat_label=threat_label,
        post_text=post_text,
        conversation_id=conv_id,
        turn_index=turn_index,
        parent_comment_id=parent_comment_id,
        parent_comment=parent_comment,
        thread_context=thread_context,
        target_message_id=record_id,
        tactic_labels=json.dumps([]),
        sarcasm_label="non_sarcastic",
        sentiment_label="neutral",
        generation_source=GENERATION_SOURCE,
        generator_model=GENERATOR_MODEL,
        generation_batch=batch_name,
        review_status=REVIEW_STATUS,
        quality_flags="context_dependent" if is_thread else "clean",
    )


def generate_threat_record(
    record_id: str, conv_id: str, is_thread: bool, batch_name: str,
    rng: random.Random, threat_type: str
) -> SynRecord:
    """Generate a message that is primarily a threat example."""
    topic = rng.choice(TOPICS)

    if threat_type == "direct_threat":
        raw = rng.choice(THREAT_MESSAGES["direct_threat"])
        # Direct threats can also be PA (implied warning) or non-PA (overt anger)
        primary = rng.choice(["passive_aggressive", "non_passive_aggressive"])
        tactic = "implied_blame" if primary == "passive_aggressive" else "none"
    elif threat_type == "implied_threat":
        raw = rng.choice(THREAT_MESSAGES["implied_threat"])
        primary = rng.choice(["passive_aggressive", "ambiguous"])
        tactic = "indirect_criticism" if primary == "passive_aggressive" else "none"
    else:  # ambiguous threat
        raw = rng.choice(THREAT_MESSAGES["ambiguous_threat"])
        primary = "ambiguous"
        tactic = "none"

    text = fill_slots(raw, topic)

    return SynRecord(
        id=record_id,
        text=text.strip(),
        primary_label=primary,
        threat_label=threat_type.replace("_threat", "_threat") if "threat" in threat_type else threat_type,
        post_text=None,
        conversation_id=conv_id,
        turn_index=0,
        parent_comment_id=None,
        parent_comment=None,
        thread_context=None,
        target_message_id=record_id,
        tactic_labels=json.dumps([tactic] if tactic != "none" else []),
        sarcasm_label="non_sarcastic",
        sentiment_label="negative",
        generation_source=GENERATION_SOURCE,
        generator_model=GENERATOR_MODEL,
        generation_batch=batch_name,
        review_status=REVIEW_STATUS,
        quality_flags="threat_example",
    )


def generate_no_threat_record(
    record_id: str, conv_id: str, batch_name: str, rng: random.Random
) -> SynRecord:
    """Generate a non-threatening message that looks threatening (figurative/hyperbolic)."""
    topic = rng.choice(TOPICS)
    raw = rng.choice(NO_THREAT_EXPRESSIONS)
    text = fill_slots(raw, topic)

    return SynRecord(
        id=record_id,
        text=text.strip(),
        primary_label="non_passive_aggressive",
        threat_label="no_threat",
        post_text=None,
        conversation_id=conv_id,
        turn_index=0,
        parent_comment_id=None,
        parent_comment=None,
        thread_context=None,
        target_message_id=record_id,
        tactic_labels=json.dumps([]),
        sarcasm_label="non_sarcastic",
        sentiment_label="negative",
        generation_source=GENERATION_SOURCE,
        generator_model=GENERATOR_MODEL,
        generation_batch=batch_name,
        review_status=REVIEW_STATUS,
        quality_flags="false_positive_threat_example",
    )


# ── Batch Generator ─────────────────────────────────────────────────────────────
def generate_batch(
    batch_cfg: dict,
    counts: dict,
    rng: random.Random,
    seen_hashes: set,
    seen_norms: set,
) -> tuple[list[SynRecord], list[dict]]:
    """Generate one batch of 2,500 records."""
    batch_name = batch_cfg["name"]
    start_num  = batch_cfg["start"]
    current_idx = start_num

    label_generators = {
        "passive_aggressive":     generate_pa_record,
        "non_passive_aggressive": generate_non_pa_record,
        "ambiguous":              generate_ambiguous_record,
    }

    accepted: list[SynRecord] = []
    rejected: list[dict] = []

    # Reserve ~5% of each batch for dedicated threat/no-threat examples
    threat_reserve = max(10, int(sum(counts.values()) * 0.03))
    no_threat_reserve = max(10, int(sum(counts.values()) * 0.02))

    # Generate primary label examples
    for label, target in counts.items():
        count = 0
        attempts = 0
        max_attempts = target * 15

        while count < target and attempts < max_attempts:
            attempts += 1
            is_thread = (count % 3 != 0)  # ~2/3 as thread messages
            conv_id = (f"conv_{batch_name}_{label[:3]}_{count//2:04d}"
                       if is_thread else f"standalone_{current_idx:05d}")
            rec_id = f"SYNPA-{current_idx:05d}"

            candidate = label_generators[label](
                record_id=rec_id,
                conv_id=conv_id,
                is_thread=is_thread,
                batch_name=batch_name,
                rng=rng,
            )

            ok, reason = is_valid(candidate, seen_hashes, seen_norms)
            if ok:
                accepted.append(candidate)
                count += 1
                current_idx += 1
            else:
                rejected.append({
                    "id": rec_id, "text": candidate.text,
                    "label": label, "reason": reason, "batch": batch_name
                })

        logger.info("[%s] %s: %d/%d accepted (%d attempts)",
                    batch_name, label, count, target, attempts)
        if count < target:
            logger.warning("[%s] Could not reach target for %s: got %d/%d",
                           batch_name, label, count, target)

    # Generate threat-specific examples (appended, replacing some no_threat slots)
    for threat_type in ["direct_threat", "implied_threat", "ambiguous"]:
        for _ in range(threat_reserve // 3):
            conv_id = f"threat_{batch_name}_{current_idx:05d}"
            rec_id  = f"SYNPA-{current_idx:05d}"
            if current_idx > batch_cfg["end"]:
                break
            candidate = generate_threat_record(
                rec_id, conv_id, False, batch_name, rng, threat_type
            )
            ok, reason = is_valid(candidate, seen_hashes, seen_norms)
            if ok:
                # Override threat label to be correct
                candidate.threat_label = (
                    threat_type if threat_type != "ambiguous"
                    else "ambiguous"
                )
                accepted.append(candidate)
                current_idx += 1

    # Generate no-threat false-positive examples
    for _ in range(no_threat_reserve):
        conv_id = f"nothreat_{batch_name}_{current_idx:05d}"
        rec_id  = f"SYNPA-{current_idx:05d}"
        if current_idx > batch_cfg["end"]:
            break
        candidate = generate_no_threat_record(rec_id, conv_id, batch_name, rng)
        ok, reason = is_valid(candidate, seen_hashes, seen_norms)
        if ok:
            accepted.append(candidate)
            current_idx += 1

    return accepted, rejected


# ── I/O ─────────────────────────────────────────────────────────────────────────
def save_csv(records: list[SynRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields(SynRecord)]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=field_names)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    logger.info("Saved %d records → %s", len(records), path)


def save_jsonl(records: list[SynRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("Saved %d records (JSONL) → %s", len(records), path)


# ── Validation Report ────────────────────────────────────────────────────────────
def validate_and_report(records: list[SynRecord], out_path: Path) -> dict:
    ids = [r.id for r in records]
    texts = [r.text for r in records]
    labels = [r.primary_label for r in records]
    threats = [r.threat_label for r in records]

    from collections import Counter
    label_dist  = dict(Counter(labels))
    threat_dist = dict(Counter(threats))
    tactic_dist: dict[str, int] = {}
    for r in records:
        for t in json.loads(r.tactic_labels or "[]"):
            tactic_dist[t] = tactic_dist.get(t, 0) + 1

    issues = []

    # Unique IDs
    if len(set(ids)) != len(ids):
        issues.append(f"DUPLICATE IDs detected: {len(ids) - len(set(ids))} duplicates")

    # Empty text
    empty = sum(1 for t in texts if not t or len(t.strip()) < 2)
    if empty:
        issues.append(f"Empty/short texts: {empty}")

    # Label coverage
    for lbl in ("passive_aggressive", "non_passive_aggressive", "ambiguous"):
        if lbl not in label_dist:
            issues.append(f"Missing primary label: {lbl}")

    report = {
        "total_records":  len(records),
        "unique_ids":     len(set(ids)),
        "label_dist":     label_dist,
        "threat_dist":    threat_dist,
        "tactic_dist":    tactic_dist,
        "issues":         issues,
        "thread_records": sum(1 for r in records if r.turn_index > 0),
        "standalone":     sum(1 for r in records if r.turn_index == 0),
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate 10,000 synthetic PA social-media records")
    parser.add_argument("--batch", type=int, choices=[1, 2, 3, 4], help="Generate a single batch (1-4)")
    parser.add_argument("--all",   action="store_true", help="Generate all 4 batches")
    parser.add_argument("--merge", action="store_true", help="Merge existing batch files")
    args = parser.parse_args()

    if not args.batch and not args.all and not args.merge:
        parser.print_help()
        sys.exit(0)

    data_dir    = Path("data/synthetic")
    reports_dir = Path("reports")
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    # Shared dedup state across all batches
    seen_hashes: set = set()
    seen_norms:  set = set()

    batch_files: list[Path] = []

    batches_to_run = []
    if args.all:
        batches_to_run = list(range(4))
    elif args.batch:
        batches_to_run = [args.batch - 1]

    for i in batches_to_run:
        cfg    = BATCH_CONFIGS[i]
        counts = BATCH_COUNTS[i]
        logger.info("=== Generating %s (IDs %d–%d) ===",
                    cfg["name"], cfg["start"], cfg["end"])

        records, rejected = generate_batch(cfg, counts, rng, seen_hashes, seen_norms)

        batch_path = data_dir / f"{cfg['name']}_SYNPA_{cfg['start']:05d}_{cfg['end']:05d}.csv"
        save_csv(records, batch_path)
        batch_files.append(batch_path)

        # Per-batch validation
        val_path = reports_dir / f"synthetic_validation_{cfg['name']}.json"
        report   = validate_and_report(records, val_path)
        logger.info("[%s] Validation: %d records, labels=%s, threats=%s, issues=%s",
                    cfg["name"], report["total_records"],
                    report["label_dist"], report["threat_dist"], report["issues"])

        # Save rejected log
        if rejected:
            rej_path = reports_dir / f"synthetic_rejected_{cfg['name']}.json"
            with open(rej_path, "w") as f:
                json.dump(rejected, f, indent=2)
            logger.info("[%s] %d records rejected — saved to %s",
                        cfg["name"], len(rejected), rej_path)

    # Merge if requested or if all batches were generated
    if args.merge or args.all:
        logger.info("=== Merging batch files ===")
        import csv as csv_mod
        all_records: list[dict] = []
        for bp in sorted(data_dir.glob("batch*_SYNPA_*.csv")):
            with open(bp, encoding="utf-8", newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    all_records.append(row)

        if not all_records:
            logger.error("No batch files found to merge")
            return

        merged_csv  = data_dir / "pa_synthetic_merged.csv"
        merged_jsonl = data_dir / "pa_synthetic_merged.jsonl"

        field_names = list(all_records[0].keys())
        with open(merged_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(all_records)
        logger.info("Merged CSV: %d records → %s", len(all_records), merged_csv)

        with open(merged_jsonl, "w", encoding="utf-8") as fh:
            for row in all_records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Merged JSONL: %d records → %s", len(all_records), merged_jsonl)

        # Final validation report
        from collections import Counter
        final_labels = [r["primary_label"] for r in all_records]
        final_threats = [r["threat_label"] for r in all_records]
        final_report = {
            "total":          len(all_records),
            "unique_ids":     len(set(r["id"] for r in all_records)),
            "label_dist":     dict(Counter(final_labels)),
            "threat_dist":    dict(Counter(final_threats)),
            "batches_merged": [str(bp.name) for bp in sorted(data_dir.glob("batch*_SYNPA_*.csv"))],
        }
        rep_path = reports_dir / "synthetic_generation_report.json"
        with open(rep_path, "w") as f:
            json.dump(final_report, f, indent=2)
        logger.info("Final report: %s", final_report)


if __name__ == "__main__":
    main()
