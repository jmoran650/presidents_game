#!/usr/bin/env python3
"""
Presidents and Assholes – terminal version (LLM-enhanced)
Author: ChatGPT (OpenAI o3)
First release:  2025-05-02
Updates:
  • 2025-05-02 – proper trick logic + auto-pass
  • 2025-05-02 – *new*: 0.8‑second delay after each bot turn
  • 2025-05-08 – optional GPT‑powered bots via OpenAI API
  • 2025-05-08 – ⚡ Every non‑human player now powered by your choice of LLM provider

Key points
~~~~~~~~~~
• Standard 52‑card deck + 2 Jokers. Rank order: 3 < … < A < 2 < Joker.
• Lead may be single/double/triple/quadruple. Followers must match size & beat rank.
• A trick ends only when every remaining player passes after the last play.
• 0.8‑second delay after each bot move for smoother pacing.
• All roles (President → Asshole) and card exchange handled between hands.
• Supports 2–8 players; *zero* human players is allowed.
• **Every** non‑human player delegates its turn to an LLM — choose *openai*,
  *anthropic* or *google/gemini* when setting up players.
• Each LLM receives the *entire match history* and *how‑to‑play rules* in the
  prompt and must reply with reasoning inside `<thinking>` tags and its final
  command inside `<action>` tags (either `pass` or `play <COUNT> <RANK>`).
"""

from __future__ import annotations
from google import genai
from openai import OpenAI
import anthropic
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()                # automatically loads keys from .env


# ─────────────────────── CONFIG ────────────────────────
BOT_DELAY = 0.8           # seconds to pause after each bot action
HISTORY_LIMIT = 120       # max lines of history sent to an LLM
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
ANTHROPIC_TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2"))
GOOGLE_TEMPERATURE = float(os.getenv("GOOGLE_TEMPERATURE", "0.2"))
GPT_TIMEOUT = 20  # seconds

SUPPORTED_PROVIDERS = {"openai", "anthropic", "google", "gemini"}

# ────────────────────────── LLM HELPERS ──────────────────────────

def build_prompt(history: List[str], hand: List[str], lead_size: int, current_rank: str) -> str:
    """Build a prompt with full match history + hand + instructions."""
    rules = (
        "You are playing the card game Presidents & Assholes.\n\n"
        "Rank order: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 < JOKER.\n"
        "A Joker can *only* be played when the lead size is 1.\n"
        "Followers must play the same number of cards as the lead and beat the previous rank.\n"
        "If no legal move exists you *must* pass.\n\n"
        "*Return your full chain of thought in <thinking> tags.*\n"
        "*Return only your final command (pass | play <COUNT> <RANK>) inside <action> tags.*\n"
    )
    table_state = (
        "Current trick: New trick (no cards on table)" if lead_size == 0 else
        f"Current trick: size {lead_size}, rank above {current_rank}"
    )
    hand_str = " ".join(hand)
    hist_lines = "\n".join(history[-HISTORY_LIMIT:]) or "(No previous actions.)"
    return (
        f"{rules}\n"
        f"=== Your hand ===\n{hand_str}\n\n"
        f"=== Table ===\n{table_state}\n\n"
        f"=== Match history (oldest first) ===\n{hist_lines}\n\n"
        "Assistant:"
    )


def call_llm(prompt: str, provider: str = DEFAULT_PROVIDER) -> str:
    """Dispatch the prompt to the selected provider and return raw text."""
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if provider in {"openai","chatgpt"}:
      
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=OPENAI_TEMPERATURE,
            timeout=GPT_TIMEOUT,
        )
        return resp.choices[0].message.content.strip()

    if provider in {"google","gemini"}:


        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=GOOGLE_MODEL,
            contents=prompt,
        )
        return resp.text.strip()

    if provider in {"anthropic","claude"}:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.completions.create(
            model=ANTHROPIC_MODEL,
            prompt=prompt,
            temperature=ANTHROPIC_TEMPERATURE,
            max_tokens_to_sample=1024
        )
        return resp.completion.strip()

    raise ValueError(f"Provider branch not handled: {provider}")


# ────────────────────────── CARD / DECK ──────────────────────────


class Card:
    SUITS = ["♠", "♥", "♦", "♣", "🃏"]
    RANKS = [
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "J",
        "Q",
        "K",
        "A",
        "2",
        "JOKER",
    ]
    ORDER = {r: i for i, r in enumerate(RANKS)}

    def __init__(self, rank: str, suit: str = ""):
        self.rank = rank
        self.suit = suit

    @classmethod
    def standard_deck(cls) -> List["Card"]:
        deck = [Card(rank, suit) for suit in cls.SUITS[:-1] for rank in cls.RANKS[:-2]]
        deck.extend(Card("JOKER", cls.SUITS[-1]) for _ in range(2))
        deck.extend(Card("2", suit) for suit in cls.SUITS[:-1])
        return deck

    def strength(self) -> int:
        return Card.ORDER[self.rank]

    def __lt__(self, other: "Card"):
        return self.strength() < other.strength()

    def __str__(self):
        return f"{self.rank}{self.suit}"

    __repr__ = __str__


# ────────────────────────── PLAYER ───────────────────────────────


class Player:
    def __init__(self, name: str, is_human: bool, provider: str | None = None):
        self.name = name
        self.is_human = is_human
        self.provider = (provider or DEFAULT_PROVIDER).lower()
        self.hand: List[Card] = []
        self.out_position: Optional[int] = None

    # ---------------- helpers ----------------
    def sort_hand(self):
        self.hand.sort()

    def remove_cards(self, cards: List[Card]):
        for c in cards:
            self.hand.remove(c)

    def receive_cards(self, cards: List[Card]):
        self.hand.extend(cards)
        self.sort_hand()

    def highest_cards(self, n: int) -> List[Card]:
        return sorted(self.hand, key=lambda c: c.strength(), reverse=True)[:n]

    def lowest_cards(self, n: int) -> List[Card]:
        return sorted(self.hand, key=lambda c: c.strength())[:n]

    # ---------------- turn logic ----------------
    def choose_action(
        self,
        lead_size: int,
        current_strength: int,
        history: List[str],
    ) -> Tuple[str, List[int]]:
        can_play = has_legal_play(self.hand, lead_size, current_strength)
        if not can_play:
            return ("pass", [])
        if self.is_human:
            return self._prompt_human(lead_size, current_strength)
        return self._llm_action(lead_size, current_strength, history)

    # ---- human interface ----
    def _prompt_human(self, lead_size: int, current_strength: int) -> Tuple[str, List[int]]:
        while True:
            self._display_hand()
            cmd = (
                input(f"{self.name}, command (play N R | pass | help | quit): ")
                .strip()
                .lower()
            )
            if cmd == "quit":
                sys.exit(0)
            if cmd == "help":
                print(
                    "Commands:\n"
                    "  play N R   – play N cards of rank R (e.g. play 2 5)\n"
                    "  pass       – pass for this trick\n"
                    "  quit       – leave game"
                )
                continue
            if cmd.startswith("pass"):
                return ("pass", [])
            if cmd.startswith("play"):
                toks = cmd.split()
                if len(toks) != 3:
                    print("❌  Usage: play N R")
                    continue
                try:
                    count = int(toks[1])
                except ValueError:
                    print("❌  N must be 1-4.")
                    continue
                if not 1 <= count <= 4:
                    print("❌  N must be 1-4.")
                    continue
                rank = toks[2].upper()
                if rank not in Card.RANKS:
                    print("❌  Unknown rank.")
                    continue
                idxs = [i for i, c in enumerate(self.hand) if c.rank == rank][:count]
                if len(idxs) < count:
                    print(f"❌  You have only {len(idxs)} of {rank}.")
                    continue
                if not legal_play(
                    [self.hand[i] for i in idxs], lead_size, current_strength
                ):
                    print("❌  Illegal play right now.")
                    continue
                return ("play", idxs)
            print("❌  Unknown command.")

    def _display_hand(self):
        print("\nYour hand:")
        for i, card in enumerate(self.hand):
            print(f"  {i:2d}: {card}")
        print()

    # ---- LLM bot ----
    def _llm_action(
        self, lead_size: int, current_strength: int, history: List[str]
    ) -> Tuple[str, List[int]]:
        current_rank = Card.RANKS[current_strength] if lead_size else "None"
        prompt = build_prompt(
            history,
            [str(c) for c in self.hand],
            lead_size,
            current_rank,
        )
        try:
            raw = call_llm(prompt, self.provider)
        except Exception as exc: 
            print(f"⚠️  {self.name} ({self.provider}) LLM error → passing. Details: {exc}")
            return ("pass", [])

        # Parse <action>...</action>
        match = re.search(r"<action>(.*?)</action>", raw, re.I | re.S)
        if not match:
            print(f"⚠️  {self.name} returned malformed response → passing. Raw: {raw}")
            return ("pass", [])
        action_line = match.group(1).strip().lower()
        if action_line == "pass":
            return ("pass", [])
        if action_line.startswith("play"):
            toks = action_line.split()
            if len(toks) != 3:
                return ("pass", [])
            try:
                count = int(toks[1])
            except ValueError:
                return ("pass", [])
            rank = toks[2].upper()
            if rank not in Card.RANKS:
                return ("pass", [])
            idxs = [i for i, c in enumerate(self.hand) if c.rank == rank][:count]
            if len(idxs) != count or not legal_play(
                [self.hand[i] for i in idxs], lead_size, current_strength
            ):
                return ("pass", [])
            return ("play", idxs)
        return ("pass", [])


# ─────────────────── PLAY / TRICK HELPERS ────────────────────────

def legal_play(cards: List[Card], lead_size: int, current_strength: int) -> bool:
    if not cards or len({c.rank for c in cards}) != 1:
        return False
    size = len(cards)
    strength = cards[0].strength()
    if lead_size == 0:
        return True
    if size != lead_size:
        return False
    if strength == Card.ORDER["JOKER"]:
        return lead_size == 1
    if strength == Card.ORDER["2"]:
        return current_strength < strength
    return strength > current_strength


def has_legal_play(hand: List[Card], lead_size: int, current_strength: int) -> bool:
    if not hand:
        return False
    if lead_size == 0:
        return True
    groups: Dict[str, int] = {}
    for c in hand:
        groups[c.rank] = groups.get(c.rank, 0) + 1
    for rank, cnt in groups.items():
        if cnt < lead_size:
            continue
        strength = Card.ORDER[rank]
        if rank == "JOKER" and lead_size == 1:
            return True
        if rank == "2" and current_strength < strength:
            return True
        if strength > current_strength:
            return True
    return False


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ────────────────────────── GAME LOOP ────────────────────────────


class Game:
    def __init__(self, names: List[str], human_idx: List[int], providers: List[str]):
        self.players = [
            Player(name=n, is_human=(i in human_idx), provider=providers[i])
            for i, n in enumerate(names)
        ]
        self.round_num = 0
        self.history: List[str] = []  # textual log for LLM prompts

    def start(self):
        print("\n=========== Presidents & Assholes ===========")
        while True:
            self.round_num += 1
            print(f"\n--- Round {self.round_num} ---")
            self._deal()
            if self.round_num > 1:
                self._exchange_cards()
            self._play_hand()
            self._announce_roles()

    # ---- setup ----
    def _deal(self):
        deck = Card.standard_deck()
        random.shuffle(deck)
        for p in self.players:
            p.hand.clear()
            p.out_position = None
        while deck:
            for p in self.players:
                if deck:
                    p.hand.append(deck.pop())
                else:
                    break
        for p in self.players:
            p.sort_hand()
        self.history.append(
            f"=== Round {self.round_num} dealt ==="
        )

    def _exchange_cards(self):
        ranked = sorted(self.players, key=lambda p: p.out_position)
        pres, ass = ranked[0], ranked[-1]
        vp = ranked[1] if len(ranked) >= 2 else None
        va = ranked[-2] if len(ranked) >= 2 else None

        def swap(giver: Player, rec: Player, num: int, high: bool):
            cards = giver.highest_cards(num) if high else giver.lowest_cards(num)
            giver.remove_cards(cards)
            rec.receive_cards(cards)
        
        swap(ass, pres, 2, high=True)
        swap(pres, ass, 2, high=False)
        if vp and va:
            swap(va, vp, 1, high=True)
            swap(vp, va, 1, high=False)
        self.history.append("Card exchange complete.")
        print("\nCard exchange complete.")

    # ---- main round ----
    def _play_hand(self):
        lead_size = 0
        current_strength = -1
        passes_in_row = 0
        leader_idx = self._starting_index()
        turn_idx = leader_idx
        finished = 0
        rank_order = 0

        while finished < len(self.players) - 1:
            player = self.players[turn_idx]
            if player.out_position is not None:
                turn_idx = (turn_idx + 1) % len(self.players)
                continue

            active_now = sum(1 for p in self.players if p.out_position is None)
            self._show_table(lead_size, current_strength, leader_idx)

            action, idxs = player.choose_action(
                lead_size, current_strength, self.history
            )

            if action == "pass":
                print(f"{player.name} passes.")
                self.history.append(f"{player.name} passes")
                if not player.is_human:
                    time.sleep(BOT_DELAY)
                passes_in_row += 1
                if passes_in_row == active_now - 1:
                    print("---- Trick ends ----")
                    self.history.append("---- Trick ends ----")
                    lead_size = 0
                    current_strength = -1
                    passes_in_row = 0
                    turn_idx = leader_idx
                    continue
            else:
                cards = [player.hand[i] for i in idxs]
                player.remove_cards(cards)
                played_str = " ".join(map(str, cards))
                print(f"{player.name} plays: {played_str}")
                self.history.append(f"{player.name} plays: {played_str}")
                if not player.is_human:
                    time.sleep(BOT_DELAY)
                lead_size = len(cards)
                current_strength = cards[0].strength()
                leader_idx = turn_idx
                passes_in_row = 0
                if not player.hand:
                    player.out_position = rank_order
                    rank_order += 1
                    finished += 1
                    print(
                        f"🏆  {player.name} is out! ("
                        f"{player.out_position + 1}{ordinal_suffix(player.out_position + 1)})"
                    )
                    self.history.append(
                        f"{player.name} out as {player.out_position + 1}{ordinal_suffix(player.out_position + 1)}"
                    )

            turn_idx = (turn_idx + 1) % len(self.players)

        for p in self.players:
            if p.out_position is None:
                p.out_position = rank_order

    def _starting_index(self) -> int:
        if self.round_num == 1:
            return 0
        for i, p in enumerate(self.players):
            if p.out_position == len(self.players) - 1:
                return i
        return 0

    # ---- UI ----
    def _show_table(self, lead_size, current_strength, leader_idx):
        print("\n" + "=" * 46)
        for p in self.players:
            pos = (
                f"({p.out_position + 1}{ordinal_suffix(p.out_position + 1)})"
                if p.out_position is not None
                else ""
            )
            arrow = "←" if self.players.index(p) == leader_idx and lead_size else " "
            prov = f"[{p.provider}]" if not p.is_human else "(human)"
            print(f"{arrow} {p.name:<12} {pos:<4} {prov:<10} cards:{len(p.hand):2d}")
        if lead_size:
            print(
                f"\nCurrent trick  – size {lead_size}, "
                f"rank above {Card.RANKS[current_strength]}"
            )
        else:
            print("\nNew trick – no cards on table.")
        print("=" * 46 + "\n")

    def _announce_roles(self):
        ordered = sorted(self.players, key=lambda p: p.out_position)
        labels = (
            ["President", "Vice-President"]
            + ["Neutral"] * (len(self.players) - 4)
            + ["Vice-Asshole", "Asshole"]
        )
        print("\nRoles for next round:")
        for pl, lab in zip(ordered, labels):
            print(f"  {lab:<15} – {pl.name}")
        print("-" * 46)


# ────────────────────────── ENTRY ────────────────────────────────


def prompt_players() -> Tuple[List[str], List[int], List[str]]:
    while True:
        try:
            n = int(input("Number of players (2-8)? "))
            if 2 <= n <= 8:
                break
        except ValueError:
            pass
        print("Choose a number from 2-8.")
    human_idxs, names, providers = [], [], []
    for i in range(n):
        default_name = f"Player{i+1}"
        name = input(f"Name for player {i + 1} [{default_name}]: ").strip() or default_name
        is_human_prompt = "Is this player human? [y/N] " if i else "Is this player human? [Y/n] "
        default_human = i == 0  # make first player human by default
        is_human_input = input(is_human_prompt).strip().lower()
        is_human = (is_human_input != "n") if default_human else (is_human_input == "y")
        provider = DEFAULT_PROVIDER
        if not is_human:
            provider_in = input(
                "LLM provider for this bot [openai / anthropic / gemini] "
                f"[{DEFAULT_PROVIDER}]: "
            ).strip().lower()
            if provider_in:
                provider = provider_in
        names.append(name)
        providers.append(provider)
        if is_human:
            human_idxs.append(i)
    return names, human_idxs, providers


if __name__ == "__main__":
    nms, hum_idxs, provs = prompt_players()
    game = Game(nms, hum_idxs, provs)
    game.start()
