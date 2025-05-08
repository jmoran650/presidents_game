#!/usr/bin/env python3
"""
Presidents and Assholes – terminal version
Author: ChatGPT (OpenAI o3)
First release:  2025-05-02
Updates:
  • 2025-05-02 – proper trick logic + auto-pass
  • 2025-05-02 – *new*: 0.8-second delay after each bot turn

• Standard 52-card deck + 2 Jokers.
• Rank order: 3 < … < A < 2 < Joker.
• Lead may be single/double/triple/quadruple; followers must match size and beat rank.
• A trick ends only when every remaining player passes after the last play.
• The last player to lay cards leads the next trick.
• Auto-pass when no legal move exists.
• 0.8-second delay after every bot move for smoother pacing.
• Roles (President, … Asshole) and card exchange handled between hands.
• Supports 2–8 players; any subset may be bots.
"""

import random
import sys
import time
from typing import List, Tuple, Optional, Dict

# ─────────────────────── CONFIG ────────────────────────
BOT_DELAY = 0.8       # seconds to pause after each bot action


# ────────────────────────── CARD / DECK ──────────────────────────
class Card:
    SUITS = ["♠", "♥", "♦", "♣", "🃏"]
    RANKS = ["3", "4", "5", "6", "7", "8", "9", "10",
             "J", "Q", "K", "A", "2", "JOKER"]
    ORDER = {r: i for i, r in enumerate(RANKS)}

    def __init__(self, rank: str, suit: str = ""):
        self.rank = rank
        self.suit = suit

    @classmethod
    def standard_deck(cls) -> List["Card"]:
        deck = [Card(rank, suit)
                for suit in cls.SUITS[:-1]
                for rank in cls.RANKS[:-2]]
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
    def __init__(self, name: str, is_human: bool):
        self.name = name
        self.is_human = is_human
        self.hand: List[Card] = []
        self.out_position: Optional[int] = None

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

    # -------- turn logic --------
    def choose_action(
        self,
        lead_size: int,
        current_strength: int,
        can_play: bool
    ) -> Tuple[str, List[int]]:
        if not can_play:
            return ("pass", [])
        return (self._prompt_human if self.is_human else self._bot_action)(
            lead_size, current_strength
        )

    # ---- human interface ----
    def _prompt_human(
        self, lead_size: int, current_strength: int
    ) -> Tuple[str, List[int]]:
        while True:
            self._display_hand()
            cmd = input(
                f"{self.name}, command (play N R | pass | help | quit): "
            ).strip().lower()
            if cmd == "quit":
                sys.exit(0)
            if cmd == "help":
                print("Commands:\n"
                      "  play N R   – play N cards of rank R "
                      "(e.g. play 2 5, play 1 joker)\n"
                      "  pass       – pass for this trick\n"
                      "  quit       – leave game")
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
                rank_in = toks[2].upper()
                alias = {"JK": "JOKER", "J": "J", "Q": "Q",
                         "K": "K", "A": "A", "JOKER": "JOKER"}
                rank = alias.get(rank_in, rank_in)
                if rank not in Card.RANKS:
                    print("❌  Unknown rank.")
                    continue
                idxs = [i for i, c in enumerate(self.hand) if c.rank == rank]
                if len(idxs) < count:
                    print(f"❌  You have only {len(idxs)} of {rank}.")
                    continue
                chosen = idxs[:count]
                if not legal_play([self.hand[i] for i in chosen],
                                  lead_size, current_strength):
                    print("❌  Illegal play right now.")
                    continue
                return ("play", chosen)
            print("❌  Unknown command.")

    def _display_hand(self):
        print("\nYour hand:")
        for i, card in enumerate(self.hand):
            print(f"  {i:2d}: {card}")
        print()

    # ---- simple bot ----
    def _bot_action(
        self, lead_size: int, current_strength: int
    ) -> Tuple[str, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(self.hand):
            groups.setdefault(c.strength(), []).append(i)
        candidates: List[Tuple[int, List[int]]] = []
        for strength, idxs in groups.items():
            if lead_size == 0:
                for sz in range(1, min(4, len(idxs)) + 1):
                    candidates.append((strength, idxs[:sz]))
            elif len(idxs) >= lead_size:
                candidates.append((strength, idxs[:lead_size]))
        candidates.sort(key=lambda t: (t[0], len(t[1])))
        for strength, idxs in candidates:
            if legal_play([self.hand[i] for i in idxs],
                          lead_size, current_strength):
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
    def __init__(self, names: List[str], human_idx: List[int]):
        self.players = [Player(n, i in human_idx) for i, n in enumerate(names)]
        self.round_num = 0

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
            can_play = has_legal_play(player.hand, lead_size, current_strength)
            self._show_table(lead_size, current_strength, leader_idx)

            # auto-pass branch
            if not can_play:
                print(f"{player.name} passes.")
                if not player.is_human:
                    time.sleep(BOT_DELAY)
                passes_in_row += 1
                if passes_in_row == active_now - 1:
                    print("---- Trick ends ----")
                    lead_size = 0
                    current_strength = -1
                    passes_in_row = 0
                    turn_idx = leader_idx
                    continue
                turn_idx = (turn_idx + 1) % len(self.players)
                continue

            action, idxs = player.choose_action(lead_size, current_strength, can_play)

            if action == "pass":
                print(f"{player.name} passes.")
                passes_in_row += 1
                if not player.is_human:
                    time.sleep(BOT_DELAY)
                if passes_in_row == active_now - 1:
                    print("---- Trick ends ----")
                    lead_size = 0
                    current_strength = -1
                    passes_in_row = 0
                    turn_idx = leader_idx
                    continue
            else:
                cards = [player.hand[i] for i in idxs]
                player.remove_cards(cards)
                print(f"{player.name} plays: {' '.join(map(str, cards))}")
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
                    print(f"🏆  {player.name} is out! "
                          f"({player.out_position+1}{ordinal_suffix(player.out_position+1)})")

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
            pos = (f"({p.out_position+1}{ordinal_suffix(p.out_position+1)})"
                   if p.out_position is not None else "")
            arrow = "←" if self.players.index(p) == leader_idx and lead_size else " "
            print(f"{arrow} {p.name:<12} {pos:<4} cards:{len(p.hand):2d}")
        if lead_size:
            print(f"\nCurrent trick  – size {lead_size}, "
                  f"rank above {Card.RANKS[current_strength]}")
        else:
            print("\nNew trick – no cards on table.")
        print("=" * 46 + "\n")

    def _announce_roles(self):
        ordered = sorted(self.players, key=lambda p: p.out_position)
        labels = ["President", "Vice-President"] + \
                 ["Neutral"] * (len(self.players) - 4) + \
                 ["Vice-Asshole", "Asshole"]
        print("\nRoles for next round:")
        for pl, lab in zip(ordered, labels):
            print(f"  {lab:<15} – {pl.name}")
        print("-" * 46)


# ────────────────────────── ENTRY ────────────────────────────────
def prompt_players() -> Tuple[List[str], List[int]]:
    while True:
        try:
            n = int(input("Number of players (2-8)? "))
            if 2 <= n <= 8:
                break
        except ValueError:
            pass
        print("Choose a number from 2-8.")
    humans, names = [], []
    for i in range(n):
        default = "You" if i == 0 else f"Bot{i}"
        name = input(f"Name for player {i+1} [{default}]: ").strip() or default
        names.append(name)
        if i == 0:
            is_human = input(f"Is {name} human? [Y/n] ").strip().lower() != "n"
        else:
            is_human = input(f"Is {name} human? [y/N] ").strip().lower() == "y"
        if is_human:
            humans.append(i)
    if not humans:
        print("Making first player human so the game is interactive.")
        humans.append(0)
    return names, humans


if __name__ == "__main__":
    nms, hum = prompt_players()
    Game(nms, hum).start()