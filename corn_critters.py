# Corn & Critters — SIMPLE with Market Scarcity (rotating order, signed cashflows) + VP tracking
# ----------------------------------------------------------------------------------------------
# - Market scarcity + live market table.
# - Buying reduces market supply; selling increases it (within turn).
# - Turn order rotates each Night.
# - Affordability guard keeps EOT bushels >= 1 (clamps buys).
# - Per-player Victory Points: Starting VP, VP earned this turn (input), End-of-Turn VP (preview).
# - VP committed on Run Night and logged to History.
# - Market probabilities scale dynamically with player count and deck composition.

import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List
import math
from collections import deque

st.set_page_config(page_title="Corn & Critters — Simple (Market + VP)", layout="wide")

# ---------- Data model ----------
@dataclass
class Animal:
    name: str
    feed_per_turn: int
    cost: int  # integer costs for clean rounding

# >>> DEFAULT ANIMALS (your chosen defaults) <<<
DEFAULT_ANIMALS: List[Animal] = [
    Animal("Chicken",  1,  6),
    Animal("Pig",      2,  9),
    Animal("Cow",      2, 12),
    Animal("Bison",    7, 40),   # Updated: mid-tier exotic (7 feed, 40 cost)
    Animal("Elephant",10, 50),   # high-tier exotic
]

# ---------- Default game config (session-driven) ----------
DEFAULT_CFG = {
    "total_turns": 15,
    "mean_growth": 0.10,
    "noise_sd": 0.18,
    "clip_low": -0.25,
    "clip_high": 0.70,
    "rng_seed": 1,
}

# ---------- Tapered probability helpers & market config ----------
def probs_tapered(expected: float, slots: int = 6, taper: float = 0.75) -> List[float]:
    """
    Return a non-increasing list of per-slot probabilities whose sum ~= expected.
    Earlier slots are more likely (geometric taper). Each probability is clipped to [0,1].
    expected is clamped to [0, slots].
    """
    expected = max(0.0, min(float(slots), float(expected)))
    if slots <= 0 or expected == 0.0:
        return [0.0] * max(0, slots)

    # geometric weights (1, taper, taper^2, ...)
    w = [taper ** i for i in range(slots)]

    # binary search scale so sum(min(1, scale*w_i)) ~= expected
    lo, hi = 0.0, 1e6
    for _ in range(50):
        mid = (lo + hi) / 2
        s = sum(min(1.0, mid * wi) for wi in w)
        if s >= expected:
            hi = mid
        else:
            lo = mid
    scale = (lo + hi) / 2
    probs = [min(1.0, scale * wi) for wi in w]

    # small normalization to hit target closely (still clipped to 1.0)
    total = sum(probs)
    if total > 0:
        adjust = expected / total
        probs = [min(1.0, p * adjust) for p in probs]
    return probs

# ---- Market config (dynamic scaling based on player count) ----
BASE_MARKET_GENEROSITY = 1.4  # Tune this: 1.0 = tight, 1.4 = balanced, 2.0 = abundant
PURCHASES_PER_PLAYER_PER_TURN = 1.0  # Expected animal purchases per player per turn

# Deck distribution from analysis:
# Chickens: 103 needed (47.2% of common animal demand)
# Pigs: 63 needed (28.9% of common animal demand)
# Cows: 52 needed (23.9% of common animal demand)
DECK_DISTRIBUTION = {
    "Chicken": 0.472,
    "Pig": 0.289,
    "Cow": 0.239,
}

SLOTS_COMMON = 10      # number of independent "slots" for Chicken/Pig/Cow
TAPER = 0.75          # 1.0 = flat; lower -> more front-loaded

def calculate_expected_per_turn(num_players: int) -> Dict[str, float]:
    """
    Calculate market supply dynamically based on:
    1. Number of players (more players = more demand)
    2. Deck composition (Chickens are 46% of demand, etc.)
    
    Returns expected animals per turn for each common type.
    """
    if num_players < 1:
        num_players = 1  # Safety: minimum 1 player
    
    total_demand = num_players * PURCHASES_PER_PLAYER_PER_TURN
    market_supply = total_demand * BASE_MARKET_GENEROSITY
    
    return {
        animal: market_supply * proportion
        for animal, proportion in DECK_DISTRIBUTION.items()
    }

# ---------- Market cooldowns for exotics ----------
BISON_COOLDOWN = 1       # must wait 1 full turn after spawn before another spawn
ELEPHANT_COOLDOWN = 2    # must wait 2 full turns after spawn

# ---------- Persistent Storage Functions ----------
def save_game_state():
    """Save current game state to browser storage."""
    try:
        state_to_save = {
            "turn": st.session_state.turn,
            "players": st.session_state.players,
            "player_color": st.session_state.player_color,
            "bushels": st.session_state.bushels,
            "herds": st.session_state.herds,
            "turn_start_bushels": st.session_state.turn_start_bushels,
            "turn_start_herd": st.session_state.turn_start_herd,
            "vp": st.session_state.vp,
            "turn_start_vp": st.session_state.turn_start_vp,
            "vp_earned": st.session_state.vp_earned,
            "history": st.session_state.history,
            "last_growth_rate": st.session_state.last_growth_rate,
            "cfg": st.session_state.cfg,
            "market_supply": st.session_state.market_supply,
            "market_draw_turn": st.session_state.market_draw_turn,
            "last_spawn_turn": st.session_state.last_spawn_turn,
        }
        # Store in session state with a special key
        st.session_state._game_backup = state_to_save
    except Exception as e:
        # Silent fail - don't break the game if storage fails
        pass

def load_game_state():
    """Load saved game state from browser storage."""
    try:
        if "_game_backup" in st.session_state and st.session_state._game_backup:
            saved = st.session_state._game_backup
            st.session_state.turn = saved.get("turn", 1)
            st.session_state.players = saved.get("players", [])
            st.session_state.player_color = saved.get("player_color", {})
            st.session_state.bushels = saved.get("bushels", {})
            st.session_state.herds = saved.get("herds", {})
            st.session_state.turn_start_bushels = saved.get("turn_start_bushels", {})
            st.session_state.turn_start_herd = saved.get("turn_start_herd", {})
            st.session_state.vp = saved.get("vp", {})
            st.session_state.turn_start_vp = saved.get("turn_start_vp", {})
            st.session_state.vp_earned = saved.get("vp_earned", {})
            st.session_state.history = saved.get("history", [])
            st.session_state.last_growth_rate = saved.get("last_growth_rate", None)
            st.session_state.cfg = saved.get("cfg", DEFAULT_CFG.copy())
            st.session_state.market_supply = saved.get("market_supply", {})
            st.session_state.market_draw_turn = saved.get("market_draw_turn", 0)
            st.session_state.last_spawn_turn = saved.get("last_spawn_turn", {"Bison": -10, "Elephant": -10})
            return True
    except Exception:
        pass
    return False

# ---------- State reset ----------
def reset_state():
    st.session_state.turn = 1
    st.session_state.players: List[str] = []
    st.session_state.player_color: Dict[str, str] = {}  # hex color per player

    # Bushels / herds
    st.session_state.bushels: Dict[str, int] = {}
    st.session_state.herds: Dict[str, Dict[str, int]] = {}
    st.session_state.turn_start_bushels: Dict[str, int] = {}
    st.session_state.turn_start_herd: Dict[str, Dict[str, int]] = {}

    # Value Points
    st.session_state.vp: Dict[str, int] = {}               # Start-of-turn VP (committed)
    st.session_state.turn_start_vp: Dict[str, int] = {}    # Baseline VP this turn
    st.session_state.vp_earned: Dict[str, int] = {}        # Input per player this turn

    # History
    st.session_state.history: List[dict] = []
    st.session_state.last_growth_rate = None

    # RNG & animals
    st.session_state.cfg = st.session_state.get("cfg", DEFAULT_CFG.copy())
    st.session_state.rng_seed = st.session_state.cfg.get("rng_seed", DEFAULT_CFG["rng_seed"])
    st.session_state.animals = [asdict(a) for a in DEFAULT_ANIMALS]

    # Market state
    st.session_state.market_supply: Dict[str, int] = {}
    st.session_state.market_draw_turn = 0
    
    # cooldown trackers (so we can space appearances)
    st.session_state.last_spawn_turn = {"Bison": -10, "Elephant": -10}
    
    # Clear backup
    st.session_state._game_backup = None

# Initialize
if "turn" not in st.session_state:
    # Try to load saved game first
    if not load_game_state():
        # No saved game, start fresh
        reset_state()
if "cfg" not in st.session_state:
    st.session_state.cfg = DEFAULT_CFG.copy()

# Convenience locals
total_turns = st.session_state.cfg["total_turns"]
mean_growth = st.session_state.cfg["mean_growth"]
noise_sd = st.session_state.cfg["noise_sd"]
clip_low = st.session_state.cfg["clip_low"]
clip_high = st.session_state.cfg["clip_high"]

# Map animals
ANIMALS = {row["name"]: row for row in st.session_state.animals}
animal_names = list(ANIMALS.keys())

# ---------- Build MARKET_PROBS dynamically from player count and deck composition ----------
def build_market_probs() -> Dict[str, List[float]]:
    """
    Build market probabilities dynamically based on:
    - Number of players (more players = more supply needed)
    - Deck composition (Chickens 46%, Pigs 26%, Cows 20%)
    """
    # Get current number of players (defaults to 3 if none added yet)
    num_players = len(st.session_state.players) if st.session_state.players else 3
    
    # Calculate expected supply for common animals
    expected_per_turn = calculate_expected_per_turn(num_players)
    
    probs: Dict[str, List[float]] = {}
    
    # Common animals: use calculated expectations
    for animal in ["Chicken", "Pig", "Cow"]:
        expected = expected_per_turn.get(animal, 2.0)
        probs[animal] = probs_tapered(expected, slots=SLOTS_COMMON, taper=TAPER)
    
    # Exotics: fixed probabilities (don't scale with players - they're "lucky finds")
    probs["Bison"] = [0.33]      # 33% chance per turn (when not on cooldown)
    probs["Elephant"] = [0.20]   # 20% chance per turn (when not on cooldown)
    
    return probs

def get_market_probs() -> Dict[str, List[float]]:
    """Get current market probabilities, building them if needed."""
    return build_market_probs()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Game Configuration")
    st.session_state.cfg["total_turns"] = st.number_input(
        "Total Turns (info)", min_value=5, max_value=50, step=1,
        value=st.session_state.cfg["total_turns"], key="cfg_total_turns",
    )
    st.session_state.cfg["mean_growth"] = st.number_input(
        "Mean Growth (0.10 = 10%)", step=0.01, format="%.2f",
        value=st.session_state.cfg["mean_growth"], key="cfg_mean_growth",
    )
    st.session_state.cfg["noise_sd"] = st.number_input(
        "Noise SD (shared each turn)", step=0.01, format="%.2f",
        value=st.session_state.cfg["noise_sd"], key="cfg_noise_sd",
    )
    st.session_state.cfg["clip_low"] = st.number_input(
        "Clip Low (e.g., -0.25)", step=0.01, format="%.2f",
        value=st.session_state.cfg["clip_low"], key="cfg_clip_low",
    )
    st.session_state.cfg["clip_high"] = st.number_input(
        "Clip High (e.g., 0.70)", step=0.01, format="%.2f",
        value=st.session_state.cfg["clip_high"], key="cfg_clip_high",
    )
    st.session_state.rng_seed = st.number_input(
        "Random Seed", step=1,
        value=int(st.session_state.cfg["rng_seed"]), key="cfg_rng_seed",
    )
    st.session_state.cfg["rng_seed"] = int(st.session_state.rng_seed)

    st.markdown("---")
    cA, cB = st.columns(2)
    with cA:
        if st.button("Reset Stats to Defaults"):
            st.session_state.cfg = DEFAULT_CFG.copy()
            st.session_state.rng_seed = DEFAULT_CFG["rng_seed"]
            st.rerun()
    with cB:
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = False

        if not st.session_state.confirm_reset:
            if st.button("🔄 Reset Game"):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning("⚠️ Are you sure you want to reset the game?")
            if st.button("✅ Yes, reset now"):
                st.session_state.confirm_reset = False
                reset_state()
                st.rerun()
            if st.button("❌ Cancel"):
                st.session_state.confirm_reset = False
                st.rerun()

# ---------- Check if game is over ----------
if st.session_state.turn > total_turns:
    st.balloons()
    st.success(f"🎉 Game Over! Final scores after {total_turns} turns:")
    
    # Calculate final standings
    final_standings = []
    for p in st.session_state.players:
        final_vp = int(st.session_state.vp[p])
        final_bushels = int(st.session_state.bushels[p])
        total_animals = sum(int(st.session_state.herds[p].get(a, 0)) for a in animal_names)
        
        final_standings.append({
            "Player": p,
            "Value Points": final_vp,
            "Bushels": final_bushels,
            "Total Animals": total_animals,
            "Color": st.session_state.player_color[p]
        })
    
    # Sort by VP (desc), then bushels (desc), then animals (desc)
    final_standings.sort(key=lambda x: (-x["Value Points"], -x["Bushels"], -x["Total Animals"]))
    
    # Display winner
    winner = final_standings[0]
    st.markdown(f"## 🏆 Winner: {winner['Player']} with {winner['Value Points']} VP!")
    
    # Display full standings
    standings_df = pd.DataFrame(final_standings)
    
    def highlight_winner(row):
        if row["Player"] == winner["Player"]:
            return ['background-color: gold; font-weight: bold'] * len(row)
        else:
            hex_color = row["Color"].lstrip("#")
            try:
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            except Exception:
                r, g, b = (255, 255, 255)
            rgba = f"background-color: rgba({r},{g},{b},0.3)"
            return [rgba] * len(row)
    
    # Remove Color column from display
    display_df = standings_df.drop(columns=["Color"])
    st.dataframe(display_df.style.apply(highlight_winner, axis=1), width="stretch")
    
    st.markdown("---")
    st.info("Game complete! Use '🔄 Reset Game' in the sidebar to play again.")
    st.stop()  # Stop rendering the rest of the UI

# ---------- Animal Costs (single table) ----------
st.subheader("Animal Costs")
costs_view = pd.DataFrame([
    {
        "Animal": n,
        "Feed/Turn": int(ANIMALS[n]["feed_per_turn"]),
        "Cost (bushels)": int(ANIMALS[n]["cost"]),
        "Sale Value (60%)": int(math.floor(0.6 * int(ANIMALS[n]["cost"]))),
    }
    for n in animal_names
])
st.dataframe(costs_view, width="stretch")

# ---------- Players ----------
st.subheader("Players")
roster_locked = st.session_state.turn > 1

if not roster_locked:
    cols = st.columns([2, 1, 1, 1.2, 1])
    with cols[0]:
        new_player = st.text_input("Add player name", value="")
    with cols[1]:
        start_bushels = st.number_input("Start Bushels", value=100, step=1, min_value=0)
    with cols[2]:
        default_color = "#99c2ff"
        new_color = st.color_picker("Player Color", value=default_color)
    with cols[3]:
        if st.button("Add Player"):
            name = new_player.strip()
            if name and name not in st.session_state.players:
                st.session_state.players.append(name)
                st.session_state.player_color[name] = new_color
                st.session_state.bushels[name] = int(start_bushels)
                st.session_state.herds[name] = {a: 0 for a in animal_names}
                st.session_state.turn_start_bushels[name] = int(start_bushels)
                st.session_state.turn_start_herd[name] = {a: 0 for a in animal_names}
                # VP init
                st.session_state.vp[name] = 0
                st.session_state.turn_start_vp[name] = 0
                st.session_state.vp_earned[name] = 0
                
                # Force market redraw on next ensure_market_supply() call
                st.session_state.market_draw_turn = 0
                
                # Save state after adding player
                save_game_state()
                
                st.success(f"Added {name}")
                st.rerun()  # Force rerun to update market display
    with cols[4]:
        if st.button("Remove Last"):
            if st.session_state.players:
                p = st.session_state.players.pop()
                for d in ("player_color","bushels","herds","turn_start_bushels","turn_start_herd",
                          "vp","turn_start_vp","vp_earned"):
                    st.session_state.get(d, {}).pop(p, None)
                
                # Force market redraw on next ensure_market_supply() call
                st.session_state.market_draw_turn = 0
                
                st.rerun()  # Force rerun to update market display
else:
    st.info("Roster locked after Turn 1 begins (Add/Remove disabled).")

if not st.session_state.players:
    st.info("Add at least one player to begin.")
    st.stop()

# Ensure baselines exist
for p in st.session_state.players:
    st.session_state.turn_start_bushels.setdefault(p, int(st.session_state.bushels[p]))
    st.session_state.turn_start_herd.setdefault(p, st.session_state.herds[p].copy())
    st.session_state.player_color.setdefault(p, "#d0e1ff")
    st.session_state.vp.setdefault(p, 0)
    st.session_state.turn_start_vp.setdefault(p, int(st.session_state.vp[p]))
    st.session_state.vp_earned.setdefault(p, 0)

# ---------- Market drawing ----------
def draw_market_for_current_turn() -> Dict[str, int]:
    """Draw counts from market probabilities once per turn, seeded for reproducibility.
       Enforces turn cooldown for Bison and Elephant.
    """
    rng = np.random.default_rng(int(st.session_state.rng_seed) + int(st.session_state.turn) + 137)
    supply = {}
    turn = int(st.session_state.turn)
    
    # Get current market probabilities
    market_probs = get_market_probs()

    for a in animal_names:
        probs = market_probs.get(a, [])
        count = 0
        for p in probs:
            if rng.random() < p:
                count += 1
        supply[a] = count

    # --- Apply cooldowns (only after successful spawn) ---
    for species, cooldown in [("Bison", BISON_COOLDOWN), ("Elephant", ELEPHANT_COOLDOWN)]:
        last_turn = st.session_state.last_spawn_turn.get(species, -10)
        # if too soon since last appearance, block spawn
        if (turn - last_turn) <= cooldown:
            supply[species] = 0
        # if one spawned this turn, record that
        elif supply.get(species, 0) > 0:
            st.session_state.last_spawn_turn[species] = turn

    return supply

def ensure_market_supply():
    """Ensure market supply is drawn for current turn."""
    if st.session_state.market_draw_turn != st.session_state.turn:
        drawn = draw_market_for_current_turn()
        for a in animal_names:
            drawn.setdefault(a, 0)
        st.session_state.market_supply = drawn
        st.session_state.market_draw_turn = st.session_state.turn

ensure_market_supply()

# ---------- Helpers (SIGNED FLOWS) ----------
def feed_cost_signed(herd: Dict[str, int]) -> int:
    raw = int(sum(int(ANIMALS[a]["feed_per_turn"]) * int(herd.get(a, 0)) for a in animal_names))
    return -raw  # negative cost

def purchase_sales_flow_signed(p: str) -> int:
    start_herd = st.session_state.turn_start_herd[p]
    curr_herd = st.session_state.herds[p]
    total = 0
    for a in animal_names:
        delta = int(curr_herd.get(a, 0)) - int(start_herd.get(a, 0))
        unit_cost = int(ANIMALS[a]["cost"])
        if delta > 0:
            total += -delta * unit_cost           # purchase = negative
        elif delta < 0:
            units_sold = -delta
            refund_per = math.floor(0.6 * unit_cost)  # 60% refund, rounded DOWN
            total += units_sold * refund_per          # sale = positive
    return int(total)

def end_of_turn_bushels_preview(p: str) -> int:
    start_b = int(st.session_state.turn_start_bushels[p])
    flow = purchase_sales_flow_signed(p)
    feed = feed_cost_signed(st.session_state.herds[p])
    return max(0, start_b + flow + feed)

def max_affordable_units(p: str, animal: str, want: int) -> int:
    start_b = int(st.session_state.turn_start_bushels[p])
    current_flow = purchase_sales_flow_signed(p)
    current_feed = feed_cost_signed(st.session_state.herds[p])
    current_eot = max(0, start_b + current_flow + current_feed)

    unit_cost = int(ANIMALS[animal]["cost"])
    unit_feed = int(ANIMALS[animal]["feed_per_turn"])
    per_unit_drop = unit_cost + unit_feed

    if per_unit_drop <= 0:
        return want
    max_units = (current_eot - 1) // per_unit_drop
    return int(max(0, min(want, max_units)))

def update_herd_and_market(p: str, animal: str, requested_count: int):
    prev = int(st.session_state.herds[p].get(animal, 0))
    supply = int(st.session_state.market_supply.get(animal, 0))

    if requested_count > prev:
        want = requested_count - prev
        from_supply = min(want, supply)
        can_afford = max_affordable_units(p, animal, from_supply)
        take = min(from_supply, can_afford)

        new_count = prev + take
        st.session_state.herds[p][animal] = new_count
        st.session_state.market_supply[animal] = supply - take
        return take, 0, new_count

    elif requested_count < prev:
        sell = prev - requested_count
        new_count = prev - sell
        st.session_state.herds[p][animal] = new_count
        st.session_state.market_supply[animal] = supply + sell
        return 0, sell, new_count

    else:
        return 0, 0, prev

# ---------- Turn order (rotating) ----------
def get_turn_order(players: List[str], turn_num: int) -> List[str]:
    if not players:
        return []
    dq = deque(players)
    rot = (turn_num - 1) % len(players)
    dq.rotate(-rot)
    return list(dq)

ordered_players = get_turn_order(st.session_state.players, st.session_state.turn)

# ---------- Per-player panels ----------
st.subheader(f"Turn {st.session_state.turn}: Manage Players")
for p in ordered_players:
    with st.expander(f"{p} — click to manage", expanded=False):
        herd = st.session_state.herds[p]
        start_b = int(st.session_state.turn_start_bushels[p])
        start_herd = st.session_state.turn_start_herd[p]
        start_vp = int(st.session_state.turn_start_vp[p])

        st.markdown(f"### {p}")
        st.session_state.player_color[p] = st.color_picker(
            "Color", value=st.session_state.player_color[p], key=f"{p}_color"
        )

        # Starting Inventory (bigger font)
        st.markdown("Starting Inventory")
        header_cols = st.columns(len(animal_names))
        for idx, a in enumerate(animal_names):
            with header_cols[idx]:
                count = int(start_herd.get(a, 0))
                st.markdown(
                    f"<div style='font-size:22px; font-weight:600;'>{a}: {count}</div>",
                    unsafe_allow_html=True
                )

        # Editable animal counts with market enforcement
        grid = st.columns(len(animal_names))
        for idx, a in enumerate(animal_names):
            with grid[idx]:
                current_saved = int(st.session_state.herds[p].get(a, 0))
                requested = st.number_input(
                    a, min_value=0, value=current_saved, step=1, key=f"{p}_{a}_count"
                )
                bought, sold, applied = update_herd_and_market(p, a, int(requested))
                
                # Show appropriate warning based on what blocked the purchase
                if requested > current_saved:
                    want = requested - current_saved
                    supply = int(st.session_state.market_supply.get(a, 0))
                    can_afford = max_affordable_units(p, a, supply)
                    
                    if bought < want:
                        if supply == 0:
                            st.warning(f"No {a}s available in market right now.")
                        elif supply < want:
                            st.warning(f"Only {supply} {a}(s) available in market (you wanted {want}).")
                        elif can_afford < want:
                            needed_bushels = (want - can_afford) * (int(ANIMALS[a]["cost"]) + int(ANIMALS[a]["feed_per_turn"]))
                            st.warning(f"Can't afford {want} {a}(s). You can only buy {can_afford} while staying above 1 bushel. Need ~{needed_bushels} more bushels.")
                        else:
                            st.warning(f"Only {bought} {a}(s) purchased (wanted {want}).")
                
                start_ct = int(st.session_state.turn_start_herd[p].get(a, 0))
                delta_vs_start = applied - start_ct
                st.markdown(
                    f"<div style='font-size:18px; color:#444;'>Change this turn: {delta_vs_start:+d}</div>",
                    unsafe_allow_html=True
                )

        # ----- Bushels metrics (TOP row) -----
        flow = purchase_sales_flow_signed(p)                 # buy -, sell +
        feed = feed_cost_signed(st.session_state.herds[p])   # negative
        eot  = end_of_turn_bushels_preview(p)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start Bushels (this turn)", f"{start_b}")
        c2.metric("New Animal Flow (net)", f"{flow}")
        c3.metric("Feed / Turn", f"{feed}")
        c4.metric("End-of-Turn Bushels (preview)", f"{eot}")

        # ----- Value Points box (BOTTOM row) -----
        st.markdown("---")
        col_vp1, col_vp2, col_vp3 = st.columns(3)
        with col_vp1:
            st.metric("Starting VP (this turn)", f"{start_vp}")
        with col_vp2:
            # IMPORTANT: include turn number in the widget key so it resets each turn
            widget_key = f"{p}_vp_earned_T{st.session_state.turn}"
            earned_now = st.number_input(
                "VP earned this turn",
                min_value=0, step=1,
                value=int(st.session_state.vp_earned[p]),
                key=widget_key
            )
            st.session_state.vp_earned[p] = int(earned_now)
        with col_vp3:
            eot_vp_preview = start_vp + int(st.session_state.vp_earned[p])
            st.metric("End-of-Turn VP (preview)", f"{eot_vp_preview}")

# ---------- Re-render market after potential changes ----------
st.subheader(f"Market Supply (Live) — Turn {st.session_state.turn}")

# Define display order for animals
animal_display_order = ["Chicken", "Pig", "Cow", "Bison", "Elephant"]

# Create ordered dataframe
market_df_live = pd.DataFrame([
    {"Animal": a, "Available": int(st.session_state.market_supply.get(a, 0))}
    for a in animal_display_order
])
st.dataframe(market_df_live, width="stretch")

# ---------- Run Night (shared growth -> ceil next start) ----------
st.markdown("---")
left, right = st.columns([1, 3])

# Guard: EOT bushels must be >= 1 for ALL players before allowing Night
blocking_players = [p for p in st.session_state.players if end_of_turn_bushels_preview(p) < 1]
if blocking_players:
    st.error("Cannot run Night: the following players must sell to keep at least 1 bushel: " +
             ", ".join(blocking_players))

with left:
    if st.button("🌙 Run Night: Apply Shared Growth & Advance Turn", type="primary",
                 disabled=len(blocking_players) > 0):
        rng = np.random.default_rng(int(st.session_state.rng_seed) + int(st.session_state.turn))
        eps = float(rng.normal(0.0, float(noise_sd)))
        eps = max(float(clip_low), min(float(clip_high), eps))
        growth_rate = float(mean_growth) + eps
        growth_mult = max(0.0, 1.0 + growth_rate)
        st.session_state.last_growth_rate = growth_rate

        snap = {"turn": st.session_state.turn, "growth_rate": growth_rate, "players": {}}

        for p in st.session_state.players:
            eot = end_of_turn_bushels_preview(p)
            next_start = math.ceil(eot * growth_mult)
            
            # Safety: ensure players always wake up with at least 1 bushel
            next_start = max(1, next_start)

            # commit VP
            start_vp = int(st.session_state.turn_start_vp[p])
            earned_vp = int(st.session_state.vp_earned[p])
            end_vp = start_vp + earned_vp
            st.session_state.vp[p] = end_vp

            snap["players"][p] = {
                "start_bushels": int(st.session_state.turn_start_bushels[p]),
                "new_animal_flow": int(purchase_sales_flow_signed(p)),
                "feed_flow": int(feed_cost_signed(st.session_state.herds[p])),
                "end_bushels_pre_growth": int(eot),
                "growth_applied_%": growth_rate * 100.0,
                "next_turn_start_bushels": int(next_start),
                "herd": st.session_state.herds[p].copy(),
                "color": st.session_state.player_color[p],
                # VP snapshot
                "start_vp": start_vp,
                "earned_vp": earned_vp,
                "end_vp": end_vp,
            }

            # commit for next turn (bushels/herds/vp)
            st.session_state.bushels[p] = int(next_start)
            st.session_state.turn_start_bushels[p] = int(next_start)
            st.session_state.turn_start_herd[p] = st.session_state.herds[p].copy()

            st.session_state.turn_start_vp[p] = int(st.session_state.vp[p])  # carry forward
            st.session_state.vp_earned[p] = 0                                 # reset earned

        st.session_state.history.append(snap)
        st.session_state.turn += 1

        # New turn => fresh market (will rebuild with current player count)
        ensure_market_supply()
        
        # Save state after running night
        save_game_state()
        
        st.rerun()

with right:
    if st.session_state.last_growth_rate is not None:
        st.success(f"Shared growth rate applied last night: **{st.session_state.last_growth_rate*100:.2f}%**")

# ---------- Scoreboard (current turn preview) ----------
def scoreboard_df() -> pd.DataFrame:
    rows = []
    for p in st.session_state.players:
        herd = st.session_state.herds[p]
        row = {
            "Player": p,
            "Start Bushels (this turn)": int(st.session_state.turn_start_bushels[p]),
            "New Animal Flow (net)": int(purchase_sales_flow_signed(p)),
            "Feed / Turn": int(feed_cost_signed(herd)),
            "End-of-Turn Bushels (preview)": int(end_of_turn_bushels_preview(p)),
            "Starting VP": int(st.session_state.turn_start_vp[p]),
            "VP Earned (this turn)": int(st.session_state.vp_earned[p]),
            "End-of-Turn VP (preview)": int(st.session_state.turn_start_vp[p] + st.session_state.vp_earned[p]),
        }
        for a in animal_names:
            row[a] = int(herd.get(a, 0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Player")

st.subheader("Scoreboard (current turn preview)")
df = scoreboard_df()

def highlight_players(row, alpha=0.30):
    hex_color = st.session_state.player_color.get(row["Player"], "#ffffff").lstrip("#")
    try:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except Exception:
        r, g, b = (255, 255, 255)
    rgba = f"background-color: rgba({r},{g},{b},{alpha})"
    return [rgba] * len(row)

st.dataframe(df.style.apply(highlight_players, axis=1), width="stretch")

# ---------- History ----------
st.subheader("History (after Night)")
if st.session_state.history:
    hist_rows = []
    for snap in st.session_state.history:
        t = snap["turn"]
        g = snap["growth_rate"] * 100.0
        for p, info in snap["players"].items():
            row = {
                "Turn": t,
                "Growth %": f"{g:.2f}%",
                "Player": p,
                "Start Bushels": info["start_bushels"],
                "New Animal Flow (net)": info["new_animal_flow"],
                "Feed Flow": info["feed_flow"],
                "End Bushels (pre-growth)": info["end_bushels_pre_growth"],
                "Next Turn Start Bushels": info["next_turn_start_bushels"],
                # VP
                "Start VP": info.get("start_vp", 0),
                "VP Earned": info.get("earned_vp", 0),
                "End VP": info.get("end_vp", 0),
            }
            for a in animal_names:
                row[a] = int(info["herd"].get(a, 0))
            hist_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)

    def highlight_hist(row, alpha=0.30):
        hex_color = st.session_state.player_color.get(row["Player"], "#ffffff").lstrip("#")
        try:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        except ValueError:
            r, g, b = 255, 255, 255
        rgba = f"background-color: rgba({r},{g},{b},{alpha})"
        return [rgba] * len(row)

    st.dataframe(hist_df.style.apply(highlight_hist, axis=1), width="stretch")
else:
    st.info("No history yet — click 'Run Night' to log a turn.")
