"""Run the same MyAgent class against local ARC-AGI-3 game environments."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if not VENDOR.is_dir():
    raise SystemExit("Missing vendor/ARC-AGI-3-Agents. Run make setup first.")
sys.path[:0] = [str(ROOT), str(VENDOR)]

from arc_agi import Arcade, OperationMode
from agent.my_agent import MyAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="Short game ID, comma-separated IDs, or omit for all")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arcade = Arcade(operation_mode=OperationMode.NORMAL)
    games = arcade.get_environments()
    if args.list:
        for game in games:
            print(game.game_id)
        return
    known = {game.game_id.split("-")[0] for game in games}
    requested = {g.strip().split("-")[0] for g in args.game.split(",")} if args.game else known
    if unknown := requested - known:
        parser.error(f"Unknown game IDs: {sorted(unknown)}")
    # The framework stops after counter <= MAX_ACTIONS (inclusive).
    MyAgent.MAX_ACTIONS = args.max_steps - 1
    results = []
    for game_id in sorted(requested):
        env = arcade.make(game_id)
        if env is None:
            raise RuntimeError(f"Could not initialize {game_id}")
        player = MyAgent(
            card_id="local-dev", game_id=game_id, agent_name="MyAgent.local",
            ROOT_URL="http://localhost", record=False, arc_env=env, tags=["local"],
        )
        player.main()
        final = player.frames[-1]
        results.append((game_id, final.levels_completed, player.action_counter, final.state))
    for game_id, levels, actions, state in results:
        print(f"{game_id}: levels={levels}, actions={actions}, state={state}")
    scorecard = arcade.get_scorecard()
    print(f"Aggregate local score: {getattr(scorecard, 'score', scorecard)}")


if __name__ == "__main__":
    main()
