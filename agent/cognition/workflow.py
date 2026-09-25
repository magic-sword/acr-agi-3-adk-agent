"""ADK 2.0 graph with persistent session state and bounded local-model calls."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
from pathlib import Path
import threading
import time

from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import node
from google.genai import types

from agent.controls import controller_context
from agent.observation import VISUAL_PAYLOAD_KEYS, render_current, png_base64, validate_cursor
from agent.local_vlm import LocalVisionLlm, MAX_REQUESTS_PER_INVOCATION
from .engine import CognitiveTurn, new_memory
from .evidence import EvidenceStore
from .instructions import instruction
from .skills import skill_toolset, selected_skills
from .state import Decision, Memory
from .completion import CompletionTool, COMPLETION_TOOLS, submission_schema
from .planning import PlanOrderTool
from .causal import CausalStore, CausalMemoryTool, BackwardPlanTool

APP_NAME = "arc_cognition"


class CognitiveRuntime:
    """One instance per game run; one event loop, writer and ADK session.

    The caller must use decision_id to avoid re-executing a retried decision.
    Journals are diagnostic snapshots, not an exactly-once restart protocol.
    """

    def __init__(self, game_id: str, model: str | None = None, *, max_calls: int = 3,
                 max_resets: int = 2, seconds: float = 600, log_dir: str | None = None,
                 knowledge_dir: str | None = None):
        if model not in (None, "local/qwen3-vl-4b-instruct"):
            raise ValueError(f"Unsupported ADK_MODEL: {model}")
        if seconds <= 0 or not 1 <= max_calls <= 8 or max_resets < 0:
            raise ValueError("invalid cognition budget")
        self.model = model
        self.memory = new_memory(game_id)
        self.max_calls, self.max_resets = max_calls, max_resets
        self.deadline = time.monotonic() + seconds
        self.log_dir = Path(log_dir) if log_dir else None
        self.loop = asyncio.new_event_loop()
        self.lock = threading.Lock()
        self.service = InMemorySessionService()
        self.session_id = self.memory.run_id
        self.turn: CognitiveTurn | None = None
        self.initialized = False
        self.closed = False
        self.agents = {}
        self._submission = None
        self._submission_attempts = []
        self.cursor = None
        self.evidence = EvidenceStore()
        self.causal = CausalStore(game_id, knowledge_dir or os.getenv('CAUSAL_MEMORY_DIR', 'outputs/causal-memory'))
        self.workflow = self._build_graph()
        self.runner = Runner(agent=self.workflow, app_name=APP_NAME, session_service=self.service)

    def _agent(self, state: str):
        selected_skills(state)
        key = state
        if key not in self.agents:
            self.agents[key] = LlmAgent(
                name=f"reasoner_{key.lower()}", include_contents="none",
                model=LocalVisionLlm(model="qwen3-vl-4b-instruct",
                                     api_base=os.getenv("VLM_API_BASE", "http://vlm:8080/v1"),
                                     max_output_tokens=1000,
                                     max_requests=3,
                                     timeout_seconds=90, completion_tools=(COMPLETION_TOOLS[state],)),
                instruction=instruction(state, submission_schema(state)),
                tools=[CompletionTool(self, state), skill_toolset(state), self.observe_current, self.list_observations,
                       self.get_observation, self.compare_observations, self.move_cursor, self.observe_animation,
                       PlanOrderTool(), CausalMemoryTool(self), BackwardPlanTool(self)],
            )
        return self.agents[key]

    def observe_current(self) -> dict:
        """Read the final received game frame with cursor/controller. Never replay history or advance time."""
        o = self.turn.obs
        return {"observation_id": o["observation_id"], "view": "current_final_frame",
                "cursor": o.get("cursor"), "animation": controller_context(o.get("animation")),
                "available_buttons": controller_context({"available_actions": o.get("available_actions", [])})["available_actions"],
                "_image_png_base64": o.get("image_png_base64")}

    def move_cursor(self, x: int, y: int) -> dict:
        """Preview a host cursor at original game pixel x,y; does NOT click or advance the game."""
        o = self.turn.obs
        if not o.get("_visual_frames"):
            return {"error": "No source frame available for cursor preview"}
        try:
            cursor = validate_cursor({"x": x, "y": y}, o["width"], o["height"])
        except ValueError as exc:
            return {"error": str(exc)}
        self.cursor = o["cursor"] = cursor
        o["image_png_base64"] = png_base64(render_current(o["_visual_frames"][-1], o["available_actions"], cursor))
        return self.observe_current()

    def list_observations(self) -> dict:
        """List retained real observation IDs, steps, boundaries and source actions."""
        return {"observations": self.evidence.list(), "capacity": self.evidence.capacity,
                "current_id": self.turn.obs.get("observation_id"), "time_advanced": False}

    def get_observation(self, observation_id: str) -> dict:
        """Retrieve a retained historical screen by ID. This is not a new game observation."""
        try:
            return controller_context(self.evidence.view(observation_id))
        except ValueError as exc:
            return {"error": str(exc)}

    def compare_observations(self, before_id: str, after_id: str, offset: int = 0) -> dict:
        """Inspect labeled before/after screens and paginated pixel differences; not inferred effects."""
        try:
            return controller_context(self.evidence.compare(before_id, after_id, offset))
        except ValueError as exc:
            return {"error": str(exc)}

    def observe_animation(self, event_id: str, start_frame: int = 0) -> dict:
        """Retrieve a retained action's intermediate frames. Replay never advances game time."""
        try:
            return controller_context(self.evidence.animation(event_id, start_frame))
        except ValueError as exc:
            return {"error": str(exc)}

    def _context(self):
        t, m = self.turn, self.turn.memory
        o = {k: v for k, v in t.obs.items() if k not in VISUAL_PAYLOAD_KEYS | {"grid", "recent_actions"}}
        previous = self.memory.pending
        # Match each past action to the following actual frame, not the change before that action.
        trials = []
        for index in range(max(0, len(m.history) - 4), len(m.history)):
            entry = m.history[index]
            after = m.history[index + 1] if index + 1 < len(m.history) else t.obs
            same_level = entry.get("levels_completed") == after.get("levels_completed")
            comparable = (same_level and entry.get("frame_hash") and after.get("frame_hash")
                          and entry.get("action", {}).get("action") != "RESET"
                          and not (index == len(m.history) - 1 and t.boundary))
            decision = entry.get("decision", {})
            trials.append({"step": entry["step"], "action": entry.get("action"),
                           "prediction": decision.get("prediction"),
                           "observed_frame_changed": (entry["frame_hash"] != after["frame_hash"])
                           if comparable else None})
        context = {"observation": o, "observation_id": t.obs["observation_id"],
                   "memory_revision": m.revision, "notebook": m.notebook,
                   "boundary": t.boundary,
                   "previous_action": previous.model_dump(exclude_none=True) if previous else None,
                   "recent_trials": trials,
                   "errors": t.errors[-1:], "remaining_seconds": round(t.time_left(), 2),
                   "observation_records": self.evidence.list()[-8:]}
        try:
            context['causal_memory'] = self.causal.overview()
        except (OSError, ValueError, sqlite3.Error) as exc:
            context['causal_memory'] = {'error': str(exc)[:200]}
        return controller_context(context)

    async def _ask(self, ctx: Context, state: str):
        t = self.turn
        if t.calls >= t.max_calls or t.time_left() <= 0:
            raise ValueError("reasoning budget exhausted")
        t.calls += 1
        context = self._context()
        context["reasoning_state"] = state
        parts = [types.Part(text=json.dumps(context, separators=(",", ":")))]
        # Supply the immediately preceding real screen without a separate VERIFY call.
        previous = self.memory.pending
        if previous and not t.boundary and previous.observation_id in self.evidence.index:
            before = self.evidence.get(previous.observation_id)
            parts.append(types.Part(text="BEFORE the previous action: " + previous.observation_id))
            if before.get("image_png_base64"):
                parts.append(types.Part.from_bytes(data=base64.b64decode(before["image_png_base64"]), mime_type="image/png"))
            elif before.get("grid") is not None:
                parts.append(types.Part(text="Before color-ID grid: " + json.dumps(before["grid"])))
        parts.append(types.Part(text="CURRENT observation (choose the next action here):"))
        if t.obs.get("image_png_base64"):
            parts.append(types.Part.from_bytes(data=base64.b64decode(t.obs["image_png_base64"]), mime_type="image/png"))
        elif t.obs.get("grid") is not None:
            parts.append(types.Part(text="Current color-ID grid: " + json.dumps(t.obs["grid"], separators=(",", ":"))))
        agent = self._agent(state)
        self._submission = None
        self._submission_attempts = []
        agent.model.begin_invocation()
        agent.model.timeout_seconds = max(1, min(90, int(t.time_left())))
        started = time.monotonic()
        record = {"observation_id": t.obs["observation_id"], "step": t.obs["step"],
                  "state": state, "call_index": t.calls, "context": context,
                  "available_skills": selected_skills(state)}
        try:
            async with asyncio.timeout(min(92, t.time_left())):
                await ctx.run_node(agent, node_input=types.Content(role="user", parts=parts))
            record.update(agent.model._last_metrics)
            if self._submission is None:
                raise ValueError("state ended without an accepted completion tool call")
            parsed = self._submission
            record["response"] = parsed.model_dump_json()
            record["completion_tool"] = COMPLETION_TOOLS[state]
            record["schema_valid"] = True
            return parsed
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.update(agent.model._last_metrics)
            record['exchanges'] = agent.model._exchanges
            record['http_requests'] = len(agent.model._exchanges)
            record['submissions'] = self._submission_attempts
            record["seconds"] = time.monotonic() - started
            self._submission = None
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / f"{self.session_id}.model.jsonl").open("a") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _offline_decision(self):
        t, m = self.turn, self.turn.memory
        allowed = [a for a in t.obs.get("available_actions", []) if a.startswith("ACTION")]
        if not allowed:
            raise ValueError("no legal actions")
        counts = {a: sum(h.get("action", {}).get("action") == a for h in m.history) for a in allowed}
        action = min(allowed, key=lambda a: (counts[a], a))
        data = {"action": action, "reason": "deterministic control-effect probe; no semantic model"}
        if action == "ACTION6":
            data.update(x=(t.obs["step"] * 11) % t.obs.get("width", 64),
                        y=(t.obs["step"] * 7) % t.obs.get("height", 64))
        return Decision(action=data, prediction="Compare the next real frame for a visible control effect.")

    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def observe(ctx: Context):
            t = self.turn
            try:
                t.observe()
                if not t.duplicate and t.memory.lifecycle == "ACTIVE":
                    self.evidence.add(t.obs, boundary=t.boundary,
                                      source_action=self.memory.pending.model_dump() if self.memory.pending else None)
                    self._log_observation(t.obs)
                    try:
                        self.causal.archive(t.obs, run_id=self.memory.run_id, boundary=t.boundary,
                                            source_action=self.memory.pending.model_dump() if self.memory.pending else None)
                    except Exception as exc:
                        # Persistence failure must not relabel a valid frame as an invalid observation.
                        t.errors.append(f"causal archive {type(exc).__name__}: {exc}"[:500])
            except Exception as exc:
                t.errors.append(f"observation {type(exc).__name__}: {exc}"[:500])
                t.stop("invalid_observation")
            if not t.duplicate and t.obs.get("observation_id"):
                # Deterministic bookkeeping only. Reflection and next action share one model call.
                t.verify()
                t.update()
            return Event(route="COMMIT" if t.duplicate or t.memory.lifecycle in ("DONE", "STOPPED") else "DECIDE")

        @node(rerun_on_resume=True)
        async def decide(ctx: Context):
            t = self.turn
            t.enter("DECIDE")
            if t.obs["state"] in ("NOT_PLAYED", "GAME_OVER"):
                t.recover()
            elif not t.obs.get("available_actions") or not any(
                    a.startswith("ACTION") for a in t.obs["available_actions"]):
                t.stop("no_legal_action")
            elif not self.model:
                t.accept_decision(self._offline_decision())
            else:
                # Retries repair malformed submissions, never route through more reasoning states.
                while t.calls < t.max_calls and t.time_left() > 0:
                    try:
                        t.accept_decision(await self._ask(ctx, "DECIDE"))
                        break
                    except Exception as exc:
                        t.errors.append(f"decision {type(exc).__name__}: {exc}"[:500])
                if t.selected is None:
                    t.stop("budget_exhausted" if t.time_left() <= 0 else "proposal_unavailable")
            return Event(output="decided")

        def commit():
            result = self.turn.commit()
            return Event(output=result, state={"cognition": self.turn.memory.model_dump(mode="json")})

        return Workflow(name="cognitive_workflow", edges=[
            ("START", observe), (observe, {"DECIDE": decide, "COMMIT": commit}),
            (decide, commit),
        ])

    def _log_observation(self, obs: dict):
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            record = {k: v for k, v in obs.items() if k not in VISUAL_PAYLOAD_KEYS}
            if obs.get("image_png_base64"):
                frames_dir = self.log_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                frame_name = f"{self.session_id}-{obs['step']:05d}.png"
                (frames_dir / frame_name).write_bytes(base64.b64decode(obs["image_png_base64"]))
                record["image_path"] = "frames/" + frame_name
            if obs.get("_visual_frames"):
                archive_name = f"{self.session_id}-{obs['step']:05d}.json"
                archive = {"frames": obs["_visual_frames"], "available_actions": obs["available_actions"],
                           "cursor": obs["cursor"], **obs["animation"]}
                (frames_dir / archive_name).write_text(json.dumps(archive))
                record["animation_archive_path"] = "frames/" + archive_name
            with (self.log_dir / f"{self.session_id}.observations.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")

    async def _decide(self, obs: dict) -> dict:
        if not self.initialized:
            await self.service.create_session(app_name=APP_NAME, user_id="player", session_id=self.session_id,
                                              state={"cognition": self.memory.model_dump(mode="json")})
            self.initialized = True
        self.turn = CognitiveTurn(self.memory, obs, max_calls=self.max_calls,
                                  max_resets=self.max_resets, deadline=self.deadline)
        message = types.Content(role="user", parts=[types.Part(text=f'Observe external step {obs.get("step")}')])
        async for _ in self.runner.run_async(user_id="player", session_id=self.session_id,
                                             new_message=message,
                                             run_config=RunConfig(max_llm_calls=self.max_calls * MAX_REQUESTS_PER_INVOCATION)):
            pass
        session = await self.service.get_session(app_name=APP_NAME, user_id="player", session_id=self.session_id)
        self.memory = Memory.model_validate(session.state["cognition"])
        if self.turn.result is None:
            raise RuntimeError("workflow did not commit a decision")
        if self.log_dir and not self.turn.duplicate:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            snapshot = self.log_dir / f"{self.session_id}.json"
            tmp = snapshot.with_suffix(".tmp")
            tmp.write_text(self.memory.model_dump_json(indent=2))
            tmp.replace(snapshot)
            with (self.log_dir / f"{self.session_id}.jsonl").open("a") as f:
                f.write(json.dumps(self.memory.history[-1]) + "\n")
        return self.turn.result

    def decide(self, observation: dict) -> dict:
        with self.lock:
            if self.closed:
                raise RuntimeError("cognitive runtime is closed")
            return self.loop.run_until_complete(self._decide(observation))

    def close(self):
        with self.lock:
            if not self.closed:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
                self.loop.close()
                self.evidence.close()
                self.closed = True
