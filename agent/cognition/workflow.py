"""ADK 2.0 graph with persistent session state and bounded local-model calls."""
from __future__ import annotations

import asyncio
import base64
import json
import os
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

from agent.local_vlm import LocalVisionLlm, MAX_REQUESTS_PER_INVOCATION
from .engine import CognitiveTurn, new_memory
from .skills import instruction, skill_toolset
from .state import Memory, Perception, Proposal
from .validation import parse_json

APP_NAME = "arc_cognition"


class CognitiveRuntime:
    """One instance per game run; one event loop, writer and ADK session.

    The caller must use decision_id to avoid re-executing a retried decision.
    Journals are diagnostic snapshots, not an exactly-once restart protocol.
    """

    def __init__(self, game_id: str, model: str | None = None, *, max_calls: int = 3,
                 max_resets: int = 2, seconds: float = 600, log_dir: str | None = None):
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
        self.workflow = self._build_graph()
        self.runner = Runner(agent=self.workflow, app_name=APP_NAME, session_service=self.service)

    def _agent(self, state: str):
        if state not in self.agents:
            schema = Perception if state == "OBSERVE" else Proposal
            self.agents[state] = LlmAgent(
                name=f"skill_{state.lower()}", include_contents="none",
                model=LocalVisionLlm(model="qwen3-vl-4b-instruct",
                                     api_base=os.getenv("VLM_API_BASE", "http://vlm:8080/v1"),
                                     max_output_tokens=1600 if state == "OBSERVE" else 2400,
                                     timeout_seconds=90),
                instruction=instruction(state, schema.model_json_schema()),
                tools=[skill_toolset(state)],
            )
        return self.agents[state]

    def _context(self):
        t, m = self.turn, self.turn.memory
        o = {k: v for k, v in t.obs.items() if k not in ("image_png_base64", "grid", "recent_actions")}
        context = {"observation": o, "observation_id": t.obs["observation_id"], "memory_revision": m.revision,
                "attempt": m.attempt, "model_revision": m.model_revision,
                "goal": m.goal, "facts": [f.model_dump() for f in m.facts.values()][-24:],
                "hypotheses": [h.model_dump() for h in m.hypotheses.values()][-16:],
                "unknowns": m.unknowns, "plan": [n.model_dump() for n in m.plan],
                "pending": m.pending.model_dump() if m.pending else None,
                "deferred": [d.model_dump() for d in m.deferred],
                "verification": t.verification, "procedures": m.procedures[-2:],
                "history": m.history[-3:], "errors": t.errors[-1:],
                "remaining_seconds": round(t.time_left(), 2)}
        # Drop optional histories first. Never truncate JSON or silently omit plan constraints.
        for key in ("history", "procedures"):
            if len(json.dumps(context)) > 24000:
                context[key] = []
        if len(json.dumps(context)) > 24000:
            raise ValueError("structured context exceeds budget")
        return context

    async def _ask(self, ctx: Context, state: str):
        t = self.turn
        if t.calls >= t.max_calls or t.time_left() <= 0:
            raise ValueError("reasoning budget exhausted")
        t.calls += 1
        context = self._context()
        parts = [types.Part(text=json.dumps(context, separators=(",", ":")))]
        if t.obs.get("image_png_base64"):
            parts.append(types.Part.from_bytes(data=base64.b64decode(t.obs["image_png_base64"]), mime_type="image/png"))
        elif t.obs.get("grid") is not None:
            parts.append(types.Part(text="Current color-ID grid: " + json.dumps(t.obs["grid"], separators=(",", ":"))))
        agent = self._agent(state)
        agent.model.begin_invocation()
        agent.model.timeout_seconds = max(1, min(90, int(t.time_left())))
        started = time.monotonic()
        record = {"observation_id": t.obs["observation_id"], "step": t.obs["step"],
                  "state": state, "call_index": t.calls, "context": context}
        try:
            async with asyncio.timeout(min(92, t.time_left())):
                answer = await ctx.run_node(agent, node_input=types.Content(role="user", parts=parts))
            record.update(agent.model._last_metrics)
            if isinstance(answer, types.Content):
                answer = "".join(p.text or "" for p in answer.parts or [])
            record["response"] = answer if isinstance(answer, str) else str(answer)
            if not isinstance(answer, str):
                raise ValueError("model returned no textual proposal")
            schema = Perception if state == "OBSERVE" else Proposal
            parsed = schema.model_validate(parse_json(answer))
            record["schema_valid"] = True
            return parsed
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.update(agent.model._last_metrics)
            record['exchanges'] = agent.model._exchanges
            record['http_requests'] = len(agent.model._exchanges)
            record["seconds"] = time.monotonic() - started
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with (self.log_dir / f"{self.session_id}.model.jsonl").open("a") as log:
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _offline_proposal(self):
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
        return Proposal.model_validate({
            "observation_id": t.obs["observation_id"], "memory_revision": m.revision,
            "purpose": "probe", "action": data,
            "experiment": {"question": f"Does {action} change the visible board?",
                           "alternatives": {"visible_effect": [{"kind": "frame_changed", "value": True}],
                                            "no_visible_effect": [{"kind": "frame_changed", "value": False}]},
                           "discriminator": "compare next real frame", "risk": "unknown control effects"},
        })

    def _build_graph(self):
        @node(rerun_on_resume=True)
        async def observe(ctx: Context):
            t = self.turn
            try:
                t.observe()
                if (not t.duplicate and t.memory.lifecycle == "ACTIVE" and self.model
                        and t.obs["state"] in ("NOT_FINISHED", "IN_PROGRESS") and t.obs.get("remaining_actions", 1) > 0
                        and not t.obs.get("evaluation_stop_reason")):
                    # One bounded formatting retry; preserve budget for the action proposal.
                    for attempt in range(2):
                        try:
                            t.apply_perception(await self._ask(ctx, "OBSERVE"))
                            break
                        except Exception as exc:
                            t.errors.append(f"perception {type(exc).__name__}: {exc}"[:500])
                            if attempt or t.calls >= t.max_calls - 1:
                                t.stop("perception_unavailable")
                                break
            except Exception as exc:
                t.errors.append(f"observation {type(exc).__name__}: {exc}"[:500])
                t.stop("invalid_observation")
            return Event(output="observed")

        def verify():
            if not self.turn.duplicate and self.turn.obs.get("observation_id"):
                self.turn.verify()
            return "verified"

        def update():
            self.turn.update()
            return Event(route=self.turn.route())

        def revise():
            self.turn.revise_model()
            return Event(route="PROBE" if self.turn.memory.unknowns or not self.turn.memory.goal else "PLAN")

        async def propose(ctx: Context, state: str):
            t = self.turn
            t.enter(state)
            if not self.model:
                try:
                    t.accept(self._offline_proposal())
                    return Event(route="ACT")
                except ValueError as exc:
                    t.errors.append(str(exc))
                    t.stop("no_legal_action")
                    return Event(route="COMMIT")
            while t.calls < t.max_calls and t.time_left() > 0:
                try:
                    p = await self._ask(ctx, "REVISE" if t.revise else state)
                    t.accept(p)
                    return Event(route="ACT")
                except Exception as exc:
                    t.errors.append(f"proposal {type(exc).__name__}: {exc}"[:500])
            t.stop("proposal_unavailable")
            return Event(route="COMMIT")

        @node(rerun_on_resume=True)
        async def probe(ctx: Context):
            return await propose(ctx, "PROBE")

        @node(rerun_on_resume=True)
        async def plan(ctx: Context):
            return await propose(ctx, "PLAN")

        def act():
            try:
                self.turn.act()
            except ValueError as exc:
                self.turn.errors.append(str(exc))
                if self.model and self.turn.calls < self.turn.max_calls and self.turn.time_left() > 0:
                    return Event(route="PLAN")
                self.turn.stop("action_validation_failed")
            return Event(route="COMMIT")

        def recover():
            self.turn.recover()
            return "recovered"

        def commit():
            result = self.turn.commit()
            return Event(output=result, state={"cognition": self.turn.memory.model_dump(mode="json")})

        routes = {"REVISE": revise, "PROBE": probe, "PLAN": plan, "ACT": act,
                  "RECOVER": recover, "COMMIT": commit}
        return Workflow(name="cognitive_workflow", edges=[
            ("START", observe, verify, update), (update, routes),
            (revise, {"PROBE": probe, "PLAN": plan}),
            (probe, {"ACT": act, "COMMIT": commit}), (plan, {"ACT": act, "COMMIT": commit}),
            (act, {"PLAN": plan, "COMMIT": commit}), (recover, commit),
        ])

    async def _decide(self, obs: dict) -> dict:
        if not self.initialized:
            await self.service.create_session(app_name=APP_NAME, user_id="player", session_id=self.session_id,
                                              state={"cognition": self.memory.model_dump(mode="json")})
            self.initialized = True
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            record = {k: v for k, v in obs.items() if k != "image_png_base64"}
            if obs.get("image_png_base64"):
                frames_dir = self.log_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                frame_name = f"{self.session_id}-{obs['step']:05d}.png"
                (frames_dir / frame_name).write_bytes(base64.b64decode(obs["image_png_base64"]))
                record["image_path"] = "frames/" + frame_name
            with (self.log_dir / f"{self.session_id}.observations.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
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
                self.closed = True
