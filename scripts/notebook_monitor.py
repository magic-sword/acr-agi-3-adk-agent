"""Jupyter widgets for live observation and replay of exactly the same journals."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from html import escape
import math
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import ipywidgets as W
from IPython.display import display

from scripts.agent_monitor import Timeline, discover_runs, dashboard_html, json_html, safe_asset, screen_html


class AgentMonitor:
    def __init__(self, root, *, project=None, interval=1.0):
        self.project = Path(project or Path(__file__).resolve().parents[1]).resolve()
        if not math.isfinite(interval) or interval < .2:
            raise ValueError('interval must be at least 0.2 seconds')
        self.interval, self.root = interval, Path(root).expanduser().resolve()
        self.timeline, self.runs = None, {}
        self.process = self.process_log = None
        self._task, self._updating, self._closed = None, False, False
        self._last_discovery, self._last_poll = 0., ''
        self.run_select = W.Dropdown(description='走行', layout=W.Layout(width='100%'))
        self.follow = W.Checkbox(value=False, description='最新を追従', indent=False)
        self.refresh_button = W.Button(description='再読込', icon='refresh')
        self.stop_button = W.Button(description='実行を停止', icon='stop', disabled=True, button_style='warning')
        self.play = W.Play(value=0, min=0, max=0, interval=400, description='再生', disabled=True)
        self.slider = W.IntSlider(value=0, min=0, max=0, description='イベント', continuous_update=False,
                                  layout=W.Layout(width='75%'))
        self.speed = W.Dropdown(options=[('ゆっくり',1000),('標準',400),('速い',100)], value=400,
                                description='再生速度', layout=W.Layout(width='220px'))
        self.prev_step, self.next_step = W.Button(description='前の操作'), W.Button(description='次の操作')
        self.status, self.board = W.HTML(), W.HTML()
        self.panels = [W.HTML() for _ in range(6)]
        self.animation = W.HTML()
        self.animation_slider = W.IntSlider(min=0,max=0,description='フレーム')
        self.animation_play = W.Play(min=0,max=0,interval=150,disabled=True)
        self._animation_key, self._frames = None, []
        self._animation_link = W.jslink((self.animation_play,'value'),(self.animation_slider,'value'))
        self.animation_slider.observe(lambda _: self._render_animation(),names='value')
        self.tabs = W.Tab(children=[*self.panels,W.VBox([W.HBox([self.animation_play,self.animation_slider]),self.animation])])
        for i, title in enumerate(('状態の入力', '状態の出力', '選択イベント', 'HTTP入力', 'モデル応答', 'ツール・学習')):
            self.tabs.set_title(i, title)
        self.tabs.set_title(6,'記録済みアニメーション')
        self.process_output = W.HTML()
        self.log_panel = W.Accordion(children=[self.process_output])
        self.log_panel.set_title(0,'起動・終了ログ')
        self.log_panel.selected_index = None
        self.widget = W.VBox([W.HTML('<h3>Agent Observatory</h3><p>画面・状態・操作を同じ時点で追跡します。'
                                   'モデルが出力した記録を表示し、出力されていない内部思考は補いません。</p>'),
                              self.run_select, W.HBox([self.follow,self.refresh_button,self.stop_button]),
                              W.HBox([self.play,self.slider]), W.HBox([self.prev_step,self.next_step,self.speed]),
                              self.status,self.board,self.tabs,self.log_panel])
        self._link = W.jslink((self.play,'value'),(self.slider,'value'))
        self.run_select.observe(self._choose_run, names='value')
        self.slider.observe(self._seek, names='value')
        self.speed.observe(lambda change: setattr(self.play,'interval',change['new']), names='value')
        self.follow.observe(self._follow_changed, names='value')
        self.refresh_button.on_click(lambda _: self.refresh(discover=True))
        self.stop_button.on_click(lambda _: self.stop())
        self.prev_step.on_click(lambda _: self.step(-1))
        self.next_step.on_click(lambda _: self.step(1))
        self.refresh(discover=True)

    def _choose_run(self, change):
        if self._updating:
            return
        self.timeline = Timeline(self.runs[change['new']]) if change['new'] in self.runs else None
        self.refresh()

    def _seek(self, change):
        if not self._updating:
            self.follow.value = False
            self.render()

    def _follow_changed(self, change):
        if change['new']:
            self.play.playing = False
            self.refresh()

    def step(self, direction):
        if not self.timeline or not self.timeline.events:
            return
        self.follow.value = False
        current = self.timeline.events[self.slider.value].get('step')
        indices = range(self.slider.value+1,len(self.timeline.events)) if direction>0 else range(self.slider.value-1,-1,-1)
        for i in indices:
            if self.timeline.events[i].get('step') != current:
                self.slider.value = i
                return

    def refresh(self, *, discover=False):
        if self._closed:
            return
        changed = False
        if discover or not self.runs or time.monotonic()-self._last_discovery > 5:
            runs = discover_runs(self.root)
            mapped = {str(r.directory / r.run_id):r for r in runs}
            if mapped != self.runs:
                self.runs = mapped
                previous = self.run_select.value
                self._updating = True
                try:
                    self.run_select.options = [(r.label,k) for k,r in mapped.items()]
                    self.run_select.value = previous if previous in mapped else next(reversed(mapped),None)
                finally:
                    self._updating = False
                self.timeline = Timeline(mapped[self.run_select.value]) if mapped else None
                changed = True
            self._last_discovery = time.monotonic()
        if self.timeline:
            changed = self.timeline.refresh() or changed
            maximum = max(0,len(self.timeline.events)-1)
            self._updating = True
            try:
                self.play.max = self.slider.max = maximum
                if self.follow.value:
                    self.slider.value = maximum
                self.play.disabled = maximum == 0
            finally:
                self._updating = False
        self.stop_button.disabled = self.process is None or self.process.poll() is not None
        self._last_poll = datetime.now().strftime('%H:%M:%S')
        self.render()
        if self.process_log and self.process_log.exists():
            with self.process_log.open('rb') as stream:
                stream.seek(max(0,self.process_log.stat().st_size-8000))
                tail = stream.read().decode('utf-8',errors='replace')
            self.process_output.value = json_html(tail or 'プロセスを起動しました。出力を待っています。')
        return changed

    def render(self):
        s = self.timeline.snapshot(self.slider.value) if self.timeline else None
        mode = '最新を追従中' if self.follow.value else '履歴再生（位置を固定）'
        process = ''
        if self.process:
            code = self.process.poll()
            process = ' · 評価プロセス実行中' if code is None else f' · 評価プロセス終了: code={code}'
        invalid = sum(j.invalid_lines for j in self.timeline.journals.values()) if self.timeline else 0
        self.status.value = (f'<p><b>{mode}{process}</b> · 最終読込 {self._last_poll} '
                             f'· {escape(str(self.root))}<br>記録の時刻はUTC。再生速度は表示用で、ゲームを進めません。'
                             + (f' 不正な完了行を{invalid}件スキップしました。' if invalid else '')+'</p>')
        directory = self.timeline.run.directory if self.timeline else self.root
        self.board.value = dashboard_html(s,directory)
        values = ([s['input'], s['output'] if s['output'] is not None else 'この時点で状態の出力はまだ記録されていません。',
                   s['event'],s['request'],s['response'],{'tools':s['tools'],'learning':s['learning']}]
                  if s else ['記録を待っています。']*6)
        for panel, value in zip(self.panels,values):
            panel.value = json_html(value)
        observation = (s or {}).get('current') or {}
        key = (str(directory),observation.get('observation_id'),observation.get('animation_archive_path'))
        if key != self._animation_key:
            self._animation_key, self._frames = key, []
            self.animation_play.playing = False
            if observation.get('animation_archive_path'):
                try:
                    self._frames = json.loads(safe_asset(directory,observation['animation_archive_path']).read_text()).get('frames',[])
                except (OSError,ValueError):
                    pass
            self.animation_play.max = self.animation_slider.max = max(0,len(self._frames)-1)
            self.animation_slider.value = 0
            self.animation_play.disabled = len(self._frames)<2
            self._render_animation()

    def _render_animation(self):
        if not self._frames:
            self.animation.value = '<p>この観測には途中フレームの記録がありません。</p>'
            return
        index = min(self.animation_slider.value,len(self._frames)-1)
        frame = self._frames[index]
        # Stored RGB frames are converted only for presentation; the game is not stepped.
        if frame and frame[0] and isinstance(frame[0][0],list):
            import base64,io
            from PIL import Image
            image = Image.new('RGB',(len(frame[0]),len(frame)))
            image.putdata([tuple(pixel[:3]) for row in frame for pixel in row])
            stream=io.BytesIO();image.save(stream,format='PNG')
            picture='<img style="width:384px;image-rendering:pixelated" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'
        else:
            picture=screen_html({'grid':frame},self.root)
        self.animation.value = ('<p>記録済みの遷移 · '+str(index+1)+'/'+str(len(self._frames))+
                                ' フレーム。表示間隔は実時間ではありません。</p>'+picture)

    async def _watch(self):
        while not self._closed:
            await asyncio.sleep(self.interval)
            try:
                self.refresh()
            except Exception as exc:
                self.status.value = '<p>ログ読込エラー: '+escape(str(exc))+'（次の更新で再試行）</p>'

    def show(self):
        display(self.widget)
        if self._task is None or self._task.done():
            try:
                self._task = asyncio.get_running_loop().create_task(self._watch())
            except RuntimeError:
                self.status.value += '<p>自動更新にはJupyterカーネルが必要です。再読込ボタンは利用できます。</p>'
        return self

    def start(self, *, game='ls20', steps=30, seconds=180, model=True, learning=True):
        """Start the existing benchmark in an owned process group, without blocking the kernel."""
        if self._closed:
            raise RuntimeError('monitor is closed')
        if self.process and self.process.poll() is None:
            raise RuntimeError('this monitor already has a running evaluation')
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)?',game) or type(steps) is not int or steps<1:
            raise ValueError('supply one game ID and positive integer steps')
        if not math.isfinite(seconds) or seconds<=0:
            raise ValueError('seconds must be positive and finite')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = self.project/'outputs/evaluations'/f'{stamp}-monitor'
        logs = self.project/'outputs/monitor-process'
        logs.mkdir(parents=True,exist_ok=True)
        self.process_log = logs/f'{stamp}.log'
        command = [sys.executable,'-u',str(self.project/'scripts/benchmark_local.py'),
                   '--games',game,'--steps',str(steps),'--seconds',str(seconds),
                   '--hard-seconds',str(seconds+30),'--output',str(output)]
        if not model:
            command.append('--offline-policy')
        if not learning:
            command.append('--no-learning')
        with self.process_log.open('w') as log:
            self.process = subprocess.Popen(command,cwd=self.project,stdout=log,stderr=subprocess.STDOUT,
                                            start_new_session=True)
        self.root = output
        self.timeline, self.runs = None, {}
        self._updating = True
        try:
            self.run_select.options = []
            self.slider.value = self.play.value = 0
            self.slider.max = self.play.max = 0
            self.play.playing = False
        finally:
            self._updating = False
        self.follow.value = True
        self.refresh(discover=True)
        return output

    def stop(self):
        """Stop only the evaluation launched by this instance, including its worker."""
        if self.process is None or self.process.poll() is not None:
            return
        proc = self.process
        try:
            os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:
            return
        # No blocking wait on the notebook UI thread.
        async def reap():
            await asyncio.sleep(2)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(proc.wait)
        try:
            asyncio.get_running_loop().create_task(reap())
        except RuntimeError:
            proc.wait(timeout=3)
        self.refresh()

    def close(self, *, stop=False):
        if stop:
            self.stop()
        self._closed = True
        if self._task:
            self._task.cancel()
        self._link.unlink()
        self._animation_link.unlink()
        self.widget.close()
