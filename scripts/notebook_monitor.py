"""On-demand Jupyter replay of a selected saved benchmark. No live runner or polling."""
from __future__ import annotations

from bisect import bisect_left
from html import escape
import io
import json
from pathlib import Path

import ipywidgets as W
from IPython.display import display
from PIL import Image, ImageDraw

from scripts.agent_monitor import (Timeline, discover_evaluations, discover_runs, dashboard_html,
                                  json_html, safe_asset, PALETTE)
from scripts.cognition_view import cognition_html
from scripts.runtime_structure import read_structure, structure_html, failure_points, trace_html


def picture(observation, directory, action=None):
    """A single PNG, avoiding thousands of SVG nodes for a pixel grid."""
    if observation is None:
        return b''
    origin, scale = (0,0), 6
    data = None
    if observation.get('image_path'):
        path = safe_asset(directory,observation['image_path'])
        data = path.read_bytes()
        view = observation.get('viewport') or {}
        origin, scale = view.get('origin'), view.get('scale')
    if data is None:
        grid = observation.get('grid')
        if not grid:
            return b''
        image = Image.new('RGB',(len(grid[0]),len(grid)))
        pixels = [tuple(int(PALETTE[v][i:i+2],16) for i in (1,3,5)) if type(v) is int else tuple(v[:3])
                  for row in grid for v in row]
        image.putdata(pixels)
        image = image.resize((image.width*6,image.height*6),Image.Resampling.NEAREST)
    else:
        image = None
    if action and action.get('action') in ('CLICK','ACTION6') and origin is not None and scale:
        x,y = action.get('x'),action.get('y')
        if type(x) is int and type(y) is int:
            image = Image.open(io.BytesIO(data)).convert('RGB') if image is None else image
            cx,cy = origin[0]+(x+.5)*scale,origin[1]+(y+.5)*scale
            radius = 2*scale
            draw = ImageDraw.Draw(image)
            draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='white',width=3)
            draw.ellipse((cx-radius+1,cy-radius+1,cx+radius-1,cy+radius-1),outline='#ff315b',width=1)
    if image is None:
        return data
    stream=io.BytesIO();image.save(stream,format='PNG')
    return stream.getvalue()


class BenchmarkReplay:
    def __init__(self, root, source_root=None):
        self.root = Path(root).expanduser().resolve()
        self.timeline, self.runs = None, {}
        self.source_root = Path(source_root) if source_root else Path(__file__).resolve().parents[1]
        self.structure = None
        self.run_settings = ''
        self.structure_refresh = W.Button(description='最新の構造を再読込', icon='refresh')
        self.structure_status = W.HTML()
        self.problem = W.Dropdown(description='問題箇所', options=[('全記録の問題イベントを選択',None)], layout=W.Layout(width='100%'))
        self.trace = W.HTML()
        self._updating, self._closed = False, False
        self._positions, self._image_keys = [], [None,None]
        self._frames, self._animation_key = [], None
        self.evaluation = W.Dropdown(description='評価ID',layout=W.Layout(width='100%'))
        self.game = W.Dropdown(description='ゲーム',layout=W.Layout(width='100%'))
        self.load_button = W.Button(description='読み込む',icon='folder-open',button_style='primary')
        self.refresh_button = W.Button(description='評価一覧を更新',icon='refresh')
        self.mode = W.Dropdown(description='再生単位',options=['状態遷移','操作','イベント'],layout=W.Layout(width='210px'))
        self.play = W.Play(min=0,max=0,value=0,interval=1000,repeat=False,disabled=True)
        self.slider = W.IntSlider(min=0,max=0,value=0,description='位置',continuous_update=False,layout=W.Layout(width='75%'))
        self.speed = W.Dropdown(description='間隔',options=[('2秒',2000),('1秒',1000),('0.5秒',500)],value=1000,layout=W.Layout(width='180px'))
        self.previous,self.next = W.Button(description='前へ'),W.Button(description='次へ')
        self.notes_button = W.Button(description='計画・スキルを読む', icon='book', disabled=True)
        self.status,self.board = W.HTML(),W.HTML()
        self.diagram = W.HTML()
        self._machine_key = None
        self.images = [W.Image(format='png',layout=W.Layout(width='100%',height='340px',object_fit='contain')) for _ in range(2)]
        self.captions = [W.HTML(),W.HTML()]
        for image in self.images:
            image.add_class('arc-replay-image')
        screens = W.HBox([W.VBox([label,img],layout=W.Layout(width='50%')) for label,img in zip(self.captions,self.images)])
        self.panels = [W.HTML() for _ in range(7)]
        self.animation = W.Image(format='png',layout=W.Layout(width='384px',height='384px',object_fit='contain'))
        self.animation_caption = W.HTML()
        self.animation_slider = W.IntSlider(min=0,max=0,description='フレーム',continuous_update=False)
        self.details = W.Accordion(children=[*self.panels,W.VBox([self.animation_slider,self.animation_caption,self.animation])])
        for i,name in enumerate(('状態の入力','状態の出力','選択イベント','HTTP入力','モデル応答','計画ツール','計画・スキル・熟考への復帰','記録済みアニメーション')):
            self.details.set_title(i,name)
        self.details.selected_index = None
        self.widget = W.VBox([W.HTML('<h3>ベンチマーク再生</h3><style>.arc-replay-image img{image-rendering:pixelated}</style>'),
                              self.evaluation,self.game,W.HBox([self.load_button,self.refresh_button]),
                              W.HBox([self.mode,self.speed,self.previous,self.next,self.notes_button]),W.HBox([self.play,self.slider]),
                              self.structure_refresh,self.structure_status,self.diagram,
                              self.problem,self.status,self.trace,screens,self.board,self.details])
        self._link = W.jslink((self.play,'value'),(self.slider,'value'))
        self.evaluation.observe(self._select_evaluation,names='value')
        self.game.observe(lambda _: self._clear(),names='value')
        self.load_button.on_click(lambda _: self.load())
        self.refresh_button.on_click(lambda _: self.refresh_evaluations())
        self.slider.observe(lambda _: self.render() if not self._updating else None,names='value')
        self.mode.observe(lambda _: self._set_positions(),names='value')
        self.speed.observe(lambda change: setattr(self.play,'interval',change['new']),names='value')
        self.previous.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        self.notes_button.on_click(lambda _: setattr(self.details, 'selected_index', 6))
        self.play.observe(self._playing,names='playing')
        self.details.observe(self._open_detail,names='selected_index')
        self.animation_slider.observe(lambda _: self._render_animation() if not self._updating else None,names='value')
        self.structure_refresh.on_click(lambda _: self._read_structure())
        self.problem.observe(self._jump_problem, names='value')
        self.refresh_evaluations()

    def refresh_evaluations(self):
        previous=self.evaluation.value
        evaluations=discover_evaluations(self.root)
        self._updating=True
        try:
            self.evaluation.options=[(p.name,str(p)) for p in evaluations]
            self.evaluation.value=previous if previous in [str(p) for p in evaluations] else str(evaluations[0]) if evaluations else None
        finally:
            self._updating=False
        self._select_evaluation(None)
        self._read_structure()

    def _select_evaluation(self, _):
        if self._updating:
            return
        runs=discover_runs(self.evaluation.value) if self.evaluation.value else []
        self.runs={r.run_id:r for r in runs}
        self._updating=True
        try:
            self.game.options=[(r.directory.parent.name+' · '+r.run_id[:10],r.run_id) for r in runs]
            self.game.value=runs[0].run_id if runs else None
        finally:
            self._updating=False
        self._clear()
        self._read_structure()

    def _clear(self):
        if self._updating:
            return
        self.play.playing=False
        self._updating=True
        try:
            self.slider.value=self.play.value=0
            self.slider.max=self.play.max=0
        finally:
            self._updating=False
        self.timeline=None
        self._positions=[]
        self._image_keys=[None,None]
        self._machine_key=None
        self.diagram.value=structure_html(self.structure) if self.structure else ''
        self.trace.value=''
        self.problem.options=[('全記録の問題イベントを選択',None)]
        self._frames,self._animation_key=[],None
        self.details.selected_index=None
        for panel in self.panels:
            panel.value=''
        for img,label in zip(self.images,self.captions):
            img.value=b'';label.value=''
        self.animation.value=b''
        self.play.disabled=True
        self.notes_button.disabled=True
        self.load_button.disabled=not self.runs
        self.board.value=''
        self.status.value='<p>評価IDとゲームを選択して「読み込む」を押してください。</p>' if self.runs else '<p>再生可能な評価がありません。make benchmark の終了後に「評価一覧を更新」を押してください。</p>'

    def load(self):
        run=self.runs.get(self.game.value)
        if run is None:
            return
        self._clear()
        self.status.value='<p>選択したゲームの記録を読み込んでいます。</p>'
        try:
            manifest=json.loads((Path(self.evaluation.value)/'manifest.json').read_text())
            if manifest.get('observatory_schema') != 3:
                raise ValueError('旧形式の実行ログです。最新実装で記録を作成してください')
            settings=manifest.get('cognition_settings') or {}
            self.run_settings=('実行時のモデル: '+str(manifest.get('agent_model','記録なし'))
                +' · 一観測の判断時間: '+str(settings.get('COGNITION_DECISION_SECONDS','45'))+'秒'
                +' · 計画の出力修正: '+str(settings.get('COGNITION_REPAIR_ATTEMPTS','1'))+'回'
                +' · 高速選択: 1トークン')
            self.timeline=Timeline(run).load()
            self.problem.options=[('全記録の問題イベントを選択',None), *failure_points(self.timeline.events)]
            self._read_structure()
            self._set_positions()
        except (OSError,ValueError) as exc:
            self.timeline=None
            self.status.value='<p>読込エラー: '+escape(str(exc))+'</p>'

    def _read_structure(self):
        try:
            self.structure=read_structure(self.source_root)
            self._machine_key=None
            self.structure_status.value='<p>最新実装の全体図 · '+str(len(self.structure['states']))+'ステート / '+str(len(self.structure['transitions']))+'遷移。ログ未選択でも表示します。</p>'
            self.diagram.value=structure_html(self.structure,(self.snapshot() or {}).get('machine'))
        except (OSError,ValueError,SyntaxError) as exc:
            self.structure=None
            self.structure_status.value='<p>構造読込エラー: '+escape(str(exc))+'</p>'
            self.diagram.value=''

    def _jump_problem(self, change):
        if self._updating or change['new'] is None or not self.timeline:
            return
        self.play.playing=False
        self.mode.value='イベント'
        self.slider.value=change['new']

    def _set_positions(self):
        if self._updating or not self.timeline:
            return
        self.play.playing=False
        old=self._positions[self.slider.value] if self._positions and self.slider.value<len(self._positions) else 0
        events=self.timeline.events
        if self.mode.value=='イベント':
            self._positions=list(range(len(events)))
        elif self.mode.value=='状態遷移':
            self._positions=[i for i,s in enumerate(self.timeline.snapshots)
                             if i==0 or s['machine']!=self.timeline.snapshots[i-1]['machine']
                             or i==len(events)-1]
        else:
            # End of each recorded step; keep the final observation and stop visible.
            self._positions=[i for i,e in enumerate(events) if i==len(events)-1 or e.get('step')!=events[i+1].get('step')]
        self._updating=True
        try:
            self.slider.value=self.play.value=0
            self.slider.max=self.play.max=max(0,len(self._positions)-1)
            self.slider.value=min(bisect_left(self._positions,old),self.slider.max)
            self.play.disabled=len(self._positions)<2
        finally:
            self._updating=False
        self.render()

    def _move(self, delta):
        self.play.playing=False
        self.slider.value=max(0,min(self.slider.max,self.slider.value+delta))

    def _playing(self, change):
        if change['new']:
            self.details.selected_index=None

    def _open_detail(self, change):
        if change['new'] is not None:
            self.play.playing=False
            self._render_detail()

    def snapshot(self):
        if self.timeline and self._positions:
            return self.timeline.snapshot(self._positions[self.slider.value])

    def render(self):
        if self._closed or self._updating:
            return
        s=self.snapshot()
        if s is None:
            self.status.value='<p>再生できるイベントがありません。</p>'
            return
        current=read_structure(self.source_root)
        if not self.structure or current['hash']!=self.structure['hash']:
            self._read_structure()
        directory=self.timeline.run.directory
        self.notes_button.disabled=False
        self.status.value=f'<p><b>記録の再生</b> · {escape(directory.parent.parent.name)} · {escape(directory.parent.name)} · {self.slider.value+1}/{len(self._positions)}<br>再生はゲームやモデルを実行しません。詳細を開くと一時停止します。</p>'
        self.status.value += '<p>'+escape(self.run_settings)+'</p>'
        if self.timeline.invalid_lines:
            self.status.value+=f'<p>不正な完了行を{self.timeline.invalid_lines}件除外しました。</p>'
        machine_key=tuple(s['machine'].items())
        if machine_key!=self._machine_key:
            self.diagram.value=structure_html(self.structure,s['machine']) if self.structure else ''
            self._machine_key=machine_key
        for i,(label,obs) in enumerate([('直前の観測',s['before']),('この時点の最新観測',s['current'])]):
            action=s['incoming_action'] if i==0 else s['action'] if obs and obs.get('step')==s['action_step'] else None
            key=(obs.get('observation_id',obs.get('sequence')) if obs else None,json.dumps(action,sort_keys=True))
            if key!=self._image_keys[i]:
                self.images[i].value=picture(obs,directory,action)
                self._image_keys[i]=key
            self.captions[i].value=f'<b>{label}</b> · step {obs.get("step","—") if obs else "—"}'
        self.trace.value=trace_html(self.timeline.events,s['index'])
        self.board.value=dashboard_html(s,directory,include_images=False)
        if self.details.selected_index is not None:
            self._render_detail()

    def _render_detail(self):
        s=self.snapshot();index=self.details.selected_index
        if s is None or index is None:
            return
        if index==6:
            self.panels[index].value=cognition_html(s)
            return
        if index==7:
            obs=s['current'] or {}
            key=obs.get('observation_id')
            if key!=self._animation_key:
                self._animation_key,self._frames=key,[]
                if obs.get('animation_archive_path'):
                    self._frames=json.loads(safe_asset(self.timeline.run.directory,obs['animation_archive_path']).read_text()).get('frames',[])
                self._updating=True
                try:
                    self.animation_slider.value=0
                    self.animation_slider.max=max(0,len(self._frames)-1)
                finally:
                    self._updating=False
            self._render_animation()
            return
        values=[s['input'],s['output'] if s['output'] is not None else 'この時点の出力はまだ記録されていません。',
                s['event'],s['request'],s['response'],{'tools':s['tools']}]
        self.panels[index].value=json_html(values[index])

    def _render_animation(self):
        if self.details.selected_index!=7:
            return
        if not self._frames:
            self.animation.value=b'';self.animation_caption.value='途中フレームの記録なし'
            return
        index=self.animation_slider.value
        self.animation.value=picture({'grid':self._frames[index]},self.root)
        self.animation_caption.value=f'記録済み遷移: {index+1}/{len(self._frames)} フレーム'

    def show(self):
        display(self.widget)
        return self

    def close(self):
        self.play.playing=False
        self._closed=True
        self._link.unlink()
        def dispose(widget):
            for child in getattr(widget,'children',()):
                dispose(child)
            widget.close()
        dispose(self.widget)
        self.timeline=None
