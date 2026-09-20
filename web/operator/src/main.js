import * as THREE from 'three';
import {clamp,mapPad,waveHeight} from './input.js';
import '@fontsource/noto-sans-jp/400.css';
import '@fontsource/noto-sans-jp/500.css';
import '@fontsource/space-grotesk/400.css';
import '@fontsource/space-grotesk/600.css';
import './style.css';
const $=id=>document.getElementById(id);
$('app').innerHTML=`
<header><a class="brand" href="/">KMT<span>OPERATOR LAB</span></a><div class="tag">HUMAN-IN-THE-LOOP <i>EXPERIMENTAL / 01</i></div><div id="connection" class="connection">● 接続中</div></header>
<main><aside>
<div class="eyebrow">TRAINING CONSOLE</div><h1>水面から、<br>操縦を学ぶ。</h1><p class="intro">飛行艇の判断と手順を練習する、<br>ブラウザ操縦席。</p>
<section><div class="section-label">01 <span>セッション設定</span></div>
<label>操作モード<select id="mode"><option value="hybrid">HYBRID · 自動離水＋手動飛行</option><option value="manual">MANUAL · 全区間手動</option><option value="observe">OBSERVE · 自動飛行を見学</option></select></label>
<div class="pair"><label>開始地点<select id="exercise"><option value="takeoff">水上・離水</option><option value="cruise">高度25 m・巡航</option><option value="landing">高度25 m・進入</option></select></label><label>有義波高<select id="hs"><option value="0.3">0.3 m</option><option value="0">0 m · 平水</option><option value="0.6">0.6 m</option></select></label></div>
<label>波のシード<input id="seed" type="number" value="42" min="0" max="999999"></label>
</section>
<section><div class="section-label">02 <span>入力デバイス</span></div>
<div class="segmented"><button id="use-key" class="selected">キーボード</button><button id="use-pad">ゲームパッド</button></div>
<p id="pad-status" class="small">デモ操作：矢印キー / W・S / A・D</p>
<div id="calibration" hidden><p class="small">標準mapping専用。スティックを離し、トリガを戻してください。非標準機器は未対応です。</p><button id="cal-neutral" class="secondary">① 中立を記録</button><button id="cal-range" class="secondary" disabled>② 全軸を動かしたら範囲を確定</button><label class="check"><input type="checkbox" id="cal-confirm">手前＝機首上げ、右＝右バンク、トリガ＝推力を確認</label><div id="raw-pad" class="small mono"></div></div>
<button id="create" class="primary">セッションを作成 <span>↗</span></button><button id="reconnect" class="secondary" hidden>再接続する</button>
</section>
<div class="limits"><span>SIMULATION, NOT CERTIFICATION</span><p>指令追従型の簡略モデルです。実機の操縦感覚・資格を再現するものではありません。</p></div>
</aside>
<article>
<div class="flight-header"><div><span class="eyebrow">LIVE FLIGHT DECK</span><h2 id="flight-title">操縦席 / Standby</h2></div><div class="badges"><span id="authority">NO SESSION</span><span id="lifecycle">SETUP</span></div></div>
<div class="viewport" id="viewport"><canvas id="scene"></canvas><div class="view-top"><span>NEU / 空間モデル</span><button id="camera">追従視点 ⇄ 前方視点</button></div><div class="crosshair">+</div><div id="overlay"><span>READY TO EXPLORE</span><strong>あなたの操作が、飛行になる。</strong><p>左の設定からセッションを作成してください。</p></div><div class="view-bottom"><span id="phase">—</span><span id="clock">SIM 00:00.00</span><span id="rtt">RTT —</span></div></div>
<div class="instruments">
<div><label>対気速度</label><strong id="airspeed">—</strong><small>m/s</small></div><div><label>高度 / 平均水面</label><strong id="altitude">—</strong><small>m</small></div><div><label>昇降率</label><strong id="vspeed">—</strong><small>m/s</small></div><div><label>波面クリアランス</label><strong id="clearance">—</strong><small>m · 船底基準</small></div><div><label>方位 / バンク</label><strong id="heading">—</strong><small id="bank">— °</small></div>
</div>
<div class="bottom-grid"><section class="control-panel"><div class="section-label">03 <span>操作と引継ぎ</span></div><div id="instruction" class="instruction">接続を待っています。</div>
<div class="actions"><button id="start" class="primary">飛行開始</button><button id="pause" class="secondary">一時停止</button><button id="resume" class="secondary">再開</button><button id="take" class="secondary">操縦を引き受ける</button><button id="land" class="secondary">自動着水を要求</button><button id="confirm-land" class="secondary" hidden>条件を確認して着水AUTOへ</button><button id="cancel" class="secondary" hidden>要求を取り消す</button><button id="abort" class="quiet">試行を終了</button></div>
<p id="gate" class="small"></p><div id="notice" role="status" aria-live="polite">操作権限はサーバーが管理します。</div></section>
<section class="input-panel"><div class="section-label">04 <span>入力モニター</span><small>要求 / 適用</small></div>
${[['throttle','推力',0,1,.01],['pitch_deg','ピッチ',-8,12,.1],['bank_deg','バンク',-25,25,.1],['rudder','ラダー',-1,1,.01]].map(([k,l,min,max,step])=>`<label class="axis-label">${l}<span id="value-${k}">0 / 0</span><input id="axis-${k}" type="range" min="${min}" max="${max}" step="${step}" value="0"></label>`).join('')}
<p class="small">↑↓ ピッチ · ←→ バンク · W/S 推力 · A/D ラダー<br>Space 停止 · キーを離すと姿勢入力は中立に戻ります</p></section></div>
<section class="review" id="review" hidden><div class="section-label">SESSION REVIEW <span id="result"></span></div><div id="review-metrics"></div><button class="secondary" id="download">結果JSONを保存</button><p class="small">初接触までの実験用評価です。水上停止の成功を意味しません。全記録はサーバーの var/operator_sessions/ に保存。</p></section>
<footer><span>KMT / FLYING BOAT SIMULATOR</span><span>物理 20 Hz · 時刻同期海面 · 実パッド検証前</span></footer>
</article></main>`;

let socket,epoch=0,sessionId=null,seq=0,state=null,previous=null,received=0,terms=[];
let inputDevice='keyboard',padIndex=null,profile=null,calStage=0,axes=[],ranges=[],trigger=[1,0];
let control={throttle:0,pitch_deg:0,bank_deg:0,rudder:0},keys=new Set(),lastButtons=[],lastReceive=0,lagSince=0;

function notice(t){$('notice').textContent=t;}
function send(type,data={}){if(socket?.readyState===1)socket.send(JSON.stringify({v:1,epoch,type,session_id:sessionId,...data}));}
function event(action){if(sessionId)send('event',{action,event_id:crypto.randomUUID()});}
async function connect(){
  await fetch('/api/capabilities');
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);
  socket.onopen=()=>{$('connection').textContent='● 接続済み';$('connection').className='connection online';$('reconnect').hidden=true;lastReceive=performance.now();};
  socket.onclose=()=>{$('connection').textContent='● 切断 / 操縦停止';$('connection').className='connection';$('reconnect').hidden=false;notice('接続が切れました。埋込み画面の場合は別タブで開いてください。重複接続・Cookie・サーバー状態も確認してください。');};
  socket.onmessage=({data})=>{
    const m=JSON.parse(data); lastReceive=performance.now();
    if(m.type==='hello'||m.type==='created'){epoch=m.epoch;sessionId=m.session_id;seq=0;terms=m.spectrum;
      if(m.config){for(const key of ['mode','exercise','hs','seed'])$(key).value=m.config[key];
        inputDevice=m.config.input_device||'keyboard';
        $('use-key').classList.toggle('selected',inputDevice==='keyboard');
        $('use-pad').classList.toggle('selected',inputDevice==='gamepad');
        $('calibration').hidden=inputDevice!=='gamepad';}  if(m.type==='created'){state=previous=null;notice('設定完了。入力を確認して「飛行開始」を押してください。');} }
    if(m.type==='state'){previous=state;state=m;received=performance.now();update();}
    if(m.type==='ack')notice(m.accepted?`受付：${m.action} / tick ${m.tick}`:m.reason);
    if(m.type==='pong'){
      const rtt=performance.now()-m.client_time;$('rtt').textContent=`RTT ${Math.round(rtt)} ms`;
      if(rtt>250){lagSince||=performance.now();if(performance.now()-lagSince>1000)event('pause');}else lagSince=0;
    }
  };
}
$('reconnect').onclick=()=>connect().catch(e=>notice(e.message));
connect().catch(e=>notice('接続できません：'+e.message));
setInterval(()=>{if(!document.hidden)send('heartbeat',{client_time:performance.now()});},200);
setInterval(()=>{
  if(sessionId&&!document.hidden&&(inputDevice==='keyboard'||padReady()))send('input',{seq:seq++,control});
  if(state?.lifecycle==='RUNNING'&&performance.now()-lastReceive>500){event('pause');notice('状態配信が途絶えました。表示は最後の受信状態です。');}
},34);
function pad(){return [...(navigator.getGamepads?.()||[])].find(p=>p&&p.index===padIndex);}
function padReady(){return pad()&&profile&&$('cal-confirm').checked&&calStage===3;}
function device(kind){
  if(state&&!['FINISHED','ABORTED'].includes(state.lifecycle)){event('pause');notice('入力デバイスの変更は現在の試行を終了してから行ってください。');return;}
  inputDevice=kind;keys.clear();$('use-key').classList.toggle('selected',kind==='keyboard');$('use-pad').classList.toggle('selected',kind==='gamepad');$('calibration').hidden=kind!=='gamepad';
}
$('use-key').onclick=()=>device('keyboard');$('use-pad').onclick=()=>device('gamepad');
window.addEventListener('gamepadconnected',e=>{if(padIndex===null){padIndex=e.gamepad.index;notice('ゲームパッドを検出しました。入力デバイスを選び校正してください。');}});
window.addEventListener('gamepaddisconnected',e=>{if(e.gamepad.index===padIndex){event('pause');padIndex=null;profile=null;calStage=0;$('cal-confirm').checked=false;notice('ゲームパッド切断：再校正が必要です。');}});
$('cal-neutral').onclick=()=>{
  if(state?.lifecycle==='RUNNING'){event('pause');notice('停止後に校正してください。');return;}
  const p=pad(); if(!p||p.mapping!=='standard'||p.axes.length<3||p.buttons.length<8){notice('標準mappingのパッドを接続しボタンを押してください。');return;}
  axes=p.axes.slice(0,3);ranges=axes.map(x=>[x,x]);trigger=[p.buttons[7].value,p.buttons[7].value];calStage=1;profile=null;$('cal-confirm').checked=false;$('cal-range').disabled=false;notice('両スティックを全方向へ、右トリガを最大まで動かして戻してください。');
};
$('cal-range').onclick=()=>{
  if(ranges.some(([a,b])=>b-a<1.2)||trigger[1]-trigger[0]<.8){notice('全範囲が不足しています。左右・上下・右トリガを最後まで動かしてください。');return;}
  profile={axes:axes.map((c,i)=>[c,...ranges[i]]),trigger};calStage=3;
  notice('入力モニターで方向を確認し、チェックを入れてください。');
};
$('create').onclick=()=>{
  if(!$('seed').reportValidity())return;
  if(state?.lifecycle==='RUNNING'){notice('先に試行を終了、または一時停止してください。');return;}
  if(inputDevice==='gamepad'&&!padReady()){notice('ゲームパッドの校正・方向確認を完了してください。');return;}
  if(state&&!['FINISHED','ABORTED'].includes(state.lifecycle)&&!confirm('現在の試行を終了し、新しい試行を作成しますか？'))return;
  send('create',{config:{mode:$('mode').value,exercise:$('exercise').value,hs:Number($('hs').value),seed:Number($('seed').value),input_device:inputDevice,calibration:inputDevice==='gamepad'?profile:null}});
};
for(const [id,action] of Object.entries({start:'start',pause:'pause',resume:'resume',take:'take_control',land:'request_auto_land','confirm-land':'confirm_auto_land',cancel:'cancel',abort:'abort'}))$(id).onclick=()=>event(action);
$('download').onclick=async()=>{const d=await (await fetch('/api/summary')).json();const u=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download=`kmt-${sessionId}.json`;a.click();URL.revokeObjectURL(u);};
for(const key of Object.keys(control))$('axis-'+key).oninput=e=>{if(inputDevice==='keyboard')control[key]=Number(e.target.value);};
window.addEventListener('keydown',e=>{
 if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;
 if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight',' '].includes(e.key))e.preventDefault();
 if(e.key===' '){event('pause');return;}keys.add(e.key.toLowerCase());
});
window.addEventListener('keyup',e=>{keys.delete(e.key.toLowerCase());if(e.key.startsWith('Arrow')){control.pitch_deg=0;control.bank_deg=0;}if(['a','d'].includes(e.key.toLowerCase()))control.rudder=0;});
window.addEventListener('blur',()=>{keys.clear();event('pause');});
document.addEventListener('visibilitychange',()=>{if(document.hidden){keys.clear();event('pause');}});

function update(){
 const o=state.observation,run=state.lifecycle==='RUNNING';
 for(const id of ['mode','exercise','hs','seed'])$(id).disabled=!['FINISHED','ABORTED'].includes(state.lifecycle);
 $('authority').textContent=state.authority==='HUMAN'?'● YOU HAVE CONTROL':'● AUTOPILOT';$('authority').className=state.authority==='HUMAN'?'human':'';
 $('lifecycle').textContent=state.lifecycle;$('flight-title').textContent=state.phase==='TAKEOFF'?'離水 / Takeoff':state.phase==='APPROACH'?'進入 / Approach':'飛行 / Cruise';
 $('phase').textContent=state.phase;$('clock').textContent=`SIM ${state.sim_time_s.toFixed(2)} s`;
 for(const [id,value] of Object.entries({airspeed:o.airspeed_m_s,altitude:o.altitude_m,vspeed:o.vertical_speed_m_s,clearance:o.keel_clearance_m}))$(id).textContent=value.toFixed(1);
 $('heading').textContent=(Math.round((o.heading_deg+360)%360)%360)+'°';$('bank').textContent=`バンク ${o.bank_deg.toFixed(1)}°`;
 $('overlay').hidden=run;
 if(!run){$('overlay').replaceChildren();for(const [tag,text] of [['span',state.lifecycle],['strong',state.lifecycle==='READY'?'飛行開始を待っています':state.reason],['p',state.lifecycle==='PAUSED'?'状態は凍結されています。入力を合わせ、確認して再開してください。':'操作と引継ぎパネルで続けてください。']]){const e=document.createElement(tag);e.textContent=text;$('overlay').append(e);}}
 $('start').disabled=state.lifecycle!=='READY';$('pause').disabled=!run;$('resume').disabled=state.lifecycle!=='PAUSED';
 $('take').disabled=!run||state.authority!=='AUTO'||$('mode').value==='observe';
 $('land').disabled=!run;$('confirm-land').hidden=$('cancel').hidden=!state.pending_land;
 $('confirm-land').disabled=state.gate_missing.length>0;
 $('gate').textContent=state.pending_land?(state.gate_missing.length?'進入ゲート未達：'+state.gate_missing.join(' / '):'実験用進入ゲート内です。確認すると自動着水へ移ります。'):'';
 $('instruction').textContent=state.authority==='AUTO'?`自動制御中。引受けには安定飛行 ${Math.min(1,state.stable_s).toFixed(1)}/1.0秒、入力一致 ${Math.min(.5,state.matched_s).toFixed(1)}/0.5秒が必要です。`:'人間が操縦しています。推力・姿勢と波面クリアランスを確認してください。';
 if(o.airspeed_m_s<8&&o.altitude_m>3)$('instruction').textContent+=' ⚠ 速度余裕が小さい状態です。';
 const done=['FINISHED','ABORTED'].includes(state.lifecycle);$('review').hidden=!done;
 if(done){$('result').textContent=state.reason;$('review-metrics').textContent=`飛行時間 ${state.sim_time_s.toFixed(1)} s　／　標本最大荷重 ${state.metrics.peak_sampled_load_w.toFixed(2)} W　／　介入 ${state.metrics.interventions} 回　／　停止 ${state.metrics.paused_count} 回`;}
}

// Scene: model north/east/up -> Three.js east/up/north.
let renderer,scene,camera,plane,ocean,geometry,follow=true,lastWave=-1,frame=0;
try{
 renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 scene=new THREE.Scene();scene.background=new THREE.Color('#b2c6cd');scene.fog=new THREE.Fog('#b2c6cd',100,350);
 camera=new THREE.PerspectiveCamera(55,1,.1,700);scene.add(new THREE.HemisphereLight(0xe8f6ff,0x334954,2.6));
 const sun=new THREE.DirectionalLight(0xffeed7,3);sun.position.set(-30,80,30);scene.add(sun);
 const far=new THREE.Mesh(new THREE.PlaneGeometry(1500,1500),new THREE.MeshPhongMaterial({color:0x416d7a,shininess:70}));far.rotation.x=-Math.PI/2;far.position.y=-1;scene.add(far);
 geometry=new THREE.PlaneGeometry(96,96,48,48);geometry.rotateX(-Math.PI/2);
 ocean=new THREE.Mesh(geometry,new THREE.MeshPhongMaterial({color:0x477b88,shininess:90,side:THREE.DoubleSide}));scene.add(ocean);
 const grid=new THREE.GridHelper(800,80,0xb6d6cd,0x6f959b);grid.position.y=-.7;scene.add(grid);
 plane=new THREE.Group();const white=new THREE.MeshStandardMaterial({color:0xefeee6,roughness:.55});const orange=new THREE.MeshStandardMaterial({color:0xe8a56a});
 function box(w,h,l,x,y,z,mat=white){const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,l),mat);m.position.set(x,y,z);plane.add(m);return m;}
 box(.65,.75,5,0,0,0);box(15,.16,1.5,0,.45,.2);box(4,.12,.7,0,.5,-2);box(.1,1,.8,0,1,-2);box(.3,.35,1.3,-5.5,-.2,.2);box(.3,.35,1.3,5.5,-.2,.2);
 box(.4,.35,.6,-2,.8,1,orange);box(.4,.35,.6,2,.8,1,orange);box(.4,.25,.8,0,.5,1,orange);scene.add(plane);
 const markerMat=new THREE.MeshBasicMaterial({color:0xefc689,wireframe:true});
 for(let z=100;z<=800;z+=100){const ring=new THREE.Mesh(new THREE.TorusGeometry(7,.15,4,32),markerMat);ring.position.set(0,25,z);scene.add(ring);}
 const resize=()=>{const {width,height}=$('viewport').getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();};new ResizeObserver(resize).observe($('viewport'));resize();
}catch(e){notice('WebGLが利用できません。計器のみ利用可能です：'+e.message);}
$('camera').onclick=()=>{follow=!follow;};
function draw(t){
 requestAnimationFrame(draw);const dt=Math.min(.05,(t-frame)/1000||.016);frame=t;
 const pads=[...(navigator.getGamepads?.()||[])].filter(Boolean);if(padIndex===null&&pads.length)padIndex=pads[0].index;
 const p=pad();
 if(inputDevice==='gamepad'){
   $('pad-status').textContent=p?`${p.id.slice(0,45)} · ${p.mapping||'非標準'}`:'未接続：USBで接続し、ボタンを押してください';
   if(p){
     $('raw-pad').textContent=`axes ${p.axes.slice(0,3).map(v=>v.toFixed(2)).join(' / ')} · RT ${(p.buttons[7]?.value??0).toFixed(2)}`;
     if(calStage===1){for(let i=0;i<3;i++){ranges[i][0]=Math.min(ranges[i][0],p.axes[i]);ranges[i][1]=Math.max(ranges[i][1],p.axes[i]);}trigger[0]=Math.min(trigger[0],p.buttons[7].value);trigger[1]=Math.max(trigger[1],p.buttons[7].value);}
     if(profile){control=mapPad(p,profile);}
     const pressed=p.buttons.map(b=>b.pressed);
     if(padReady())for(const [i,a] of [[0,state?.pending_land?'confirm_auto_land':state?.lifecycle==='READY'?'start':'take_control'],[1,'cancel'],[5,'request_auto_land'],[9,'pause']])if(pressed[i]&&!lastButtons[i])event(a);
     lastButtons=pressed;
   }
 }else{
   $('pad-status').textContent='デモ操作：矢印キー / W・S / A・D';
   if(keys.has('w'))control.throttle=clamp(control.throttle+dt*.25,0,1);if(keys.has('s'))control.throttle=clamp(control.throttle-dt*.25,0,1);
   if(keys.has('arrowup'))control.pitch_deg=-5;if(keys.has('arrowdown'))control.pitch_deg=8;
   if(keys.has('arrowleft'))control.bank_deg=-15;if(keys.has('arrowright'))control.bank_deg=15;
   if(keys.has('a'))control.rudder=-.4;if(keys.has('d'))control.rudder=.4;
 }
 for(const k of Object.keys(control)){if(document.activeElement!==$('axis-'+k))$('axis-'+k).value=control[k];$('axis-'+k).disabled=inputDevice==='gamepad';$('value-'+k).textContent=`${control[k].toFixed(2)} / ${(state?.applied_control[k]??0).toFixed(2)}`;}
 if(!renderer)return;
 const s=state;let pos=s?.position??[0,0,2],att=s?.attitude??[0,0,0],sim=s?.sim_time_s??0;
 if(s&&previous&&s.lifecycle==='RUNNING'&&previous.session_id===s.session_id){
   const f=clamp((performance.now()-received)/50,0,1);pos=pos.map((v,i)=>THREE.MathUtils.lerp(previous.position[i],v,f));sim=THREE.MathUtils.lerp(previous.sim_time_s,s.sim_time_s,f);
   att=att.map((v,i)=>previous.attitude[i]+Math.atan2(Math.sin(v-previous.attitude[i]),Math.cos(v-previous.attitude[i]))*f);
 }
 const [pitch,bank,heading]=att;
 // Forward NEU -> ENU render coordinates; local plane nose is +Z.
 const forward=new THREE.Vector3(Math.sin(heading)*Math.cos(pitch),Math.sin(pitch),Math.cos(heading)*Math.cos(pitch));
 const right=new THREE.Vector3(Math.cos(heading),0,-Math.sin(heading));const up=new THREE.Vector3().crossVectors(forward,right).normalize();
 const rolledRight=right.clone().multiplyScalar(Math.cos(bank)).addScaledVector(up,-Math.sin(bank));const rolledUp=up.clone().multiplyScalar(Math.cos(bank)).addScaledVector(right,Math.sin(bank));
 plane.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(rolledRight,rolledUp,forward));plane.position.set(pos[1],pos[2],pos[0]);plane.visible=follow;
 if(follow){camera.position.copy(plane.position).addScaledVector(forward,-22).add(new THREE.Vector3(0,9,0));camera.up.set(0,1,0);camera.lookAt(plane.position.clone().addScaledVector(forward,10));}else{camera.position.copy(plane.position).addScaledVector(forward,2).addScaledVector(rolledUp,.8);camera.up.copy(rolledUp);camera.lookAt(camera.position.clone().add(forward));}
 // Recompute the exact model spectrum at mesh vertices. Mesh interpolation is illustrative.
 if(sim!==lastWave||lastWave<0){
   ocean.position.set(Math.round(pos[1]/2)*2,0,Math.round(pos[0]/2)*2);
   const a=geometry.attributes.position;
   for(let i=0;i<a.count;i++)a.setY(i,waveHeight(terms,a.getZ(i)+ocean.position.z,a.getX(i)+ocean.position.x,sim));
   a.needsUpdate=true;geometry.computeVertexNormals();lastWave=sim;
 }
 renderer.render(scene,camera);
}
requestAnimationFrame(draw);
