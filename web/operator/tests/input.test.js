import test from 'node:test';
import assert from 'node:assert/strict';
import {axis,mapPad,waveHeight} from '../src/input.js';
test('deadzone and clamping',()=>{assert.equal(axis(.03),0);assert.equal(axis(2),1);assert.equal(axis(-2),-1);assert.equal(axis(0,.1,-1,1,.1),0);});
test('stick direction and throttle',()=>{
 const pad={axes:[1,-1,-1],buttons:Array.from({length:8},()=>({value:1}))};
 const profile={axes:[[0,-1,1],[0,-1,1],[0,-1,1]],trigger:[0,1]};
 assert.deepEqual(mapPad(pad,profile),{throttle:1,pitch_deg:12,bank_deg:25,rudder:-1});
 pad.axes[1]=1;assert.equal(mapPad(pad,profile).pitch_deg,-8);
});
test('shared wave convention',()=>{assert.equal(waveHeight([[0,0,0,2,0]],1,2,3),2);assert.ok(Math.abs(waveHeight([[1,0,1,1,0]],1,4,1)-1)<1e-10);});
