export const clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
export function axis(value,center=0,lo=-1,hi=1,dead=.05){
  const span=value>=center?hi-center:center-lo;
  if(span<=.1) return 0;
  const n=clamp((value-center)/span,-1,1);
  return Math.abs(n)<=dead?0:Math.sign(n)*(Math.abs(n)-dead)/(1-dead);
}
export function mapPad(pad,profile){
  const p=axis(pad.axes[1]??0,...profile.axes[1]);
  const b=axis(pad.axes[0]??0,...profile.axes[0]);
  const r=axis(pad.axes[2]??0,...profile.axes[2]);
  const t=clamp(((pad.buttons[7]?.value??0)-profile.trigger[0])/(profile.trigger[1]-profile.trigger[0]),0,1);
  return {throttle:t,pitch_deg:p<0?-p*12:-p*8,bank_deg:b*25,rudder:r};
}
export function waveHeight(terms,north,east,t){
  let z=0; for(const [kx,ky,w,a,p] of terms) z+=a*Math.cos(kx*north+ky*east-w*t+p); return z;
}
