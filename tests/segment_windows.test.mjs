import assert from 'node:assert/strict';
import {createWindows,editWindow,alignedLength} from '../js/segment_windows.mjs';
function valid(w){
  assert.equal(w[0].startFrame,0);
  w.forEach((s,i)=>{
    const length=s.endFrame-s.startFrame;
    assert(length>=5&&length<=3609&&(length-5)%17===0);
    if(i){const p=w[i-1],o=p.endFrame-s.startFrame;
      assert(s.startFrame>p.startFrame&&s.endFrame>p.endFrame);
      assert(o>=0&&o<Math.min(length,p.endFrame-p.startFrame));
      assert(o===0||o===1||(o>=5&&(o-5)%17===0));
    }
  });
}
let w=createWindows([],4,243); valid(w);
assert.deepEqual(w.map(s=>s.startFrame),[0,204,408,612]);
w=editWindow(w,1,'move',w[0].endFrame);valid(w);
assert.equal(w[1].startFrame,w[0].endFrame);
const durations=w.map(s=>s.endFrame-s.startFrame);
w=editWindow(w,1,'right',w[1].endFrame+100);valid(w);
assert.equal(w[2].endFrame-w[2].startFrame,durations[2]);
let seed=123;
for(let i=0;i<3000;i++){
  seed=(Math.imul(seed,1664525)+1013904223)>>>0;
  const index=seed%4, mode=['move','left','right'][(seed>>>3)%3];
  w=editWindow(w,index,mode,(seed%20000)-1000);valid(w);
}
assert.equal(createWindows(w,0,243).length,0);
assert.equal(alignedLength(360),362);
console.log('segment window geometry: PASS');
