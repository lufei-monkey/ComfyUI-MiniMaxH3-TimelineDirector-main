import { app } from '/scripts/app.js';
app.registerExtension({
  name: 'MiniMaxH3.ExternalAggregators',
  beforeRegisterNodeDef(type, data) {
    if (!['MiniMaxH3ExternalMedia','MiniMaxH3ExternalSegments'].includes(data.name)) return;
    const definitions = data.input.optional;
    const isSegments = data.name === 'MiniMaxH3ExternalSegments';
    function refresh() {
      if(this.__h3Refreshing) return;
      this.__h3Refreshing = true;
      try {
        const linked = new Set((this.inputs || []).filter(x=>x.link!=null).map(x=>x.name));
        const wanted = new Set();
        function group(prefix, first, last) {
          let highest = first-1;
          for(let i=first;i<=last;i++) if(linked.has(`${prefix}${i}`)) highest=i;
          for(let i=first;i<=Math.min(last,highest+1);i++) wanted.add(`${prefix}${i}`);
        }
        if(isSegments) {
          let highest=0;
          for(const name of linked) {const match=name.match(/^segment_(\d+)_/);if(match) highest=Math.max(highest,Number(match[1]));}
          for(let segment=1;segment<=Math.min(64,highest+1);segment++) {
            group(`segment_${segment}_image_`,1,9);
            group(`segment_${segment}_audio_`,1,3);
            for(const field of ['segment_frames','segment_prompt','overlap_frames','duration']) wanted.add(`segment_${segment}_${field}`);
          }
        } else {
          for(const [prefix,max] of [['image_',8],['ref_video_',2],['ref_audio_',2],['ref_video_audio_',2]]) group(prefix,0,max);
          wanted.add('global_prompt');
        }
        for(let i=(this.inputs?.length||0)-1;i>=0;i--) {
          const input=this.inputs[i];
          if(definitions[input.name] && !wanted.has(input.name) && input.link==null) this.removeInput(i);
        }
        for(const name of wanted) if(!this.inputs?.some(x=>x.name===name)) this.addInput(name,definitions[name][0]);
      } finally {this.__h3Refreshing=false;}
      this.setDirtyCanvas?.(true,true);
    }
    for(const event of ['onNodeCreated','onConfigure','onConnectionsChange']) {
      const original=type.prototype[event];
      type.prototype[event]=function(...args){const result=original?.apply(this,args);queueMicrotask(()=>refresh.call(this));return result;};
    }
  }
});
