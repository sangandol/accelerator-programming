export default {
  '29-safetensors'(f) {
    const segs=[['h（8 字节）',90,'gray'],['JSON 文件头（h 字节）',190,'orange'],['张量 A 的字节',150,'blue'],['张量 B 的字节',150,'blue'],['…',60,'blue']];
    let x=20;const b=segs.map(([s,w,c])=>{const r=f.box(x,60,w,46,s,{color:c});x+=w;return r;});
    f.brace(b[2].x,140,x,'数据区：偏移从这里算起');
    f.note(b[1].cx,25,'每个张量：dtype、shape、data_offsets');
    f.note(b[2].x,205,'每个张量按行优先存放，一个接一个',{anchor:'start'});
  },
  '29-pipeline'(f) {
    const top=['对话','模板与词元化','词元编号','嵌入：按编号取行','各层：写入 KV cache'];
    const w=[70,120,90,140,150];let x=20;
    const a=top.map((s,i)=>{const r=f.box(x,40,w[i],44,s,{color:i<2?'gray':'blue'});x+=w[i]+28;return r;});
    for(let i=0;i<4;i++)f.link(a[i],a[i+1],{arrow:'end'});
    const bot=['文字','流式解码','新词元','抽样','输出层：logits'];
    x=20;const c=bot.map((s,i)=>{const r=f.box(x,170,w[i],44,s,{color:i<2?'gray':i===3?'orange':'blue'});x+=w[i]+28;return r;});
    for(let i=4;i>0;i--)f.link(c[i],c[i-1],{arrow:'end'});
    f.link(a[4],c[4],{arrow:'end'});
    f.note(a[4].cx+8,127,'只取要抽样的位置',{anchor:'start'});
    f.link(c[2],a[2],{arrow:'end',dash:'5 4'});
    f.note(a[2].cx+8,127,'decode 只送入这一个词元',{anchor:'start'});
  },
};
