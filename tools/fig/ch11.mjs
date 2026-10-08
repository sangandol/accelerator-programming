export default {
  '11-layer'(f) {
    const labels=['X [N,D]','RMSNorm','注意力','加残差','RMSNorm','门控 MLP','加残差'];
    const b=labels.map((s,i)=>f.box(180,25+i*75,160,42,s,{color:i===2||i===5?'blue':'gray'}));
    for(let i=0;i<6;i++)f.link(b[i],b[i+1],{arrow:'end'});
    f.line([b[0].L(),[65,b[0].cy],[65,b[3].cy],b[3].L()],{arrow:'end',color:'orange'});
    f.line([b[3].R(),[455,b[3].cy],[455,b[6].cy],b[6].R()],{arrow:'end',color:'orange'});
    f.note(380,b[2].cy,'Q、K、V → softmax → O',{anchor:'start'});
    f.note(20,b[5].cy,'两次上投影 [N,F]',{anchor:'start'});
    f.note(20,b[5].cy+25,'门控后下投影 [N,D]',{anchor:'start'});
  },
  '11-prefill-decode'(f) {
    for(let i=0;i<2;i++) {
      const x=25+i*300;
      f.text(x+115,20,i?'decode：少量新词元':'prefill：许多词元');
      const a=f.grid(x,65,{rows:i?1:6,cols:3,cw:28,ch:25,fill:()=> 'blue'});
      const w=f.box(x+150,65,80,150,'权重 W',{color:'orange'});f.link(a,w,{arrow:'end'});
      f.note(x,255,i?'同样的权重，摊到很少行':'同样的权重，摊到很多行',{anchor:'start'});
    }
  },
  '11-kv'(f) {
    const k=f.box(25,50,210,56,'K：T × Hkv × dh',{color:'blue'}),v=f.box(285,50,210,56,'V：T × Hkv × dh',{color:'green'});
    f.brace(25,145,495,'一层、一序列：两份键值');
    const a=f.box(120,205,280,48,'再乘层数 nL、批量 B、字节 skv',{color:'orange'});
    f.line([k.B(),[130,185],a.T(0.2)],{arrow:'end'});f.line([v.B(),[390,185],a.T(0.8)],{arrow:'end'});
    f.note(25,292,'查询头共享键值时，不额外复制 KV',{anchor:'start'});
  },
  '11-moe'(f) {
    const xs=[f.box(20,40,130,70,'6 个词元',{color:'blue'}),f.box(230,40,170,70,'各选 2 专家：12 份',{color:'orange'}),f.box(480,40,130,70,'按专家计算',{color:'green'})];
    f.link(xs[0],xs[1],{arrow:'end'});f.link(xs[1],xs[2],{arrow:'end'});
    const out=f.box(480,195,130,55,'还原并加权',{color:'purple'});f.link(xs[2],out,{arrow:'end'});
    f.note(20,195,'每份保留原词元编号和路由权重',{anchor:'start'});
    f.note(20,220,'最终仍输出 6 个词元的隐藏向量',{anchor:'start'});
  },
  "11-kv-budget": function(f) {
    f.text(330,18,'32 层、头维度 128、bf16：KV 的容量乘法');
    f.bars(230,65,{items:[['B=1, T=8192, KV头=8',1,'blue'],['B=4, T=8192, KV头=8',4,'orange'],['B=1, T=32768, KV头=8',4,'green'],['B=1, T=8192, KV头=32',4,'purple']],w:330,fmt:v=>`${v} GiB`});
    f.note(330,245,'基准为 1 GiB；任一因子扩大四倍，载荷也扩大四倍');
  },
};
