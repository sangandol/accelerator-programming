export default {
  '18-attention-io'(f) {
    const q=f.box(25,45,150,65,'驻留 Q + 状态',{color:'blue'});
    const bs=[0,1,2,3].map(i=>f.box(260,25+i*85,155,48,`键值块 ${i}`,{color:'green'}));
    bs.forEach(b=>f.link(b,q,{arrow:'end'}));
    const tmp=f.box(500,120,170,65,'片上临时 S / P',{color:'orange'});f.link(q,tmp,{arrow:'end'});
    f.note(25,380,'换一个查询块后，可能再次读取全部 K/V',{anchor:'start'});
  },
  '18-state'(f) {
    const st=f.box(30,100,160,60,'旧状态 (m,ℓ,u)',{color:'blue'}),kv=f.box(280,20,160,45,'当前 S 与 V',{color:'green'}),up=f.box(280,105,160,50,'参考点统一、合并',{color:'orange'}),out=f.box(520,105,160,50,'新状态 (m′,ℓ′,u′)',{color:'purple'});
    f.link(st,up,{arrow:'end'});f.link(kv,up,{arrow:'end'});f.link(up,out,{arrow:'end'});
    f.line([out.B(),[600,230],[110,230],st.B()],{arrow:'end'});f.note(355,270,'键值循环只传递小状态');
  },
  '18-causal-blocks'(f) {
    f.grid(70,70,{rows:4,cols:4,cw:72,ch:62,rowLabels:[0,1,2,3],colLabels:[0,1,2,3],label:(i,j)=>i===j?'部分':j<i?'全部':'跳过',fill:(i,j)=>i===j?'orange':j<i?'green':null});
    f.note(70,370,'行：查询块；列：键块。10 个非空块中只有 4 个需元素掩码',{anchor:'start'});
  },
  '18-split-kv'(f) {
    const q=f.box(230,20,170,40,'同一个查询 Q',{color:'blue'});
    const b=[0,1,2,3].map(i=>f.box(20+i*170,130,140,50,`段 ${i} → (m,ℓ,u)`,{color:'green',size:12}));b.forEach(v=>f.link(q,v,{arrow:'end'}));
    const out=f.box(230,285,170,50,'合并状态 → u/ℓ',{color:'orange'});b.forEach(v=>f.link(v,out,{arrow:'end'}));
  },
};
