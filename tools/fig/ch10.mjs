export default {
  '10-state-tree'(f) {
    const a=[0,1,2,3].map(i=>f.box(20+i*140,20,110,42,`块 ${i} → 状态`,{color:'blue'}));
    const b=[f.box(90,110,110,42,'合并 0、1',{color:'green'}),f.box(370,110,110,42,'合并 2、3',{color:'green'})];
    a.forEach((v,i)=>f.link(v,b[Math.floor(i/2)],{arrow:'end'}));
    const c=f.box(230,205,110,42,'最终状态',{color:'orange'});b.forEach(v=>f.link(v,c,{arrow:'end'}));
    const d=f.box(230,295,110,42,'收尾 → 输出',{color:'purple'});f.link(c,d,{arrow:'end'});
  },
  '10-softmax-state'(f) {
    const a=f.box(20,30,205,55,'m=log 2，ℓ=3/2，u=25',{color:'blue'});
    const b=f.box(305,30,205,55,'m=log 4，ℓ=1，u=30',{color:'green'});
    const aa=f.box(20,150,205,55,'m=log 4，ℓ=3/4，u=25/2',{color:'blue'});
    const bb=f.box(305,150,205,55,'参考点不变',{color:'green'});
    f.link(a,aa,{arrow:'end'});f.link(b,bb,{arrow:'end'});f.note(240,120,'ℓ 和 u 都乘 1/2');
    const c=f.box(155,270,220,52,'ℓ=7/4，u=85/2 → o=170/7',{color:'orange',size:13});
    f.link(aa,c,{arrow:'end'});f.link(bb,c,{arrow:'end'});
  },
  '10-block-scan'(f) {
    const vals=[['1, 2','3, 4','5, 6'],['1, 3','3, 7','5, 11'],['0','3','10'],['1, 3','6, 10','15, 21']];
    f.grid(130,50,{rows:4,cols:3,cw:120,ch:55,rowLabels:['输入块','块内前缀','排他偏移','全局前缀'],label:(i,j)=>vals[i][j],fill:i=>['gray','blue','orange','green'][i]});
    f.note(130,305,'块总量：3、7、11；加回前面块的总量',{anchor:'start'});
  },
  "10-rescale": function(f) {
    const a=f.box(30,35,230,50,'旧状态：m=2, ℓ=1, u=10',{color:'blue',size:13});
    const b=f.box(355,35,230,50,'新块：m=4, ℓ=1, u=30',{color:'orange',size:13});
    const c=f.box(125,170,360,60,'参考点 4：旧 ℓ、u 同乘 exp(−2)',{color:'green',size:13});
    f.link(a,c,{arrow:'end'});f.link(b,c,{arrow:'end'});
    const d=f.box(125,305,360,55,'ℓ≈1.1353，u≈31.3534，输出≈27.616',{color:'purple',size:13});f.link(c,d,{arrow:'end'});
    f.note(305,403,'必须重缩放分子与分母；不能只修改最大值');
  },
};
