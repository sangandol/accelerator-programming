export default {
  '13-trace'(f) {
    const a=f.box(20,35,170,55,'Python f + f32[4]',{color:'blue'}),b=f.box(280,35,170,55,'jaxpr：乘、加',{color:'green'});f.link(a,b,{arrow:'end'});
    const c=f.box(280,165,170,55,'设备执行同一程序',{color:'orange'});f.link(b,c,{arrow:'end'});
    const x=f.box(20,165,170,55,'实际值：1,2,3,4',{color:'purple'});f.link(x,c,{arrow:'end'});
    f.note(20,280,'类型用于构造与缓存；数值用于执行',{anchor:'start'});
  },
  '13-compiler'(f) {
    const labels=['jaxpr：原语与类型','StableHLO：张量运算','后端：布局、分块、调度','机器程序：资源操作'];
    const bs=labels.map((s,i)=>f.box(50,25+i*90,285,48,s,{color:['blue','green','orange','purple'][i]}));
    for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(380,bs[0].cy,'决定算什么',{anchor:'start'});f.note(380,bs[2].cy,'决定怎样送到硬件',{anchor:'start'});
  },
  '13-ad'(f) {
    for(let k=0;k<2;k++) {
      const y=40+k*150;const a=f.box(25,y,125,48,k?'输出余切 u':'输入切向量 v',{color:'orange'});
      const b=f.box(235,y,160,48,k?'转置局部线性规则':'局部线性规则',{color:'blue'});
      const c=f.box(470,y,125,48,k?'输入余切 Jᵀu':'输出切向量 Jv',{color:'green'});
      f.link(a,b,{arrow:'end'});f.link(b,c,{arrow:'end'});f.note(25,y+90,k?'VJP：反向应用转置规则':'JVP：前向传播变化量',{anchor:'start'});
    }
  },
  '13-scan'(f) {
    const bs=[0,1,2].map(i=>f.box(80+i*195,110,130,50,`循环体 f：步 ${i}`,{color:'blue'}));
    bs.forEach((b,i)=>{const a=f.box(100+i*195,25,90,35,`x${i}`,{color:'orange'});f.link(a,b,{arrow:'end'});const c=f.box(100+i*195,235,90,35,`y${i}`,{color:'green'});f.link(b,c,{arrow:'end'});});
    f.link(bs[0],bs[1],{arrow:'end'});f.link(bs[1],bs[2],{arrow:'end'});
    f.note(242,95,'c₁');f.note(437,95,'c₂');f.note(80,320,'携带状态类型固定；各 yᵢ 按已知长度堆叠',{anchor:'start'});
  },
};
