// 第 8 章：同一数组在逻辑坐标、偏移、通道与存储体之间的映射。
export default {
  '08-view'(f) {
    const a=f.grid(30,35,{rows:2,cols:3,cw:48,ch:36,label:(i,j)=>'abcdef'[3*i+j],fill:(i)=>i?'green':'orange'});
    const t=f.grid(310,35,{rows:3,cols:2,cw:48,ch:36,label:(i,j)=>'abcdef'[3*j+i],fill:(i,j)=>j?'green':'orange'});
    f.text(102,15,'原数组 (2,3)'); f.text(358,15,'转置视图 (3,2)');
    f.arrow([195,75],[282,75]); f.note(238,56,'交换步长');
    const m=f.grid(80,210,{rows:1,cols:6,cw:48,ch:36,label:(i,j)=>'abcdef'[j],fill:(i,j)=>j<3?'orange':'green'});
    f.text(224,266,'存储始终为 a b c d e f');
    f.line([t.cell(0,0).L(),[282,53],[282,170],[104,170],m.cell(0,0).T()],{arrow:'end',color:'blue'});
    f.line([t.cell(0,1).R(),[440,53],[440,185],[248,185],m.cell(0,3).T()],{arrow:'end',color:'blue'});
    f.note(32,310,'转置后的第 0 行：偏移 0 和 3',{anchor:'start'});
  },
  '08-tile'(f) {
    for(let k=0;k<2;k++) {
      const x=30+k*270;
      f.text(x+88,20,k?'T(2,2) 分块':'行优先');
      f.grid(x,50,{rows:4,cols:4,cw:44,ch:38,label:(i,j)=>k?(Math.floor(i/2)*2+Math.floor(j/2))*4+(i%2)*2+j%2:4*i+j,fill:(i,j)=>['orange','blue','green','purple'][Math.floor(i/2)*2+Math.floor(j/2)]});
    }
    f.note(30,228,'每格的数字是偏移；颜色是逻辑块',{anchor:'start'});
  },
  '08-pack'(f) {
    f.text(215,18,'bf16：上下两行共用四个 32 位位置');
    const g=f.grid(40,50,{rows:2,cols:4,cw:75,ch:32,label:(i,j)=>`$x_{${i}${j}}$`,fill:i=>i?'green':'orange'});
    f.brace(40,135,340,'四个 32 位位置');
    const a=f.box(40,205,300,38,'f32 的 R₀：x₀₀  x₀₁  x₀₂  x₀₃',{color:'orange'});
    const b=f.box(40,278,300,38,'f32 的 R₁：x₁₀  x₁₁  x₁₂  x₁₃',{color:'green'});
    f.line([g.cell(0,0).L(),[10,66],[10,224],a.L()],{arrow:'end',color:'orange'});
    f.line([g.cell(1,3).R(),[370,98],[370,297],b.R()],{arrow:'end',color:'green'});
  },
  '08-swizzle'(f) {
    for(let k=0;k<2;k++) {
      const x=30+k*260; f.text(x+80,18,k?'XOR：bank = i XOR j':'行优先：bank = j');
      const g=f.grid(x,50,{rows:4,cols:4,cw:40,ch:38,label:(i,j)=>k?i^j:j,fill:(i,j)=>j===1?'orange':null});
      f.frame(g.region(0,1,3,1),{color:'red'});
      f.text(x+80,235,k?'读列：1, 0, 3, 2':'读列：1, 1, 1, 1');
    }
  },
  '08-segments'(f) {
    const labels=['起点 0：4 段','起点 4：5 段','跨步：32 段（画前 8 段）'];
    for(let k=0;k<3;k++) {
      const y=55+k*95; f.text(20,y-20,labels[k],{anchor:'start'});
      f.grid(20,y,{rows:1,cols:k===2?8:5,cw:64,ch:30,label:(i,j)=>`${j}`,fill:(i,j)=>k===0&&j===4?null:'gray'});
      const n=k===0?4:k===1?5:8;
      for(let j=0;j<n;j++) {
        const x=20+j*64;
        if(k===2) f.rect(x+1,y+2,7,26,{fill:'orange'});
        else f.rect(x+(k===1&&j===0?8:1),y+2,k===1&&j===4?7:k===1&&j===0?55:62,26,{fill:'orange'});
      }
    }
    f.note(20,330,'格是 32 字节段；段内有用字节以橙色表示',{anchor:'start'});
  },
  "08-layout-compose": function(f) {
    const labels=['逻辑下标 j','布局 L：旧位置','目标布局：新位置'];
    const maps=[[0,1,2,3],[0,2,1,3],[3,2,1,0]];
    for(let k=0;k<3;k++) {
      f.text(70,71+k*130,labels[k],{anchor:'end',size:12});
      f.grid(80,50+k*130,{rows:1,cols:4,cw:90,ch:42,label:(_,j)=>maps[k][j],fill:(_,j)=>['blue','orange','green','purple'][j]});
      if(k<2) for(let j=0;j<4;j++) f.arrow([125+j*90,94+k*130],[125+j*90,174+k*130]);
    }
    f.note(260,400,'同色对应同一逻辑元素；转换是目标布局与旧布局逆映射的复合');
  },
};
