export default {
  '17-tile-budget'(f) {
    f.box(25,30,170,80,'A：512×256 bf16',{color:'blue'});f.box(260,30,170,80,'B：256×512 bf16',{color:'green'});
    f.box(142,165,170,80,'累加：512×512 f32',{color:'orange'});
    f.text(227,300,'输入双缓冲 1 MiB + 累加 1 MiB + 输出双缓冲 1 MiB');
    f.note(25,345,'只是显式预算；为后端临时值与对齐另留余量',{anchor:'start'});
  },
  '17-epilogue'(f) {
    const labels=['K 部分积之和','加一次偏置','激活','舍入 / 写回'];
    const bs=labels.map((s,i)=>f.box(20+i*155,50,120,50,s,{color:i?'green':'blue'}));for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(20,160,'非线性在完整和上执行，不能对每个部分积分别执行',{anchor:'start'});
  },
  '17-group-scale'(f) {
    const a=f.box(25,35,155,50,'部分积 3 × 因子 2',{color:'blue'}),b=f.box(325,35,155,50,'部分积 5 × 因子 10',{color:'green'});
    const c=f.box(175,170,155,50,'累加：6+50=56',{color:'orange'});f.link(a,c,{arrow:'end'});f.link(b,c,{arrow:'end'});
    f.note(25,280,'不能先加 3+5 再乘一个共用因子',{anchor:'start'});
  },
  '17-groups'(f) {
    f.grid(100,55,{rows:3,cols:3,cw:125,ch:50,rowLabels:['专家 0','专家 1','专家 2'],colLabels:['真实行','补齐后行','额外行'],label:(i,j)=>[[5,128,123],[130,256,126],[0,0,0]][i][j],fill:(i,j)=>j===2?'gray':'orange'});
    f.text(285,260,'135 行 → 384 行，约 2.84 倍');
  },
};
