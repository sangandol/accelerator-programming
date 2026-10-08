export default {
  '25-level-roofline'(f) {
    const labels=['HBM','L2','共享内存','寄存器 / 矩阵单元'];
    const bs=labels.map((s,i)=>f.box(25+i*175,65,140,60,s,{color:['gray','purple','blue','orange'][i]}));for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(180,185,'块间复用');f.note(360,185,'块内复用');f.note(545,185,'片段复用');
    f.text(360,280,'T ≥ max(F/P, Q_H/B_H, Q_L2/B_L2, Q_S/B_S)');
  },
  '25-block-scale'(f) {
    f.grid(25,60,{rows:1,cols:8,cw:45,ch:45,label:(i,j)=>`${j}`,fill:()=> 'blue'});f.brace(25,155,385,'缩画 32 个 FP4：共 16 字节');
    f.box(455,60,145,45,'缩放：另 1 字节',{color:'orange'});
    f.box(105,245,410,50,'平均 17/32 = 0.53125 字节 / 元素',{color:'green'});
    f.note(25,345,'量化 payload 与缩放元数据分别计账',{anchor:'start'});
  },
  '25-group-schedule'(f) {
    f.grid(25,55,{rows:1,cols:6,cw:84,ch:60,label:(i,j)=>`u=${j}`,fill:(i,j)=>j<2?'blue':'green'});
    f.brace(25,155,193,'专家 0');f.brace(193,155,529,'专家 1');
    f.text(280,245,'边界前缀：0, 2, 6, 6');f.note(25,310,'专家 2 区间为空；u=4 → 专家 1 内第 2 块',{anchor:'start'});
  },
  '25-softmax-mask'(f) {
    const rows=[['−5','−3','−4','−∞'],['−2','0','−1','−∞'],['exp(−2)','1','exp(−1)','0'],['输出','输出','输出','不写']];
    f.grid(120,65,{rows:4,cols:4,cw:115,ch:58,rowLabels:['载入 / 掩码','减最大值','求指数','写回'],label:(i,j)=>rows[i][j],fill:(i,j)=>j===3?'gray':'blue'});
    f.note(120,360,'填充值须分别适合最大值、指数和写回的语义',{anchor:'start'});
  },
  "25-budget-spaces": function(f) {
    const a=f.box(25,55,220,70,'共享内存：多份 A、B 块',{color:'blue',size:13}),b=f.box(355,55,260,70,'寄存器 / 矩阵存储：累加值',{color:'orange',size:13});
    f.box(125,240,390,65,'分别检查容量，再看复用与驻留块数',{color:'green'});
    f.note(320,390,'更大的输出块省输入流量，却增加累加状态；不能合并两种存储预算');
  },
};
