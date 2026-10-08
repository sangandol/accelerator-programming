export default {
  '26-query-kv'(f) {
    const q=f.box(25,110,155,65,'Q + 行状态驻留',{color:'blue'});
    const kv=f.box(300,20,190,55,'K/V 第 j 块',{color:'green'}),s=f.box(300,120,190,55,'临时分数 S / 概率 P',{color:'orange'}),u=f.box(300,230,190,55,'更新 m、ℓ、u',{color:'purple'});
    f.link(kv,s,{arrow:'end'});f.link(q,s,{arrow:'end'});f.link(s,u,{arrow:'end'});
    f.line([u.R(),[565,257],[565,47],kv.R()],{arrow:'end'});f.note(25,350,'临时分数块不离开片上存储；下一个 j 继续更新同一状态',{anchor:'start'});
  },
  '26-warp-partition'(f) {
    for(let k=0;k<2;k++) {
      const x=25+k*335;f.text(x+128,20,k?'沿键列分：行状态需合并':'沿查询行分：各持完整行');
      f.grid(x,65,{rows:8,cols:8,cw:32,ch:30,fill:(i,j)=>['blue','orange','green','purple'][Math.floor((k?j:i)/2)]});
    }
    f.note(25,370,'颜色代表四个 warp 的数据归属，不代表四份独立注意力',{anchor:'start'});
  },
  '26-pingpong'(f) {
    const bars=[];for(let i=0;i<3;i++){bars.push([0,4*i,4*i+2,`A${i}`,'blue'],[1,4*i+2,4*i+4,`B${i}`,'green'],[2,4*i+2,4*i+4,`A${i}`,'blue'],[3,4*i+4,4*i+6,`B${i}`,'green']);}
    f.timeline(120,30,{lanes:['A：矩阵','B：矩阵','A：softmax','B：softmax'],bars,unit:38,ticks:[0,2,4,6,8,10,12,14]});
    f.note(120,325,'不同查询组可重叠；同组的状态更新顺序不变',{anchor:'start'});
  },
  '26-split-kv'(f) {
    f.grid(90,65,{rows:8,cols:16,cw:27,ch:25,rowLabels:Array.from({length:8},(_,i)=>`KV 头 ${i}`),fill:()=> 'blue'});
    f.brace(90,320,522,'每头 16 段：共 128 个独立任务');
    f.box(135,390,340,45,'每头合并 16 份 (m,ℓ,u)，输出仍按头排列',{color:'orange',size:13});
  },
};
