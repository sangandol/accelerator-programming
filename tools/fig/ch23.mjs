export default {
  '23-stage-protocol'(f) {
    const p=f.box(25,105,150,60,'生产者：TMA',{color:'blue'}),s=f.box(300,105,170,60,'槽 s：数据 + 相位',{color:'orange'}),c=f.box(595,105,150,60,'消费者：MMA',{color:'green'});
    f.link(p,s,{arrow:'end'});f.link(s,c,{arrow:'end'});
    f.line([c.B(),[670,280],[100,280],p.B()],{arrow:'end',color:'purple'});f.note(380,320,'MMA 读完 → empty → 允许下一轮写');
    f.note(380,45,'写入完成 → full → 允许本轮读');
  },
  '23-wgmma-groups'(f) {
    f.timeline(105,30,{lanes:['组 0','组 1','读取最终累加'],bars:[[0,0,4,'在途','blue'],[1,2,7,'在途','orange'],[2,7,9,'读取','green']],unit:55,ticks:[0,2,4,7,9]});
    f.note(105,255,'t=4 可保留一组未完成；t=7 才能读取最终累加结果',{anchor:'start'});
  },
  '23-tmem-double'(f) {
    f.grid(145,65,{rows:2,cols:3,cw:150,ch:70,rowLabels:['TMEM C₀','TMEM C₁'],colLabels:['时段 0','时段 1','时段 2'],label:(i,j)=>[['MMA 写','结尾读','下一块 MMA'],['结尾读','MMA 写','结尾读']][i][j],fill:(i,j)=>i===j%2?'blue':'orange'});
    f.note(145,270,'复用 C₀ 或 C₁ 前等待其结尾读者结束',{anchor:'start'});
  },
  '23-waves'(f) {
    f.grid(130,60,{rows:2,cols:10,cw:48,ch:45,rowLabels:['第 1 波','第 2 波'],label:(i,j)=>i===1&&j>=5?'空闲':'工作',fill:(i,j)=>i===1&&j>=5?'gray':'blue'});
    f.text(370,205,'每格代表 10 个工作单元，共 100 个');
    f.box(130,280,480,45,'Stream-K：按 15000 个 K 迭代平均分配',{color:'orange'});
    f.note(130,360,'减少半满尾波，但部分输出需额外合并',{anchor:'start'});
  },
  "23-full-empty-cycle": function(f) {
    const labels=['生产者写入','满：写入完成','消费者 / MMA 读取','空：全部读完','下一轮覆盖'];
    const boxes=labels.map((s,i)=>f.box(25+i*135,65,118,64,s,{color:['blue','green','orange','green','blue'][i],size:12}));
    for(let i=0;i<4;i++)f.link(boxes[i],boxes[i+1],{arrow:'end'});
    f.note(350,210,'“满”授权消费，“空”授权复用；提交 MMA 还不等于共享块读完');
  },
};
