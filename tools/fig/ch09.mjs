export default {
  '09-ownership'(f) {
    const labels=['可写','读入中','可读','读者使用中','再次可写'];
    const b=labels.map((s,i)=>f.box(20+i*128,50,102,46,s,{color:['gray','orange','green','blue','gray'][i]}));
    for(let i=0;i<4;i++)f.link(b[i],b[i+1],{arrow:'end'});
    f.note(260,124,'写入完成'); f.note(516,124,'全部读者结束');
    f.note(20,165,'复用槽位要证明两条边，不能只等到发起完成',{anchor:'start'});
  },
  '09-counter'(f) {
    f.timeline(125,30,{lanes:['DMA A','DMA B','共用等待'],bars:[[0,0,6,'A：慢','blue'],[1,1,3,'B：先完成','orange'],[2,3,4,'已返回','red']],unit:58,ticks:[0,1,2,3,4,5,6]});
    f.note(125,250,'t=3 计数已满 4096，却不能证明 A 的数据到了',{anchor:'start'});
    const a=f.box(125,300,160,40,'A → 完成对象 A',{color:'blue'}), b=f.box(325,300,160,40,'B → 完成对象 B',{color:'orange'});
    f.note(125,365,'分开等待：完成身份与数据块一一对应',{anchor:'start'});
  },
  '09-pipeline'(f) {
    const colors=['orange','blue','green','purple']; const bars=[];
    for(let i=0;i<4;i++) {const t=5*i;bars.push([0,t,t+3,`L${i}`,colors[i]],[1,t+3,t+8,`C${i}`,colors[i]],[2,t+8,t+10,`S${i}`,colors[i]]);}
    f.timeline(95,30,{lanes:['读入 3','计算 5','写回 2'],bars,unit:25,ticks:[0,5,10,15,20,25]});
    f.note(95,250,'首块用 10，之后每 5 完成一块',{anchor:'start'});
  },
  '09-double-buffer'(f) {
    f.text(230,18,'同一槽位在不同时刻属于不同块');
    const rows=['输入槽 0','输入槽 1','输出槽 0','输出槽 1'];
    const texts=[['块 0','空闲','块 2'],['预取块 1','块 1','空闲'],['算块 0','写回块 0','等写回后算块 2'],['空闲','算块 1','写回块 1']];
    f.grid(125,65,{rows:4,cols:3,cw:155,ch:48,rowLabels:rows,colLabels:['迭代 0','迭代 1','迭代 2'],label:(i,j)=>texts[i][j],fill:(i,j)=>i<2?'blue':'orange'});
    f.note(125,290,'输入和输出是不同的存储；两种等待各保护自己的槽',{anchor:'start'});
  },
  "09-latency-slots": function(f) {
    f.text(330,20,'读入延迟 7，计算每块 2，需要 1 + ceil(7/2) = 5 槽');
    const labels=['算块 0','读块 1','读块 2','读块 3','准备块 4'];
    for(let i=0;i<5;i++) {
      f.text(82+i*130,63,`槽 ${i}`);
      f.box(25+i*130,88,114,54,labels[i],{color:i===0?'green':i===4?'gray':'blue'});
      f.note(82+i*130,173,i===0?'7–9 占用':i===4?'8 发起':`${2*i} 发起`);
    }
    f.note(330,225,'时刻 7 的截面；输入槽从发起读入到消费者结束一直存活');
  },
};
