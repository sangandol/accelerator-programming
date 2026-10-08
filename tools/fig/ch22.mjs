export default {
  '22-thread-map'(f) {
    f.grid(100,60,{rows:4,cols:4,cw:120,ch:50,rowLabels:['块 0','块 1','块 2','块 3'],label:(i,j)=>`${256*i+64*j}–${256*i+64*j+63}`,fill:(i,j)=>i===3&&j===3?'orange':'blue'});
    f.text(340,312,'块 3 末尾：有效 960–999；越界 1000–1023');
  },
  '22-isa-path'(f) {
    const labels=['CUDA：一线程 z=x+y','PTX：载入、加法、存储','目标代码：分配寄存器','硬件：warp 请求与数值单元'];
    const bs=labels.map((s,i)=>f.box(60,25+i*90,310,45,s,{color:['blue','green','orange','purple'][i]}));for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(410,bs[1].cy,'公开虚拟 ISA',{anchor:'start'});f.note(410,bs[2].cy,'与目标相关',{anchor:'start'});
  },
  '22-reduction'(f) {
    const ws=[0,1,2,3,4,5,6,7].map(i=>f.box(20+i*77,30,65,45,`warp ${i}`,{color:'blue',size:12}));
    ws.forEach(v=>f.text(v.cx,12,'32'));
    const a=f.box(145,175,360,50,'共享部分和：32,32,32,32,32,32,32,32',{color:'orange',size:13});
    for(let i=0;i<8;i++)f.line([ws[i].B(),[ws[i].cx,145],a.T((i+.5)/8)],{arrow:'end'});
    const b=f.box(245,295,160,45,'首 warp：总和 256',{color:'green'});f.link(a,b,{arrow:'end'});
    f.note(540,200,'屏障后才读',{anchor:'start'});
  },
  '22-two-barriers'(f) {
    const labels=['协作载入','满：全部写好','计算读取','空：全部读完','下一轮覆盖'];
    const bs=labels.map((s,i)=>f.box(20+i*145,65,115,52,s,{color:i===1||i===3?'orange':'blue'}));for(let i=0;i<4;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(20,195,'第一个屏障保护本轮读取；第二个保护下一轮覆盖',{anchor:'start'});
  },
  "22-resource-min": function(f) {
    f.bars(185,45,{items:[['寄存器上限',4,'blue'],['共享内存上限',2,'orange'],['线程 / warp 上限',8,'green'],['块数上限',16,'purple']],w:360,fmt:v=>`${v} 块`});
    f.note(340,245,'取最小值：2 块 × 8 warp = 16 warp，算术占用率 25%');
    f.note(340,285,'假设 256 线程/块、64 寄存器/线程、80 KiB 共享/块');
  },
};
