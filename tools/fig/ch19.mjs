export default {
  '19-remote-dma'(f) {
    const a=f.box(30,65,150,55,'发送设备 0：A',{color:'blue'}),b=f.box(340,65,150,55,'接收设备 1：B',{color:'green'});f.link(a,b,{arrow:'end'});f.note(255,35,'远程 DMA');
    f.box(30,190,150,40,'send：A 可复用',{color:'blue'});f.box(340,190,150,40,'recv：B 可读取',{color:'green'});
    f.line([b.R(),[525,92],[525,300],[10,300],[10,92],a.L()],{arrow:'end',color:'orange'});f.note(255,340,'B 的所有读者结束，再回送信用');
  },
  '19-ring-identity'(f) {
    const R=f.ring(150,155,{n:4,r:100,label:i=>`${i}`});for(let i=0;i<4;i++)f.link(R.node(i),R.node((i+1)%4),{arrow:'end'});
    f.grid(340,80,{rows:3,cols:3,cw:90,ch:55,colLabels:['步','发送块','收到块'],label:(i,j)=>[[1,2,1],[2,1,0],[3,0,3]][i][j],fill:(i,j)=>j===2?'orange':'blue'});
    f.note(340,305,'固定观察设备 2',{anchor:'start'});
  },
  '19-reduce-scatter'(f) {
    const labels=['设备 1：x₁','设备 2：+x₂','设备 3：+x₃','设备 0：+x₀'];
    const bs=labels.map((s,i)=>f.box(25+i*160,50,125,52,s,{color:['blue','green','purple','orange'][i]}));for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.text(332,180,'固定看块 0 的贡献：每个设备恰好加入一次');
  },
  '19-alltoall-counts'(f) {
    f.grid(40,55,{rows:3,cols:3,cw:60,ch:52,rowLabels:['源 0','源 1','源 2'],colLabels:['目标 0','目标 1','目标 2'],label:(i,j)=>[[0,2,1],[3,0,1],[1,4,0]][i][j],fill:(i,j)=>j===1?'orange':null});
    f.grid(335,115,{rows:1,cols:6,cw:43,ch:55,label:(i,j)=>j<2?'源 0':'源 2',fill:(i,j)=>j<2?'blue':'green'});f.text(464,75,'设备 1 的接收区：6 行');
    f.note(335,225,'源 0： [0,2)；源 2： [2,6)',{anchor:'start'});
  },
  "19-storage-endpoints": function(f) {
    for(let k=0;k<2;k++) {
      const x=25+k*350;f.box(x,35,290,220,'',{hollow:true,color:'gray'});f.text(x+145,58,`芯片 ${k}`);
      f.box(x+20,100,112,55,'TC0 VMEM',{color:'blue'});f.box(x+158,100,112,55,'TC1 VMEM',{color:'blue'});
      f.box(x+20,190,250,40,'共享 CMEM / HBM',{color:'orange'});
    }
    f.arrow([315,210],[375,210]);
    f.box(210,340,270,55,'主机 pinned memory',{color:'green'});
    f.note(350,465,'目的存储、路由接收者与完成信号归属要分别确定');
  },
};
