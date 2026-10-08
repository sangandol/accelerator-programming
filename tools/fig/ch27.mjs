export default {
  '27-hierarchy'(f) {
    const nics=[];
    for(let k=0;k<2;k++) {
      const x=25+k*385;f.box(x,30,330,215,'',{color:'gray',hollow:true});f.text(x+165,53,`NVLink 域 ${k}`);
      const s=f.box(x+95,85,140,40,'NVSwitch',{color:'purple'});
      let representative;
      for(let i=0;i<4;i++) {const g=f.box(x+15+i*77,170,65,42,`GPU ${4*k+i}`,{color:'blue',size:12});f.link(s,g,{arrow:'both'});if(i===1)representative=g;}
      const nic=f.box(x+55,330,140,42,'GPU 对应网卡',{color:'orange'});nics.push(nic);f.link(representative,nic,{arrow:'both'});
    }
    f.link(nics[0],nics[1],{arrow:'both',color:'orange'});f.note(343,315,'跨域网络');
    f.note(25,435,'每域只画一条代表性的 GPU / 网卡路径；域内与域间的带宽分别计账',{anchor:'start'});
  },
  '27-symmetric'(f) {
    f.grid(30,65,{rows:4,cols:3,cw:155,ch:50,rowLabels:['GPU 0','GPU 1','GPU 2','GPU 3'],colLabels:['偏移 0','偏移 4096','偏移 8192'],label:(i,j)=>`(设备 ${i}, 偏移 ${j*4096})`,fill:(i,j)=>i===2&&j===1?'orange':'blue'});
    f.note(30,330,'统一的是逻辑位置；库或对等映射给出实际远端地址',{anchor:'start'});
  },
  '27-overlap'(f) {
    f.timeline(100,30,{lanes:['GEMM 块','通信块'],bars:[[0,0,2,'0','blue'],[0,2,4,'1','blue'],[0,4,6,'2','blue'],[0,6,8,'3','blue'],[1,2,3.1,'0','orange'],[1,4,5.1,'1','orange'],[1,6,7.1,'2','orange'],[1,8,9.1,'3','orange']],unit:55,ticks:[0,2,4,6,8,9.1]});
    f.note(100,220,'每块算完才能通信，四块总时间 9.1 ms（不计争用）',{anchor:'start'});
  },
  '27-moe-routing'(f) {
    const src=f.box(25,130,145,60,'词元隐藏向量 X',{color:'blue'}),recv=f.box(280,130,155,60,'远端域：一份 X',{color:'orange'}),a=f.box(545,40,145,55,'专家 A',{color:'green'}),b=f.box(545,245,145,55,'专家 B',{color:'purple'});
    f.link(src,recv,{arrow:'end'});f.link(recv,a,{arrow:'end'});f.link(recv,b,{arrow:'end'});
    f.note(220,95,'跨网卡');f.note(505,190,'域内复制');f.note(25,350,'路由和返回仍保留词元编号、专家身份与权重',{anchor:'start'});
  },
};
