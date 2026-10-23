export default {
  '31-arch'(f) {
    const cl=f.box(20,95,80,50,'客户端',{color:'gray'});
    const api=f.box(140,70,150,100,'接入层',{color:'orange'});
    f.note(215,140,'模板、词元化、流式解码');
    const eng=f.box(340,70,140,100,'引擎',{color:'blue'});
    f.note(410,140,'调度、页管理');
    f.link(cl,api,{arrow:'both'});f.link(api,eng,{arrow:'both'});
    const ws=[0,1,2,3].map(i=>f.box(540,20+i*52,90,38,'执行者 '+i,{color:'green'}));
    const ds=[0,1,2,3].map(i=>f.box(680,20+i*52,90,38,'加速器 '+i,{color:'purple'}));
    ws.forEach((w,i)=>{f.link(eng,w,{arrow:'end'});f.link(w,ds[i],{arrow:'both'});});
    f.note(215,200,'JSON 与文字');f.note(410,200,'词元编号');f.note(655,240,'设备上的数组');
  },
  '31-async'(f) {
    f.text(20,15,'串行：每步用时 $T_g + T_c$',{anchor:'start'});
    f.timeline(60,35,{lanes:['CPU','设备'],unit:36,bars:[[0,0,1,'S1','orange'],[1,1,4,'F1+A1','blue'],[0,4,5,'P1','green'],[0,5,6,'S2','orange'],[1,6,9,'F2+A2','blue'],[0,9,10,'P2','green']],ticks:false});
    f.text(20,155,'异步调度：每步用时 $\\max(T_g, T_c)$',{anchor:'start'});
    f.timeline(60,175,{lanes:['CPU','设备'],unit:36,bars:[[0,0,1,'S1','orange'],[1,1,4,'F1+A1','blue'],[0,1,2,'S2','orange'],[1,4,7,'F2+A2','blue'],[0,4,5,'S3','orange'],[0,5,6,'P1','green'],[1,7,10,'F3+A3','blue'],[0,7,8,'S4','orange'],[0,8,9,'P2','green']],ticks:false});
    f.note(60,295,'S：调度并发起；F+A：前向与抽样；P：处理输出',{anchor:'start'});
  },
};
