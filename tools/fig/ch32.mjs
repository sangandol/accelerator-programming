export default {
  '32-chain'(f) {
    const r=f.box(20,90,70,40,'$h_0$',{color:'gray'});
    const p1=f.box(130,90,90,40,'第 1 页',{color:'blue'});
    const p2=f.box(260,90,90,40,'第 2 页',{color:'blue'});
    const a3=f.box(390,40,110,40,'A 的第 3 页',{color:'green'});
    const b3=f.box(390,140,110,40,'B 的第 3 页',{color:'orange'});
    const a4=f.box(540,40,110,40,'A 的第 4 页',{color:'green'});
    f.link(r,p1,{arrow:'end'});f.link(p1,p2,{arrow:'end'});f.link(p2,a3,{arrow:'end'});f.link(p2,b3,{arrow:'end'});f.link(a3,a4,{arrow:'end'});
    f.note(175,150,'$h_1$');f.note(305,150,'$h_2$');
    f.note(20,215,'每页的键 $h_j$ 由上一页的键和本页的词元算出；A、B 共用前两页',{anchor:'start'});
    f.note(540,110,'末端：可先淘汰',{anchor:'start'});
  },
  '32-tiers'(f) {
    const hbm=f.box(170,20,260,48,'HBM：运行中的请求（页池）',{color:'blue'});
    const dram=f.box(170,120,260,48,'主机内存：窗口快照，几分钟',{color:'green'});
    const st=f.box(170,220,260,48,'存储集群：全局缓存，≥ 72 小时',{color:'orange'});
    f.link(st,dram,{arrow:'end'});f.link(dram,hbm,{arrow:'end'});
    f.note(310,194,'取回（异步）',{anchor:'start'});f.note(310,94,'逐层换入，与前向重叠',{anchor:'start'});
    const pf=f.box(500,20,130,48,'prefill 机器',{color:'gray'});
    const dc=f.box(500,120,130,48,'decode 机器',{color:'gray'});
    f.link(pf,dc,{arrow:'end'});f.note(575,94,'逐层交接',{anchor:'start'});
    f.line([[pf.cx,pf.y],[pf.cx,8],[20,8],[20,st.cy],[st.x,st.cy]],{arrow:'end',dash:'4 3'});
    f.note(28,150,'写入新缓存',{anchor:'start'});
  },
};
