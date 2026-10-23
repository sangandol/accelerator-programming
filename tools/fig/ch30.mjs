export default {
  '30-mla'(f) {
    f.text(150,18,'按定义：先还原每个头的键');
    f.text(470,18,'吸收之后：把还原矩阵并进查询');
    const c1=f.box(20,50,90,40,'缓存 $c_j$',{color:'orange'});
    const u1=f.box(140,50,100,40,'乘 $W^{UK}_h$',{color:'gray'});
    const k1=f.box(270,50,70,40,'$k_{h,j}$',{color:'blue'});
    f.link(c1,u1,{arrow:'end'});f.link(u1,k1,{arrow:'end'});
    const s1=f.box(270,130,70,40,'分数',{color:'green'});f.link(k1,s1,{arrow:'end'});
    const q1=f.box(140,130,100,40,'$q_h$',{color:'blue'});f.link(q1,s1,{arrow:'end'});
    f.note(20,200,'每步对 $T$ 项、$H$ 个头各还原一次',{anchor:'start'});
    const q2=f.box(390,50,70,40,'$q_h$',{color:'blue'});
    const u2=f.box(490,50,120,40,'乘 $W^{UK}_h$ 的转置',{color:'gray'});
    const t2=f.box(640,50,110,40,'吸收后的查询',{color:'blue'});
    f.link(q2,u2,{arrow:'end'});f.link(u2,t2,{arrow:'end'});
    const s2=f.box(660,130,70,40,'分数',{color:'green'});f.link(t2,s2,{arrow:'end'});
    const c2=f.box(490,130,120,40,'缓存 $c_j$',{color:'orange'});f.link(c2,s2,{arrow:'end'});
    f.note(390,200,'每步每个头只变换一次查询',{anchor:'start'});
  },
  '30-v41-layers'(f) {
    const mode=(l)=>l<2?'S':l<20?((l-2)%6===0?'F':'U'):((l-20)%4===0?(l===20?'F':'R'):'U');
    const col={S:'gray',F:'blue',R:'green',U:'white'};
    for(let row=0;row<2;row++){
      f.text(60,52+row*70,row?'解码器':'编码器',{anchor:'end'});
      f.grid(75,35+row*70,{rows:1,cols:20,cw:30,ch:34,fill:(r,c)=>col[mode(row*20+c)],label:(r,c)=>mode(row*20+c)});
      f.note(690,52+row*70,row?'$r = 1$':'$r = 2$（第 2–19 层）',{anchor:'start'});
    }
    const items=[['S','只有滑动窗口','gray'],['F','完整：产生缓存项','blue'],['R','重选：重新打分','green'],['U','复用','white']];
    items.forEach(([k,s,c],i)=>{f.box(75+i*170,190,26,26,k,{color:c});f.note(108+i*170,203,s,{anchor:'start'});});
  },
};
