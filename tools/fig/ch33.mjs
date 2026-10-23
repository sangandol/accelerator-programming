export default {
  '33-instance'(f) {
    const n=4,w=136,gap=26,x0=110;
    const top=[],bot=[];
    for(let g=0;g<n;g++){
      const x=x0+g*(w+gap);
      f.text(x+w/2,18,'GPU '+g);
      top.push(f.box(x,35,w,46,'各自请求的注意力',{color:'blue'}));
      bot.push(f.box(x,185,w,46,`专家 ${g*96}–${g*96+95}`,{color:'orange'}));
    }
    for(const a of top)for(const b of bot)f.link(a,b,{arrow:'end',color:'gray'});
    f.text(95,58,'数据并行',{anchor:'end'});f.text(95,208,'专家并行',{anchor:'end'});
    f.text(95,133,'全交换',{anchor:'end'});
    f.note(x0,265,'每层：各自算注意力 → 全交换分发 → 专家计算 → 全交换回收；最慢的一块决定整层',{anchor:'start'});
  },
};
