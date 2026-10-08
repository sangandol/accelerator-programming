export default {
  '20-source-to-slots'(f) {
    const src=f.box(165,25,220,50,'y = exp(x) + z',{color:'blue'});
    const labels=['VMEM 载入','EUP 指数','向量加法','VMEM 写回'];
    const bs=labels.map((s,i)=>f.box(20+i*145,160,115,45,s,{color:['orange','purple','green','orange'][i]}));
    for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});f.link(src,bs[1],{arrow:'end'});
    f.note(20,265,'每类操作对应不同资源，存在结果就绪与存储依赖',{anchor:'start'});
  },
  '20-ready-times'(f) {
    f.timeline(95,30,{lanes:['指数 0','指数 1','指数 2'],bars:[[0,0,7,'延迟 7','blue'],[1,2,9,'延迟 7','green'],[2,4,11,'延迟 7','orange']],unit:46,ticks:[0,2,4,7,9,11]});
    f.note(95,245,'发射：0、2、4；结果就绪：7、9、11',{anchor:'start'});
  },
  '20-chain-throughput'(f) {
    const bars=[];for(let i=0;i<4;i++)bars.push([0,2*i,2*i+7,`${i}`,'blue'],[1,7*i,7*(i+1),`${i}`,'orange']);
    // 独立操作各自占一道，以免把并行的条画在同一道上。
    f.timeline(100,30,{lanes:['独立 0','独立 1','独立 2','独立 3','依赖链'],bars:[...Array.from({length:4},(_,i)=>[i,2*i,2*i+7,`${i}`,'blue']),...Array.from({length:4},(_,i)=>[4,7*i,7*(i+1),`${i}`,'orange'])],unit:22,ticks:[0,7,13,21,28]});
    f.note(100,390,'独立完成于 13；串行依赖完成于 28',{anchor:'start'});
  },
};
