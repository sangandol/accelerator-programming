export default {
  '12-mesh'(f) {
    for(let k=0;k<2;k++) {
      const x=25+k*350;f.text(x+140,20,k?'行列都切分':'仅沿 x 切行');
      f.grid(x,60,{rows:2,cols:2,cw:140,ch:75,label:(i,j)=>`行 ${4*i}–${4*i+3}，列 ${k?2*j:0}–${k?2*j+1:3}`,fill:(i,j)=>k?['blue','orange','green','purple'][2*i+j]:i?'green':'blue'});
    }
    f.note(25,252,'左右列是 y 坐标，上下行是 x 坐标',{anchor:'start'});
  },
  '12-matmul-shards'(f) {
    const a=f.box(25,35,175,52,'切输出行：各算一半行',{color:'blue'}),b=f.box(300,35,180,52,'拼接即完整 C',{color:'green'});f.link(a,b,{arrow:'end'});
    const c=f.box(25,160,175,52,'切 K：各算半个点积',{color:'orange'}),d=f.box(300,160,180,52,'50 + 250 = 300',{color:'purple'});f.link(c,d,{arrow:'end'});
    f.note(220,125,'切求和下标，需要合并贡献');
    f.note(25,260,'两块部分和覆盖同一输出位置，不是两块完整输出',{anchor:'start'});
  },
  '12-fsdp-lifetime'(f) {
    f.timeline(115,30,{lanes:['层 ℓ 完整权重','层 ℓ+1 完整权重'],bars:[[0,1,5,'汇集、计算、释放','blue'],[1,3,7,'预取、计算、释放','orange']],unit:70,ticks:[0,1,2,3,4,5,6,7]});
    f.brace(325,205,465,'两个工作区同时存在');
    f.note(115,270,'持久分片状态在整步都存在，不画在此时间线上',{anchor:'start'});
  },
  '12-context-balance'(f) {
    const cs=['blue','orange','green','purple'];
    f.grid(30,55,{rows:8,cols:8,cw:32,ch:28,fill:(i,j)=>j<=i?cs[Math.min(i,7-i)]:null,label:(i,j)=>j===i?`${i}`:''});
    for(let i=0;i<4;i++) f.box(340,55+i*50,205,36,`设备 ${i}：段 ${i} 与 ${7-i}，共 9 对`,{color:cs[i],size:13});
    f.note(30,320,'每行是一个查询，每列是一个键；同色行配给同一设备',{anchor:'start'});
  },
  '12-pipeline'(f) {
    const bars=[];for(let s=0;s<4;s++)for(let b=0;b<8;b++)bars.push([s,s+b,s+b+1,`${b}`,['blue','orange','green','purple'][b%4]]);
    f.timeline(95,30,{lanes:['段 0','段 1','段 2','段 3'],bars,unit:48,ticks:Array.from({length:12},(_,i)=>i)});
    f.note(95,320,'每段工作 8 个单位，总跨度 11；空泡占 3/11',{anchor:'start'});
  },
};
