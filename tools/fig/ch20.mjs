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
  "20-exact-container": function(f) {
    const a=f.box(25,30,185,55,'机器映像 I',{color:'blue'}),b=f.box(355,30,250,55,'精确清单 D(I) + 编码约束',{color:'orange',size:13});f.link(a,b,{arrow:'end'});
    const c=f.box(25,200,185,55,'A(D(I)) = I',{color:'green'});f.line([b.B(),[480,150],[120,150],c.T()],{arrow:'end'});
    f.box(280,190,325,150,'',{hollow:true,color:'gray'});f.text(442,220,'可执行容器');f.note(442,263,'映像 + 分配 + 调用约定');f.note(442,303,'入口 + 程序身份 + 元数据');
    f.note(320,405,'恢复字节是表示正确；调度、同步与调用约定还要独立证明');
  },
  "20-time-boundaries": function(f) {
    f.timeline(160,35,{lanes:['主机提交','设备程序','kernel 内部','结果通知'],bars:[[0,0,2,'提交','gray'],[1,2,10,'前导、kernel、尾声','blue'],[2,4,8,'关注的片段','orange'],[3,10,12,'结果就绪','green']],unit:36,ticks:[0,2,4,6,8,10,12]});
    f.note(365,330,'四个区间有不同的起止事件；图中数字只是边界示意');
  },
};
