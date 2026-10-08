export default {
  '21-sm-chip'(f) {
    const sms=[0,1,2].map(i=>{
      const x=25+i*210;f.box(x,25,180,205,'',{hollow:true,color:'blue'});f.text(x+90,48,`SM ${i}`);
      f.box(x+15,75,150,40,'warp 调度与运算',{color:'green'});
      f.box(x+15,130,150,35,'每线程寄存器',{color:'orange'});
      return f.box(x+15,178,150,35,'共享内存 / L1',{color:'yellow'});
    });
    const l2=f.box(210,310,230,45,'全芯片 L2',{color:'purple'}),h=f.box(210,420,230,45,'全芯片 HBM',{color:'gray'});
    sms.forEach(v=>f.link(v,l2,{arrow:'both'}));f.link(l2,h,{arrow:'both'});
  },
  '21-warp-mask'(f) {
    for(let i=0;i<2;i++) {
      f.text(25,30+i*115,i?'奇数分支：5 条':'偶数分支：3 条',{anchor:'start'});
      f.grid(25,65+i*115,{rows:1,cols:16,cw:32,ch:34,label:(r,j)=>`${j}`,fill:(r,j)=>j%2===i?'orange':'gray'});
    }
    f.note(25,270,'画前 16 个通道；实际 warp 有 32 个。灰色通道被屏蔽',{anchor:'start'});
  },
  '21-memory-path'(f) {
    const a=f.box(25,45,120,50,'HBM / L2',{color:'gray'}),b=f.box(250,45,160,50,'共享内存',{color:'blue'}),c=f.box(515,45,120,50,'寄存器',{color:'orange'});
    f.link(a,b,{arrow:'end'});f.link(b,c,{arrow:'end'});f.note(196,150,'请求合并：段数');f.note(464,150,'bank 冲突：映射');
    f.note(25,220,'每一条传输边分别分析，修好一条不会自动修好另一条',{anchor:'start'});
  },
  '21-mma-interface'(f) {
    const data=[['Ampere','32 线程 warp','寄存器片段','寄存器'],['Hopper','128 线程组','共享内存 / 寄存器','寄存器'],['Blackwell','单线程发起','共享内存 / TMEM','TMEM']];
    f.grid(25,70,{rows:3,cols:4,cw:175,ch:68,colLabels:['代际','发起者','操作数来源','累加位置'],label:(i,j)=>data[i][j],fill:(i,j)=>j===3?'orange':'blue'});
    f.note(25,330,'发起粒度改变后，操作数布局、完成协议和输出所有权都要重查',{anchor:'start'});
  },
};
