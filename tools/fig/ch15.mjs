export default {
  '15-tensorcore'(f) {
    f.box(20,20,620,350,'',{color:'gray',hollow:true});f.text(330,40,'一个 TPU TensorCore');
    const ctl=f.box(240,65,180,40,'标量控制 / SMEM',{color:'orange'});
    const v=f.box(60,155,130,48,'VPU / EUP / XLU',{color:'green',size:12});
    const reg=f.box(240,155,180,48,'向量寄存器',{color:'blue'}),mx=f.box(465,155,130,48,'MXU × 4',{color:'purple'});
    f.link(v,reg,{arrow:'both'});f.link(reg,mx,{arrow:'both'});f.link(ctl,reg,{arrow:'end',dash:'4 3'});
    const vm=f.box(240,280,180,48,'VMEM：操作数块',{color:'yellow'});f.link(vm,reg,{arrow:'both'});
    const dma=f.box(240,435,180,48,'DMA + 完成对象',{color:'orange'}),hbm=f.box(240,570,180,48,'HBM：全数组',{color:'gray'});
    f.link(hbm,dma,{arrow:'both'});f.link(dma,vm,{arrow:'both'});
    f.note(25,530,'实线是数据；虚线是控制',{anchor:'start'});
  },
  '15-bundle'(f) {
    f.grid(80,70,{rows:3,cols:5,cw:120,ch:52,rowLabels:['拍 0','拍 1','拍 2'],colLabels:['标量','向量 ALU','向量读','MXU 提交','结果取回'],label:(i,j)=>[['地址加法','独立加法','读 R₀','',''],['循环计数','','','乘 R₀',''],['','消费旧结果','','','取早先结果']][i][j],fill:(i,j)=>[['orange','green','blue',null,null],['orange',null,null,'purple',null],[null,'green',null,null,'purple']][i][j]});
    f.note(80,290,'槽并排不保证结果同时完成；此图是资源示意，不是真实汇编',{anchor:'start'});
  },
  '15-vector-layout'(f) {
    f.grid(35,45,{rows:2,cols:2,cw:200,ch:75,label:(i,j)=>`R${2*i+j}：8×128 f32`,fill:(i,j)=>['blue','orange','green','purple'][2*i+j]});
    f.brace(35,240,435,'逻辑块：16×256');
    f.text(545,75,'R₀ + R₁');f.text(545,150,'R₂ + R₃');
    f.arrow([435,82],[470,82]);f.arrow([435,157],[470,157]);
    f.note(35,300,'先把两列块按对应位置相加，再跨 128 通道归约',{anchor:'start'});
  },
  '15-mxu-sequence'(f) {
    const labels=['权重暂存','活动权重','乘法在途','结果队列'];
    const bs=labels.map((s,i)=>f.box(25+i*160,70,125,55,s,{color:['orange','blue','purple','green'][i]}));
    for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(175,42,'装入');f.note(495,42,'计算完成');
    f.note(345,160,'左操作数从寄存器提交');f.note(585,200,'取回到寄存器');
    f.note(25,260,'当前权重运算时，可以准备下一份暂存权重',{anchor:'start'});
  },
  '15-generations'(f) {
    const rows=['矩阵单元','片上 / HBM','设备边界','互联'];
    const labels=['形状、格式、数量','容量、带宽、布局','TC / 芯粒 / device','拓扑、方向、注入'];
    for(let i=0;i<4;i++) {f.box(25,40+i*65,140,42,rows[i],{color:'blue'});f.box(230,40+i*65,260,42,labels[i],{color:'orange'});f.arrow([165,61+i*65],[230,61+i*65]);}
    f.note(25,335,'迁移保留算法与不变量；按目标代际重填四列参数',{anchor:'start'});
  },
};
