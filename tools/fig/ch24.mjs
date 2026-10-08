export default {
  '24-tile-grid'(f) {
    f.grid(30,55,{rows:2,cols:3,cw:145,ch:85,label:(i,j)=>`实例 (${i},${j})`,fill:(i,j)=>i===1&&j===2?'orange':'blue'});
    f.brace(30,270,465,'C：256×384，每实例输出 128×128');
    f.note(30,330,'块内线程、寄存器与流水线由后端分配',{anchor:'start'});
  },
  '24-pointer-tile'(f) {
    const a=f.box(25,65,150,50,'行：4,5；步长 8',{color:'blue'}),b=f.box(25,200,150,50,'K：0,1,2；步长 1',{color:'green'});
    const g=f.grid(335,115,{rows:2,cols:3,cw:65,ch:48,label:(i,j)=>32+8*i+j,fill:()=> 'orange'});
    f.link(a,g.cell(0,0),{arrow:'end'});f.link(b,g.cell(1,0),{arrow:'end'});f.note(335,260,'偏移 = 行×8 + K',{anchor:'start'});
  },
  '24-abstraction'(f) {
    const rows=[['数学函数','输出关系、精度'],['tile 程序','块、网格、K 循环'],['显式 warp / 布局','数据归属、资源预算'],['异步硬件协议','完成身份、相位、所有权']];
    f.grid(30,50,{rows:4,cols:2,cw:230,ch:62,label:(i,j)=>rows[i][j],fill:(i,j)=>i%2?'orange':'blue'});
    f.note(30,355,'下降一层是增加控制能力，也增加需要证明的条件',{anchor:'start'});
  },
  "24-lowering-contract": function(f) {
    const a=f.box(25,40,190,65,'数学 tile：形状与掩码',{color:'blue',size:13});
    const b=f.box(330,40,270,65,'后端：线程、布局、搬运、MMA',{color:'orange',size:13});f.link(a,b,{arrow:'end'});
    const c=f.box(330,215,270,65,'机器程序：资源与同步约束',{color:'green'});f.link(b,c,{arrow:'end'});
    f.note(290,360,'抽象省去映射细节，但数学覆盖、精度和依赖仍由规格确定');
  },
};
