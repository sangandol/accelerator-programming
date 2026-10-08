export default {
  '16-grid-map'(f) {
    const a=f.box(25,50,135,55,'grid 点 (1,0)',{color:'blue'}),b=f.box(235,50,155,55,'块坐标 (1,0)',{color:'orange'}),c=f.box(465,50,185,55,'元素起点 (8,0)',{color:'green'});
    f.link(a,b,{arrow:'end'});f.link(b,c,{arrow:'end'});f.note(195,140,'index_map');f.note(430,140,'乘块形状 (8,128)');
    f.note(25,205,'grid / 块坐标 / 元素偏移使用不同的单位',{anchor:'start'});
  },
  '16-buffers'(f) {
    f.grid(130,60,{rows:3,cols:2,cw:160,ch:70,rowLabels:['输入 X','输入 Y','输出 O'],colLabels:['槽 0','槽 1'],label:(i,j)=>`128 KiB`,fill:(i)=>['blue','green','orange'][i]});
    f.note(130,315,'六块 × 128 KiB = 768 KiB，尚未计临时值与填充',{anchor:'start'});
  },
  '16-k-accumulator'(f) {
    const bs=['清零 + A₀B₀','加 A₁B₁','加 A₂B₂','舍入并写出'].map((s,i)=>f.box(20+i*160,55,125,55,s,{color:i===3?'orange':'blue'}));
    for(let i=0;i<3;i++)f.link(bs[i],bs[i+1],{arrow:'end'});
    f.note(20,170,'同一输出块：(i,j) 固定；K 维状态按序传递',{anchor:'start'});
  },
  '16-page-table'(f) {
    const table=f.grid(30,60,{rows:3,cols:1,cw:105,ch:46,rowLabels:['逻辑页 0','逻辑页 1','逻辑页 2'],label:i=>[3,0,5][i],fill:()=> 'orange'});
    const pool=f.grid(320,30,{rows:6,cols:1,cw:170,ch:34,label:i=>`物理页 ${i}`,fill:i=>[0,3,5].includes(i)?'blue':null});
    for(let i=0;i<3;i++)f.line([table.cell(i,0).R(),[205+i*22,83+i*46],[205+i*22,47+34*[3,0,5][i]],pool.cell([3,0,5][i],0).L()],{arrow:'end',color:'orange'});
    f.note(30,290,'页表预取到 SMEM；页池数据通过 DMA 搬进 VMEM',{anchor:'start'});
  },
};
