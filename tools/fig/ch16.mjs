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
  "16-grid-order": function(f) {
    for(let r=0;r<2;r++) {
      f.text(25,45+r*120,r?'k 在外层':'k 在内层',{anchor:'start'});
      f.grid(155,20+r*120,{rows:1,cols:6,cw:73,ch:60,label:(_,j)=>r?`C${j%2}:k${Math.floor(j/2)}`:`C${Math.floor(j/3)}:k${j%3}`,fill:(_,j)=>(r?j%2:Math.floor(j/3))?'orange':'blue'});
    }
    f.note(330,290,'缩小到两个输出块、三个 K 块；同色是同一输出的状态');
    f.note(330,330,'自动驻留累加要求同色连续，并把 K 维声明为顺序维');
  },
  "16-gather-rotation": function(f) {
    f.text(285,20,'四子通道缩小模型：目标位置 0 想读取源位置 2');
    for(let k=0;k<4;k++) {
      f.text(40,80+k*80,`k=${k}`);
      f.grid(100,55+k*80,{rows:1,cols:4,cw:70,ch:45,label:(_,j)=>`x${(j+k)%4}`,fill:(_,j)=>j===0&&k===2?'green':'gray'});
    }
    f.note(460,240,'第 2 轮选中 x₂');
    f.note(285,405,'循环移位产生候选，选择掩码决定每个位置在哪一轮更新');
  },
  "16-dma-window": function(f) {
    for(let k=0;k<2;k++) {
      const x=30+k*270;f.text(x+110,18,k?'列 tile 1':'列 tile 0');
      const g=f.grid(x,55,{rows:16,cols:1,cw:170,ch:18,label:i=>i,fill:i=>i>=3&&i<8?'blue':i>=8&&i<11?'orange':null});
      f.frame(g.region(3,0,7,0),{color:'blue'});f.frame(g.region(8,0,10,0),{color:'orange'});
    }
    f.note(255,395,'蓝段：5 行 × 2 列 tile = 10 个 granule');
    f.note(255,430,'橙段：3 行 × 2 列 tile = 6 个 granule；总 8192 bytes');
  },
  "16-rng-address": function(f) {
    f.text(320,20,'全局 16×128；第 0 列的计数器 c = 128i');
    const a=f.box(25,70,240,60,'行 0–7：c=0,…,896',{color:'blue'}),b=f.box(360,70,240,60,'行 8–15：c=1024,…,1920',{color:'orange'});
    const c=f.box(165,220,285,65,'固定 key：rᵢ=F(key,cᵢ)',{color:'green'});f.link(a,c,{arrow:'end'});f.link(b,c,{arrow:'end'});
    f.note(320,350,'改变块大小不改逻辑身份；第二块从零计数会重用随机样本');
  },
};
