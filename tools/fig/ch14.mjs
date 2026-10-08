export default {
  '14-global-local'(f) {
    const g=f.grid(25,50,{rows:8,cols:16,cw:24,ch:24,fill:(i,j)=>i>=4&&j>=8&&j<12?'orange':null});
    f.frame(g.region(4,8,7,11),{color:'orange'});f.text(217,20,'全局 [8,16]，mesh [2,4]');
    const a=f.box(470,115,170,60,'设备 (1,2)：[4,4]',{color:'orange'});f.link(g.region(4,8,7,11),a,{arrow:'end'});
    f.note(470,225,'行 4–7，列 8–11',{anchor:'start'});
  },
  '14-sharding-choice'(f) {
    const a=f.box(25,115,190,65,'每设备：C⁽ʳ⁾ [8,32]',{color:'orange'});
    const b=f.box(350,25,200,60,'all-reduce → [8,32]',{color:'blue'}),c=f.box(350,210,200,60,'reduce-scatter → [4,32]',{color:'green'});
    f.link(a,b,{arrow:'end'});f.link(a,c,{arrow:'end'});f.note(350,105,'P(None,None)：复制',{anchor:'start'});f.note(350,293,'P(x,None)：行分片',{anchor:'start'});
  },
  '14-shard-map'(f) {
    const a=f.box(20,35,150,50,'设备 0：1,2 → 3',{color:'blue'}),b=f.box(330,35,150,50,'设备 1：3,4 → 7',{color:'green'});
    const c=f.box(175,155,150,50,'psum：3+7=10',{color:'orange'});f.link(a,c,{arrow:'end'});f.link(b,c,{arrow:'end'});
    const d=f.box(20,275,150,50,'设备 0：10',{color:'purple'}),e=f.box(330,275,150,50,'设备 1：10',{color:'purple'});f.link(c,d,{arrow:'end'});f.link(c,e,{arrow:'end'});
  },
  '14-multihost'(f) {
    for(let k=0;k<2;k++) {
      const x=25+k*335;f.box(x,30,280,145,'',{hollow:true,color:'gray'});f.text(x+140,52,`主机 ${k}：同一全局函数`);
      for(let j=0;j<4;j++)f.box(x+15+j*65,95,55,40,`设备 ${4*k+j}`,{color:k?'green':'blue',size:12});
      f.note(x+140,152,`本地装入行 ${k*512}–${k*512+511}`);
    }
    f.arrow([305,115],[360,115],{arrow:'both'});f.note(335,230,'全局 mesh 和集合调用跨两主机');
  },
};
