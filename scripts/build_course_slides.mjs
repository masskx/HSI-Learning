// Native editable classroom decks. Uses the bundled @oai/artifact-tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {pathToFileURL} from 'node:url';
const root=path.resolve(import.meta.dirname,'..');
const runtime=process.env.HSI_RUNTIME_ROOT || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies');
process.env.RUNTIME_NODE_MODULES ||= path.join(runtime,'node/node_modules');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const data=JSON.parse(await fs.readFile(process.argv[2] || path.join(root,'results/slide_revision_build/content.json'),'utf8'));
const selected=process.argv.slice(3);
const build=path.join(root,'results/slide_revision_build');
const output=path.join(root,'slides/classroom');
await fs.mkdir(build,{recursive:true}); await fs.mkdir(output,{recursive:true});
const C={ink:'#152E50',sub:'#5C6B7A',accent:'#315DAB',line:'#D9E2E9',panel:'#F4F6F9',code:'#202E3C'};
const W=1280,H=720,FONT='Microsoft YaHei';
function rect(slide,x,y,w,h,fill,line='none') {
 return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:line==='none'?0:1}});
}
function text(slide,value,x,y,w,h,size=26,options={}) {
 const shape=slide.shapes.add({geometry:'textbox',name:options.name || String(value).slice(0,30),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=String(value); shape.text.style={typeface:options.mono?'Consolas':FONT,fontSize:size,color:options.color || C.ink,bold:!!options.bold,wrap:'square',verticalAlignment:'middle',autoFit:'none',insets:0};
 return shape;
}
const covers={L00:'环境与\n第一次前向',L01:'理解\n高光谱数据',L02:'划分与\n评价指标',L03:'建立可信的\nSVM 基线',L04:'从像元\n构建 Patch',L05A:'模型的一次\n参数更新',L05:'卷积的轴\n与感受野',L06:'读懂\nHybridSN',L07:'类别不均衡\n与错误分析',L08:'原型网络\n与少样本任务',L09:'未知类\n与拒识决策',L10:'从实验记录\n到研究证据'};
function frame(p,l,title,notes='') {
 const slide=p.slides.add(); slide.background.fill='#FFFFFF';
 text(slide,`${l.id}  /  HSI LEARNING`,72,30,650,30,16,{color:C.sub});
 text(slide,String(p.slides.items.length).padStart(2,'0'),1150,30,58,30,16,{color:C.sub});
 text(slide,title,72,90,1136,85,40,{bold:true});
 rect(slide,72,193,56,3,C.accent);
 slide.speakerNotes.text=notes+`\n课堂入口: ${l.notebook || 'docs/teaching/research-case.md'}\n定位: ${l.cells.join(', ')}\n先预测，再运行；答错时回到手算例。\nAI辅助：${l.ai}\n人工核验：${l.verify}\n`+l.sources.map(k=>data.sources[k]?.join('\n')).join('\n')+'\n运行配置、数据哈希和结果：artifacts/teaching/classroom-facts.json';
 return slide;
}
async function picture(slide,file,x,y,w,h) {
 const bytes=await fs.readFile(file);
 slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:path.basename(file),fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
function bullets(slide,points,x=88,y=230,w=560,size=28,gap=84) {
 points.forEach((v,i)=>text(slide,v,x,y+i*gap,w,gap-10,size));
}
function nativeTable(slide,values,x,y,w,h) {
 const rowH=h/values.length, colW=w/values[0].length;
 rect(slide,x,y,w,2,C.ink);
 values.forEach((row,r)=>{
  if(r===0)rect(slide,x,y+rowH,w,1.5,C.line);
  row.forEach((v,c)=>text(slide,v,x+c*colW+14,y+r*rowH+6,colW-28,rowH-12,values.length>5?22:25,{bold:r===0,color:r===0?C.ink:C.sub}));
 });
 rect(slide,x,y+h,w,2,C.ink);
}
function flow(slide,labels,x=690,y=265,w=490,h=350) {
 const step=h/labels.length;
 labels.forEach((v,i)=>{
  text(slide,String(i+1).padStart(2,'0'),x,y+i*step,56,45,24,{color:C.accent});
  text(slide,v,x+76,y+i*step,w-76,step-16,27,{bold:true});
  if(i<labels.length-1)rect(slide,x+20,y+i*step+51,1,step-58,C.line);
 });
}
const manifest=[];
for(const l of data.lessons) {
 if(selected.length && !selected.includes(l.id))continue;
 const p=Presentation.create({slideSize:{width:W,height:H}}); const pages=[];
 let s=frame(p,l,'',`开场问题：${l.exit_question}\n原标题：${l.title}\n参考节奏：5分钟回顾，10分钟手算，10分钟代码，12分钟练习，5分钟错误分析，3分钟退出题。`);
 text(s,covers[l.id] || l.title,72,222,580,206,68,{bold:true});
 const hero=l.pages.find(e=>e.image_asset) || data.lessons.find(e=>e.id==='L01').pages.find(e=>e.title.includes('光谱') && e.image_asset);
 if(hero)await picture(s,hero.image_asset,692,218,516,234);
 s.speakerNotes.text+='\n封面图源：'+(hero?.image?.join(' / ') || '');
 rect(s,72,470,1136,1,C.line);
 text(s,l.subtitle || l.title,72,497,1136,58,26,{color:C.sub});
 text(s,'先修  /  '+l.prerequisites,72,601,1136,64,22,{color:C.sub});
 pages.push({type:'cover',title:l.title});
 s=frame(p,l,'这节课，你将完成什么','交付物以代码、配置、输出和解释为准。先做完整示例，再做填空与迁移任务。');
 l.outputs.forEach((v,i)=>{
  text(s,String(i+1).padStart(2,'0'),72,254+i*125,80,72,36,{color:C.accent});
  text(s,v,184,254+i*125,1000,90,32);
  if(i<l.outputs.length-1)rect(s,184,359+i*125,1000,1,C.line);
 });
 pages.push({type:'objectives',title:'这节课，你将完成什么'});
 for(const entry of l.pages) {
  s=frame(p,l,entry.title,(entry.notes || '')+'\n'+(entry.visual || []).join(' / ')+(entry.image?'\n图源：'+entry.image.join(' / '):''));
  if(entry.image_asset) {
   await picture(s,entry.image_asset,72,212,1136,410);
   entry.points.forEach((v,i)=>{
    const cw=1136/entry.points.length;
    rect(s,72+i*cw,627,28,2,C.accent);
    text(s,v,72+i*cw,640,cw-28,60,23);
   });
  } else if(entry.table) {
   text(s,entry.points.join('  ·  '),72,218,1136,100,25,{color:C.sub});
   nativeTable(s,entry.table,72,345,1136,300);
  } else if(entry.formula_asset) {
   bullets(s,entry.points,72,240,522,28,entry.points.length>3?86:105);
   await picture(s,entry.formula_asset,640,228,568,entry.run?255:355);
   if(entry.diagram)flow(s,entry.diagram.labels,650,480,550,172);
   if(entry.run){
    rect(s,72,588,1136,95,C.panel);
    text(s,'>>> '+entry.run[0],94,600,1092,32,20,{mono:true});
    text(s,entry.run[1],94,641,1092,32,23,{mono:true,color:C.accent,bold:true});
   }
  } else if(entry.diagram) {
   bullets(s,entry.points,72,237,512,28,entry.points.length>3?91:110);
   flow(s,entry.diagram.labels,662,242,546,410);
  } else {
   entry.points.forEach((v,i)=>{
    const gap=entry.points.length>3?100:128;
    text(s,String(i+1).padStart(2,'0'),72,247+i*gap,80,50,25,{color:C.accent});
    text(s,v,184,237+i*gap,1000,gap-16,32);
   });
   if(entry.run){
    rect(s,72,592,1136,91,C.panel);
    text(s,'>>> '+entry.run[0]+'\n'+entry.run[1],94,602,1092,70,22,{mono:true,color:C.accent});
   }
  }
  pages.push({type:entry.table?'table':entry.diagram?'diagram':entry.image?'image':'theory',title:entry.title,image:entry.image || null,math:entry.math || null});
 }
 s=frame(p,l,'让代码验证你的判断','只运行指定短cell；完整训练已在课前完成。缺资产时使用已执行输出备用并说明。');
 rect(s,72,235,1136,326,C.code);
 text(s,l.code,100,255,1080,286,25,{mono:true,color:'#EAF0F7'});
 text(s,'先预测输出，再运行对应单元格。',72,592,1136,65,28,{color:C.sub});
 pages.push({type:'practice',title:'让代码验证你的判断'});
 s=frame(p,l,'停下来，做一次判断','停下来收集预测，发现错误后重新解释；不等到课后作业才知道学生没理解。\n作业：'+l.assignment+'；完整示例 / 填空 / 迁移；提交代码、配置、结果和限制。');
 text(s,'课堂预测',72,231,1136,40,20,{color:C.sub});
 text(s,l.change,72,290,1136,138,33);
 rect(s,72,466,1136,1,C.line);
 text(s,'退出题',72,492,1136,40,20,{color:C.sub});
 text(s,l.exit_question,72,549,1136,117,32,{bold:true,color:C.accent});
 pages.push({type:'exercise',title:'停下来，做一次判断'});
 s=frame(p,l,'结论要有边界','区分已观察事实、候选解释与下一项检验。\n退出题：'+l.exit_question);
 l.limits.forEach((v,i)=>{
  text(s,i===0?'已知':'边界',72,259+i*183,95,48,22,{color:C.accent});
  text(s,v,208,238+i*183,976,126,34);
  if(i===0)rect(s,208,394,976,1,C.line);
 });
 text(s,'来源、复现入口与辅助提示见讲者备注。',72,640,1136,35,18,{color:C.sub});
 pages.push({type:'limits',title:'结论要有边界'});
 const candidate=path.join(build,l.id+'-candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
 const skill=process.env.HSI_PRESENTATIONS_SKILL || path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
 const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
 const finalized=path.join(root,'results/slide_revision_final');await fs.mkdir(finalized,{recursive:true});
 const finalPath=path.join(finalized,l.id+'-final-'+Date.now()+'.pptx');
 await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
  pythonExecutable:path.join(runtime,'python/python.exe'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[FONT,'Consolas']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,l.id+'.validation-'+Date.now()+'.json')});
 const name=`${l.id}-${l.slug}.pptx`;
 await fs.copyFile(finalPath,path.join(output,name));
 await fs.copyFile(finalPath,path.join(root,'slides/decks',name));
 // PowerPoint COM provides final rendering on this Windows host. Artifact
 // export is also attempted for portable preview, but never faked if unavailable.
 const previews=path.join(root,'slides/previews/scientific-minimal',l.id+'-'+Date.now());await fs.mkdir(previews,{recursive:true});
 let rendered=0;let blocker;
 try {for(const [index,slide]of p.slides.items.entries()) {
  const png=await p.export({slide,format:'png',scale:1});await fs.writeFile(path.join(previews,String(index+1).padStart(2,'0')+'.png'),new Uint8Array(await png.arrayBuffer()));rendered++;
 }}catch(e){blocker=String(e);}
 manifest.push({id:l.id,title:l.title,path:'decks/'+name,slides:pages.length,pages,notebook:l.notebook,cells:l.cells,generated:true,rendered:rendered===pages.length,render_blocker:blocker,visual_review:'pending'});
 console.log(`Built ${name}: ${pages.length} slides, editable text/lines, rendered ${rendered}`);
}
const manifestPath=path.join(root,'slides/lesson-manifest.json');
let decks=manifest;
if(selected.length) {
 const previous=JSON.parse(await fs.readFile(manifestPath,'utf8'));
 const updated=new Map([...previous.decks,...manifest].map(d=>[d.id,d]));
 decks=data.lessons.map(l=>updated.get(l.id)).filter(Boolean);
}
await fs.writeFile(manifestPath,JSON.stringify({design:{engine:'@oai/artifact-tool',style:'科研极简：浅底、深蓝、大图、克制排版',version:2,fonts:{cjk:FONT,code:'Consolas'},source:'slides/src/course_content.py + classroom_content.py'},decks,note:'Package validation is separate from visual inspection and actual trial teaching.'},null,2));
