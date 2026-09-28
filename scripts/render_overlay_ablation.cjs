// Same frozen UI and renderer; isolate background dimming and RGB mixing.
const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
 const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();
  await page.route('http://**/*',r=>r.abort());await page.route('https://**/*',r=>r.abort());
  await page.goto('file://'+path.resolve('outputs/temporal-overlay-preview-20260928/preview.html'));
  const ui=fs.readFileSync('outputs/action-cue-20260927/ui-template.png').toString('base64');
  const results=await page.evaluate(async ui=>{
   const image=new Image();image.src='data:image/png;base64,'+ui;await image.decode();
   const all=[];
   for(const c of window.overlayPreview.cases)for(const [mode,dim,mix] of [['outline_bright',1,.5],['after_edges',1,1]]){
    document.getElementById('dim').value=dim;document.getElementById('mix').value=mix;
    document.getElementById('radius').value=1;
    const board=document.createElement('canvas');window.overlayPreview.paint(board,'outline',c,8);
    const canvas=document.createElement('canvas');canvas.width=640;canvas.height=820;
    const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);ctx.drawImage(board,59,80);
    all.push({name:c.name,mode,png:canvas.toDataURL('image/png').split(',')[1]});
   }return all;
  },ui);
  for(const r of results)fs.writeFileSync(path.join(out,r.name+'-'+r.mode+'.png'),Buffer.from(r.png,'base64'));
  console.log('Rendered '+results.length+' ablation images');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
