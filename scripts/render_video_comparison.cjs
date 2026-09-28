// Generate intermediate RGB blends with the existing canvas renderer.
const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
 const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();await page.route('http://**/*',r=>r.abort());await page.route('https://**/*',r=>r.abort());
  await page.goto('file://'+path.resolve('outputs/temporal-overlay-preview-20260928/preview.html'));
  const ui=fs.readFileSync('outputs/action-cue-20260927/ui-template.png').toString('base64');
  const results=await page.evaluate(async ui=>{
   const shell=new Image();shell.src='data:image/png;base64,'+ui;await shell.decode();const all=[];
   document.getElementById('dim').value=1;document.getElementById('mix').step='any';
   for(const c of window.overlayPreview.cases)for(const [name,alpha] of [['mix1',1/3],['mix2',2/3]]){
    document.getElementById('mix').value=alpha;
    const board=document.createElement('canvas');window.overlayPreview.paint(board,'blend',c,8);
    const canvas=document.createElement('canvas');canvas.width=640;canvas.height=820;
    const ctx=canvas.getContext('2d');ctx.drawImage(shell,0,0);ctx.drawImage(board,59,80);
    all.push({name:c.name+'-'+name,png:canvas.toDataURL('image/png').split(',')[1]});
   }return all;
  },ui);
  for(const r of results)fs.writeFileSync(path.join(out,r.name+'.png'),Buffer.from(r.png,'base64'));
  console.log('Rendered',results.length,'intermediate frames');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
