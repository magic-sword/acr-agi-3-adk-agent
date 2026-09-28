const fs=require('fs');const path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
  const b=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  const p=await b.newPage({viewport:{width:640,height:820},deviceScaleFactor:1});
  await p.route('**/*',r=>r.abort());
  await p.setContent(fs.readFileSync('outputs/retro-ui-20260927/assets/renderer.html','utf8'));
  await p.evaluate(()=>document.fonts.ready);
  const result=await p.evaluate(()=>{
    const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};};
    const buttons=[...document.querySelectorAll('.d-pad-grid .tactile-button')];
    const click=[...document.querySelectorAll('.control-group')].find(e=>e.querySelector('.control-label')?.textContent.trim()==='CLICK');
    return {up:rect(buttons[0]),click:rect(click.querySelector('.tactile-button'))};
  });
  fs.writeFileSync(process.argv[2],JSON.stringify(result,null,2)+'\n');console.log(result);await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
