const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
  const out=path.resolve(process.argv[2]),errors=[];
  const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(out,'gallery.html'));
    const scenes=await page.locator('#scene option').evaluateAll(xs=>xs.map(x=>x.value));let subjects=0;
    for(const scene of scenes){
      await page.selectOption('#scene',scene);
      const ids=await page.locator('#subject option').evaluateAll(xs=>xs.map(x=>x.value));
      if(!ids.length){await page.waitForFunction(k=>document.body.dataset.ready===k,scene+'|none');if(await page.locator('#answers .card').count())throw Error('Stale cards')}
      for(const id of ids){await page.selectOption('#subject',id);await page.waitForFunction(k=>document.body.dataset.ready===k,scene+'|'+id);if(await page.locator('#answers .card').count()!==4)throw Error('Missing arm');subjects++}
      await page.waitForFunction(()=>['before','after'].every(id=>{const i=document.getElementById(id);return i.complete&&i.naturalWidth>0}));
    }
    await page.selectOption('#scene','motion_3_2_T1');
    await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
    if(scenes.length!==41||subjects!==87||errors.length)throw Error(JSON.stringify({scenes:scenes.length,subjects,errors}));
    fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,scenes:scenes.length,subjects,errors},null,2)+'\n');
    console.log('Verified 41 scenes and 87 subjects across four arms');
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
