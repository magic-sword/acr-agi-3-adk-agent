const fs = require('fs'),path = require('path');
const {chromium} = require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
  const out=path.resolve(process.argv[2]),errors=[];
  const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(out,'gallery.html'));
    const cases=await page.locator('#case option').evaluateAll(xs=>xs.map(x=>x.value));
    for(const c of cases){
      await page.selectOption('#case',c);
      await page.waitForFunction(c=>document.body.dataset.ready===c&&[...document.images].every(im=>im.complete&&im.naturalWidth===512),c);
      if(await page.locator('#answers .card').count()!==3)throw Error('missing answer');
    }
    await page.selectOption('#case','motion_0_0');
    await page.waitForFunction(()=>document.body.dataset.ready==='motion_0_0');
    await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
    if(cases.length!==29||errors.length)throw Error(JSON.stringify({cases,errors}));
    fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:29,arms:3,errors},null,2)+'\n');
    console.log('Verified 29 cases, 3 arms, and all images');
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
