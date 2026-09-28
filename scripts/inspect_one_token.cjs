const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
  const out=path.resolve(process.argv[2]),errors=[];
  const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file://'+path.join(out,'gallery.html'));
    const cases=await page.locator('#case option').evaluateAll(xs=>xs.map(x=>x.value));
    for(const c of cases){await page.selectOption('#case',c);await page.waitForFunction(c=>document.body.dataset.caseReady===c,c);if(await page.locator('#answers .card').count()!==2)throw Error('Missing result source')}
    const groups=await page.locator('#generated option').evaluateAll(xs=>xs.map(x=>x.value));let questions=0;
    for(const g of groups){
      await page.selectOption('#generated',g);const items=await page.locator('#item option').evaluateAll(xs=>xs.map(x=>x.value));
      for(const i of items){await page.selectOption('#item',i);await page.waitForFunction(key=>document.body.dataset.generatedReady===key,g+'|'+i);questions++}
    }
    await page.selectOption('#case','motion_0_0');await page.selectOption('#generated','4');await page.selectOption('#item','0');
    await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
    if(cases.length!==29||groups.length!==33||questions!==228||errors.length)throw Error(JSON.stringify({cases:cases.length,groups:groups.length,questions,errors}));
    fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:29,groups:33,questions,errors},null,2)+'\n');console.log('Verified 29 benchmark cases and 228 generated questions');
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
