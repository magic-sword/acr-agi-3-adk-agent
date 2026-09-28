const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{const out=path.resolve(process.argv[2]),errors=[];const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
try{const page=await browser.newPage({viewport:{width:1500,height:1100}});page.on('pageerror',e=>errors.push(String(e)));await page.goto('file://'+path.join(out,'gallery.html'));
const cases=await page.locator('#case option').evaluateAll(xs=>xs.map(x=>x.value));let views=0;
for(const c of cases){await page.selectOption('#case',c);for(const s of ['measured','vlm']){await page.selectOption('#source',s);await page.waitForFunction(v=>document.body.dataset.ready===v&&['before','after'].every(id=>document.getElementById(id).complete&&document.getElementById(id).naturalWidth===592),c+'|'+s);if(await page.locator('#judgments .card').count()!==4)throw Error('missing judgments');views++}}
await page.selectOption('#case','four_panels_change');await page.selectOption('#source','measured');await page.waitForFunction(()=>document.querySelectorAll('.pattern').length===8);await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
if(errors.length||views!==20)throw Error(JSON.stringify({errors,views}));fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,views,errors},null,2)+'\n');console.log('Verified',views,'comparison views');
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
