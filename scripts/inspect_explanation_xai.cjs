const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{const out=path.resolve(process.argv[2]),errors=[];const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
try{const page=await browser.newPage({viewport:{width:1500,height:1100}});page.on('pageerror',e=>errors.push(String(e)));await page.goto('file://'+path.join(out,'gallery.html'));
const cases=await page.locator('#case option').evaluateAll(xs=>xs.map(x=>x.value));let views=0;
for(const c of cases){await page.selectOption('#case',c);const fields=await page.locator('#field option').evaluateAll(xs=>xs.map(x=>x.value));for(const f of fields){await page.selectOption('#field',f);for(const l of ['1','12','24','36']){await page.selectOption('#layer',l);await page.waitForFunction(x=>document.body.dataset.ready===x,[c,f,l].join('|'));if(await page.locator('#mass tbody tr').count()<5)throw Error('missing attribution');views++}}}
await page.selectOption('#case','block_1px');await page.selectOption('#field','before_position_start');await page.selectOption('#layer','24');await page.waitForFunction(()=>document.body.dataset.ready==='block_1px|before_position_start|24');await page.uncheck('#overlay');await page.check('#overlay');await page.screenshot({path:path.join(out,'analysis-preview.png'),fullPage:true});
if(errors.length||views!==64)throw Error(JSON.stringify({errors,views}));fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,views,errors},null,2)+'\n');console.log('Verified',views,'attribution views');
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
