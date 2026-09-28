const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{const out=path.resolve(process.argv[2]),errors=[],plan=JSON.parse(fs.readFileSync(path.join(out,'plan.json'))),expectedPanels=plan.arms.length/2;const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
try{const page=await browser.newPage({viewport:{width:1400,height:1000}});page.on('pageerror',e=>errors.push(String(e)));await page.goto('file://'+path.join(out,'gallery.html'));
const cases=await page.locator('#case option').evaluateAll(os=>os.map(o=>o.value));if(cases.length*expectedPanels*2!==plan.requests)throw Error('case count');
for(const c of cases){await page.selectOption('#case',c);for(const b of ['llama','official']){await page.selectOption('#backend',b);await page.waitForFunction(()=>['before','after','edges'].every(id=>document.getElementById(id).complete&&document.getElementById(id).naturalWidth===640));if(await page.locator('.card').count()!==expectedPanels)throw Error('missing panels')}}
await page.selectOption('#case','cross_1px');await page.selectOption('#backend','llama');await page.check('#all');await page.uncheck('#all');await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
if(errors.length)throw Error(errors.join('\n'));fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,panels:expectedPanels,errors},null,2)+'\n');console.log('Verified comparison gallery');
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
