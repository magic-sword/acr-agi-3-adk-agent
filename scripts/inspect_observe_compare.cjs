const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
 const out=path.resolve(process.argv[2]),errors=[];
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1400,height:1000}});page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file://'+path.join(out,'gallery.html'));
  await page.waitForFunction(()=>document.getElementById('direct').textContent.includes('tokens'));
  const cases=await page.locator('#case option').evaluateAll(os=>os.map(o=>o.value));
  for(const c of cases){await page.selectOption('#case',c);for(const b of ['llama','official']){
    await page.selectOption('#backend',b);await page.waitForFunction(()=>['before','after'].every(id=>document.getElementById(id).complete&&document.getElementById(id).naturalWidth===640));
    const reps=await page.locator('#repeat option').evaluateAll(os=>os.filter(o=>!o.disabled).map(o=>o.value));
    for(const r of reps){await page.selectOption('#repeat',r);if(!(await page.locator('#evidence').textContent()).includes('tokens'))throw Error('missing result')}
  }}
  await page.selectOption('#case','color_only');await page.selectOption('#backend','official');
  await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,errors},null,2)+'\n');console.log('Verified paired gallery',cases.length,'cases');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
