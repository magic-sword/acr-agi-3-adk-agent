const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
 const out=path.resolve(process.argv[2]),errors=[];
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1150,height:960}});
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file://'+path.join(out,'gallery.html'));
  await page.waitForFunction(()=>document.getElementById('info').textContent.includes('生成した分類'));
  const counts=await page.evaluate(()=>({cases:document.getElementById('case').options.length,heads:document.getElementById('head').options.length,layers:document.getElementById('layer').options.length}));
  if(counts.cases!==5||counts.heads!==33||counts.layers!==4)throw Error(JSON.stringify(counts));
  await page.selectOption('#case','cross_1px');await page.selectOption('#layer','23');await page.selectOption('#head','7');
  await page.waitForFunction(()=>document.getElementById('info').textContent.includes('対象ID: B'));
  await page.selectOption('#head','avg');
  await page.screenshot({path:path.join(out,'attention-preview.png'),fullPage:true});
  await page.selectOption('#mode','features');
  if(await page.locator('#layer option').count()!==9)throw Error('feature stages');
  await page.selectOption('#layer','merger');
  await page.screenshot({path:path.join(out,'features-preview.png'),fullPage:true});
  await page.uncheck('#overlay');await page.check('#overlay');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,counts,feature_stages:9,errors},null,2)+'\n');
  console.log('Interactive gallery verified');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
