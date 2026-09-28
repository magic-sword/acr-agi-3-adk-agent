const fs=require('fs'),path=require('path');
const {chromium}=require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async()=>{
 const out=path.resolve(process.argv[2]),errors=[];
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1450,height:1100}});page.on('pageerror',e=>errors.push(String(e)));
  await page.goto('file://'+path.join(out,'gallery.html'));
  const cases=await page.locator('#case option').evaluateAll(xs=>xs.map(x=>x.value));
  for(const c of cases){await page.selectOption('#case',c);await page.waitForFunction(c=>document.body.dataset.caseReady===c,c);if(await page.locator('#answers .card').count()!==2)throw Error('Missing result arm')}
  const observations=await page.locator('#observation option').evaluateAll(xs=>xs.map(x=>x.value));
  for(const i of observations){await page.selectOption('#observation',i);await page.waitForFunction(i=>document.body.dataset.observationReady===i,i);await page.waitForFunction(()=>{const x=document.getElementById('live-image');return x.hidden||(x.complete&&x.naturalWidth>0)})}
  await page.selectOption('#case','motion_0_0');
  const choice=await page.locator('#observation option').evaluateAll(xs=>xs.find(x=>x.textContent.includes('ls20')&&x.textContent.includes('測定入力')&&x.textContent.includes('step 2'))?.value);
  if(choice)await page.selectOption('#observation',choice);
  await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
  if(cases.length!==29||errors.length)throw Error(JSON.stringify({cases:cases.length,errors}));
  await page.goto('file://'+path.join(out,'workflow/index.html'));
  if(await page.locator('[data-state="answer_question"]').count()!==1)throw Error('New state absent');
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,observations:observations.length,new_state_present:true,errors},null,2)+'\n');
  console.log('Verified',cases.length,'cases and',observations.length,'live observations');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
