const fs = require('fs'), path = require('path');
const {chromium} = require('/home/prog/.cache/ms-playwright-go/1.57.0/package');
(async () => {
  const out = path.resolve(process.argv[2]), errors = [];
  const browser = await chromium.launch({executablePath:'/usr/bin/google-chrome', headless:true, args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1500,height:1100}});
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto('file://'+path.join(out,'gallery.html'));
    const cases = await page.locator('#case option').evaluateAll(xs => xs.map(x=>x.value));
    for (const c of cases) {
      await page.selectOption('#case', c);
      await page.waitForFunction(c => document.body.dataset.ready === c && [...document.images].every(im=>im.complete && im.naturalWidth === 592), c);
      if (await page.locator('#answers .card').count() !== 5) throw Error('missing comparison arms');
    }
    await page.selectOption('#case','panels_0');
    await page.waitForFunction(()=>document.body.dataset.ready === 'panels_0');
    await page.screenshot({path:path.join(out,'comparison-preview.png'),fullPage:true});
    if (errors.length || cases.length !== 14) throw Error(JSON.stringify({errors,cases}));
    fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({passed:true,cases:cases.length,arms:5,errors},null,2)+'\n');
    console.log('Verified',cases.length,'cases, 5 arms and all frame images');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
