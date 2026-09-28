// Deterministic HTML/canvas rendering from the public ARC Prize shell and frozen boards.
const fs = require('fs');
const path = require('path');
const {chromium} = require('/home/prog/.cache/ms-playwright-go/1.57.0/package');

(async () => {
  const output=path.resolve(process.argv[2]);
  const assets=path.join(output,'assets');
  const cases=JSON.parse(fs.readFileSync(path.join(output,'cases.json'),'utf8'));
  let css=fs.readFileSync(path.join(assets,'source.css'),'utf8');
  const font=fs.readFileSync(path.join(assets,'TronicaMono.otf')).toString('base64');
  const texture=fs.readFileSync(path.join(assets,'brushed-metal.png')).toString('base64');
  css=css.replaceAll('/gameplay/images/brushed-metal.png','data:image/png;base64,'+texture)
         .replaceAll('/gameplay/fonts/TronicaMono.otf','data:font/otf;base64,'+font);
  const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
  const page=await browser.newPage({viewport:{width:640,height:820},deviceScaleFactor:1});
  await page.route('**/*',route=>route.abort());
  await page.setContent('<html><body></body></html>');
  const source=fs.readFileSync(path.join(assets,'source.html'),'utf8');
  const shell=await page.evaluate(source=>{
    const doc=new DOMParser().parseFromString(source,'text/html');
    return doc.querySelector('.shell-root').outerHTML;
  },source);
  const override=`html,body{margin:0!important;padding:0!important;width:640px!important;height:820px!important;background:#0e0c0d!important;overflow:hidden}
    *,*::before,*::after{animation:none!important;transition:none!important}
    .shell-root{position:absolute!important;left:20px;top:20px;width:600px!important;max-width:none!important;}
    .shell-screen-outer{width:544px!important;max-width:none!important;height:543px!important;aspect-ratio:auto!important;flex:none!important}
    .shell-screen-inner{border-radius:0!important;box-shadow:none!important}
    .screen-root{display:block!important;width:512px!important;height:512px!important;min-height:512px!important;overflow:visible!important}
    #frozen-board{display:block;width:512px;height:512px;image-rendering:pixelated}
    .shell-header-inner{max-width:none!important}
    .control-label{font-family:'Tronica Mono',monospace!important}`;
  await page.setContent('<!doctype html><meta charset="utf-8"><style>'+css+'</style><style>'+override+'</style>'+shell);
  await page.evaluate(()=>{
    document.querySelector('.screen-root').innerHTML='<canvas id="frozen-board" width="512" height="512"></canvas>';
    document.querySelector('.handheld-title').textContent='GAME';
    document.querySelector('.shell-level-badge').textContent='LEVEL 1 / 7';
    // Neutral, constant labels across real and synthetic cases; no task ID or action hint.
    for (const el of document.querySelectorAll('button')) el.disabled=false;
    const board=document.querySelector('#frozen-board');
    const r=board.getBoundingClientRect();board.style.transform=`translate(${Math.round(r.x)-r.x}px,${Math.round(r.y)-r.y}px)`;
  });
  await page.evaluate(()=>document.fonts.ready);
  const geometry=await page.evaluate(()=>{
    const r=document.querySelector('#frozen-board').getBoundingClientRect();
    return {x:r.x,y:r.y,width:r.width,height:r.height,canvas_width:640,canvas_height:820,
      shell_height:document.querySelector('.shell-root').getBoundingClientRect().height};
  });
  if(!Number.isInteger(geometry.x)||!Number.isInteger(geometry.y)||geometry.width!==512||geometry.height!==512||geometry.shell_height>780) throw Error(JSON.stringify(geometry));
  fs.writeFileSync(path.join(output,'geometry.json'),JSON.stringify(geometry,null,2)+'\n');
  fs.writeFileSync(path.join(assets,'renderer.html'),await page.content());
  await page.evaluate(()=>{
    const ctx=document.querySelector('#frozen-board').getContext('2d');ctx.fillStyle='#333333';ctx.fillRect(0,0,512,512);
  });
  const template=await page.screenshot({path:path.join(output,'blank-shell.png')});
  // Freeze the HTML UI once: browser repainting otherwise changes header antialiasing.
  // Canvas export keeps the entire surrounding UI byte-identical across conditions.
  await page.evaluate(async data=>{
    window.uiTemplate=new Image();window.uiTemplate.src='data:image/png;base64,'+data;await window.uiTemplate.decode();
  },template.toString('base64'));
  for (const c of cases){
    for (const frame of ['before','after']){
      for (const grid of [false,true]){
        const png=await page.evaluate(({pixels,palette,grid,geometry})=>{
          const canvas=document.createElement('canvas');canvas.width=640;canvas.height=820;
          const ctx=canvas.getContext('2d');ctx.drawImage(window.uiTemplate,0,0);ctx.save();ctx.translate(geometry.x,geometry.y);
          for(let y=0;y<64;y++) for(let x=0;x<64;x++){
            ctx.fillStyle=palette[pixels[y][x]];ctx.fillRect(x*8,y*8,8,8);
          }
          if(grid){
            ctx.fillStyle='rgba(0,0,0,0.15)';
            for(let p=0;p<512;p+=8){ctx.fillRect(p,0,1,512);ctx.fillRect(0,p,512,1);}
          }
          ctx.restore();return canvas.toDataURL('image/png').split(',')[1];
        },{pixels:c[frame],palette:c.palette,grid,geometry});
        fs.writeFileSync(path.join(output,`${c.name}-${frame}-${grid?'shell_grid':'shell'}.png`),Buffer.from(png,'base64'));
      }
    }
  }
  console.log(JSON.stringify({rendered:cases.length*4+1,geometry}));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
