"""Prompt sizing and wheel regression; isolated CPU test server only."""
from playwright.sync_api import sync_playwright
import json

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,channel='msedge')
    page=browser.new_page(viewport={'width':1800,'height':2600})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:8893')
    page.wait_for_function('()=>!!window.LiteGraph?.registered_node_types?.MiniMaxH3TimelineDirector')
    for vue in [False,True]:
        for kind in ['MiniMaxH3TimelineDirector','MiniMaxH3TimelinePlanner']:
            page.evaluate('''async(vue)=>{
              const {app}=await import('/scripts/app.js');window.a=app;
              await app.ui.settings.setSettingValue('Comfy.VueNodes.Enabled',vue);
            }''',vue)
            page.reload()
            page.wait_for_function('()=>!!window.LiteGraph?.registered_node_types?.MiniMaxH3TimelineDirector')
            page.evaluate('''async(kind)=>{
              const {app}=await import('/scripts/app.js');window.a=app;
              app.graph.clear();window.n=LiteGraph.createNode(kind);app.graph.add(n);
              n.pos=[30,30];app.canvas.ds.scale=.8;app.canvas.ds.offset=[0,0];
            }''',kind)
            page.wait_for_timeout(1500)
            global_prompt=page.locator('[data-global-prompt]')
            assert global_prompt.count()==0,(vue,kind,'global prompt input must not be rendered')
            if kind.endswith('Director'):
                native=page.locator('textarea:visible').first
                before={'height':native.bounding_box()['height']}
                if vue:
                    handle=page.locator('[data-corner="SE"]').bounding_box()
                    x=handle['x']+handle['width']/2;y=handle['y']+handle['height']/2
                    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+96,y+128,steps=15);page.mouse.up()
                else:
                    page.evaluate('''()=>{
                  a.canvas.resizing_node=n;
                  window.dispatchEvent(new PointerEvent('pointerdown',{clientY:100}));
                  window.dispatchEvent(new PointerEvent('pointermove',{clientY:260}));
                  n.setSize([n.size[0]+120,n.size[1]+160]);
                  window.dispatchEvent(new PointerEvent('pointerup'));
                  a.canvas.resizing_node=null;
                    }''')
                page.wait_for_timeout(800)
                after={'height':native.bounding_box()['height']}
                assert after['height']>before['height']+70,(vue,before,after)
                if vue:
                    handle=page.locator('[data-corner="SE"]').bounding_box()
                    x=handle['x']+handle['width']/2;y=handle['y']+handle['height']/2
                    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x-48,y-96,steps=15);page.mouse.up()
                else:
                    page.evaluate('''()=>{
                      a.canvas.resizing_node=n;
                      window.dispatchEvent(new PointerEvent('pointerdown',{clientY:260}));
                      window.dispatchEvent(new PointerEvent('pointermove',{clientY:140}));
                      n.setSize([n.size[0]-60,n.size[1]-120]);
                      window.dispatchEvent(new PointerEvent('pointerup'));
                      a.canvas.resizing_node=null;
                    }''')
                page.wait_for_timeout(800)
                shrunk={'height':native.bounding_box()['height']}
                assert shrunk['height']<after['height']-50,(vue,before,after,shrunk)
                assert shrunk['height']>=60,(vue,shrunk)
                stable=page.evaluate('()=>({prompt:n.properties.m3tdPromptHeight,size:[...n.size]})')
                page.mouse.click(1650,1500)
                page.wait_for_timeout(800)
                clicked=page.evaluate('()=>({prompt:n.properties.m3tdPromptHeight,size:[...n.size]})')
                assert abs(clicked['prompt']-stable['prompt'])<1,(vue,'click changed prompt',stable,clicked)
                assert abs(clicked['size'][1]-stable['size'][1])<2,(vue,'click changed node height',stable,clicked)
            page.locator('[data-field="segmentCount"]').fill('3')
            page.locator('[data-action="applySegments"]').click()
            page.wait_for_timeout(500)
            inputs=[page.locator('[data-segment-prompt]')]
            if kind.endswith('Planner'):
                inputs.append(global_prompt)
            if kind.endswith('Director'):
                inputs.append(page.locator('textarea:visible').first)
            for input in inputs:
                input.fill('\n'.join(f'Prompt line {i}' for i in range(150)))
                input.click()
                input.evaluate('(e)=>{e.scrollTop=0;}')
                before=page.evaluate('a.canvas.ds.scale')
                box=input.bounding_box()
                page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2)
                page.wait_for_timeout(250)
                page.mouse.wheel(0,250)
                page.wait_for_timeout(200)
                assert input.evaluate('(e)=>e.scrollTop')>0,(vue,kind,'no scroll',box,page.evaluate('([x,y])=>document.elementFromPoint(x,y)?.outerHTML.slice(0,250)',[box['x']+box['width']/2,box['y']+box['height']/2]))
                assert page.evaluate('a.canvas.ds.scale')==before,(vue,kind,'canvas zoomed',before,page.evaluate('a.canvas.ds.scale'),input.evaluate('(e)=>({html:e.outerHTML.slice(0,250),focused:e===document.activeElement,parent:e.parentElement.outerHTML.slice(0,250)})'))
            if kind.endswith('Planner'):
                assert page.locator('.m3td-global-prompt.inactive').count()==1,(vue,kind,'global prompt mode indicator missing')
            page.locator('[data-field="segmentCount"]').fill('0')
            page.locator('[data-action="applySegments"]').click()
            page.wait_for_timeout(500)
            page.evaluate('()=>{n=a.graph.getNodeById(n.id)}')
            page.evaluate('()=>{a.canvas.ds.offset=[0,0];a.canvas.setDirty(true,true)}')
            page.wait_for_timeout(500)
            assert page.evaluate('Math.abs(n.__m3td.root.offsetHeight-n.__m3td.requiredDirectorHeight())')<12,page.evaluate('({root:n.__m3td.root.offsetHeight,required:n.__m3td.requiredDirectorHeight(),size:n.size})')
            print(json.dumps({'nodes2':vue,'node':kind,'passed':True}))
    assert not errors,errors
    browser.close()
